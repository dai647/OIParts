import numpy as np
import torch
import PIL.Image as Image
import cv2
from PIL.Image import Resampling
import torch.nn.functional as F
import os
import  bilateral_solver
import time

face_dict = {
0:(0,0,0),     
1:(99,60,4),   
2:(203,220,1),  
3:(21,198,193), 
4:(204,44,193), 
5:(1,94,124),   
6:(129,0,67),   
7:(204,189,174),
8:(203,63,52),  
9:(0,118,49),   
}

def viz(label, h,w,x_dict):
    result = np.zeros((h * w, 3), dtype=np.uint8)
    for pixel in range(len(label)):
        result[pixel] = x_dict[label[pixel]]
    result = result.reshape((h, w, 3))
    return result

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  
    return torch.stack(result_list, dim=2) 


def dilate_and_erode(mask_data, struc="ELLIPSE", size=(10, 10)):

    if struc == "RECT":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CORSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    msk = mask_data / 255

    dilated = cv2.dilate(msk, kernel, iterations=1) * 255
    eroded = cv2.erode(msk, kernel, iterations=1) * 255
    res = dilated.copy()
    res[((dilated == 255) & (eroded == 0))] = 128
    return res

  
def get_novel_sigle_mask(save_p,kk,base_feature_fpath,base_mask_fpath,novel_feature_fpath,novel_gt_mask_fpath,base_file_name,novel_file_name,channels):
    
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        descriptors2=torch.load(novel_feature_fpath+'/'+novel_file_name[:-4]+'_'+'fuse.pth')
        
        base_mask_path=base_mask_fpath+'/'+base_file_name[:-4]+'.npy'
        base_mask=np.load(base_mask_path)
        
        base_class_ids=torch.tensor(np.array(base_mask)).unique().cpu().numpy()

        base_masks=np.array(Image.fromarray(base_mask).resize((512,512),resample=Resampling.NEAREST))

        img_path=novel_gt_mask_fpath+'/'+novel_file_name
       
        descriptors1=torch.load(base_feature_fpath+'/'+base_file_name[:-4]+'_'+'fuse.pth')
        part_masks=[]
        for cid in range(len(base_class_ids)):
            cid=int(cid)
            
            if cid ==0:
                continue
            
            base_mask=np.zeros_like(base_masks)
            base_mask[base_masks==cid]=1
                
            resize_base_mask = cv2.resize(base_mask.astype(np.float64), 
                                            dsize=(60,60)).astype(float)
            
            base_mask_f = torch.tensor(resize_base_mask.reshape(-1), device= device).int()

            sim=chunk_cosine_sim(descriptors1[:,:,:,channels[cid]], descriptors2[:,:,:,channels[cid]])
            similarities=(sim-sim.min())/(sim.max()-sim.min())
            
            ########KNN
  
            topk_v ,topk_indices= torch.topk(similarities[0,0], k=kk,dim=-2,largest=True)
            topk_indices=topk_indices.reshape(-1)
            novel_knn=base_mask_f[topk_indices].float()
            novel_knn=novel_knn.reshape(kk,3600)
            belta=.1
            alpha=torch.softmax(topk_v/belta,dim=0)
            novel_mask_f=(alpha*novel_knn).sum(0)
            novel_mask = novel_mask_f.reshape(60,60)
            
            [h,w]=base_mask.shape

            resize_novel_mask = F.interpolate(torch.tensor(novel_mask[None][None]).float().to(descriptors1.device), size=(h, w),
                                            mode='bilinear', align_corners=False)[0,0].cpu().numpy()

            part_masks.append(resize_novel_mask)

        mx=np.concatenate([part_masks],0)
        full_mask=np.argmax(mx,0)
        mm=np.max(mx,0)
        full_mask+=1
        full_mask[mm<0.1]=0

        ########################FBS
        res_m=[]
        res_m.append(np.zeros_like(base_mask))
        for cid in range(len(base_class_ids)):
            if cid==0:
                continue
            cmask=np.zeros_like(full_mask)
            cmask[full_mask==cid]=1

            contours, _ = cv2.findContours(cmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
            
            size=int(6-2*int(((100*max_area/(512**2))>1)))
            cmask= dilate_and_erode(cmask*255, size=(size, size))
            cmask[cmask>0]=1
    
            cp,cm=bilateral_solver.bilateral_solver_output(img_path,cmask,10,9,3,0.5)
            res_m.append(cm)
        part_area={}
        for k in range(len(base_class_ids)):
            part_area[k]=np.sum(res_m[k])
        sort_k=sorted(part_area.items(),key=lambda x: -x[1])
 
        sort_k=[x[0] for x in sort_k]
        full_mask=np.zeros_like(base_mask)
        for k in sort_k:
            if k==0:
                continue
            full_mask[res_m[k]==1]=k

        novelmask2=viz(full_mask.reshape(-1).astype(np.int64),512, 512,face_dict)
        novel_m2 = Image.fromarray( novelmask2 )
        save_p=save_p+'/'+novel_file_name[:-4]+'_'+'mask.png'
        novel_m2.save(save_p)            

            
     


def get_masks():
    base_feature_fpath='Features/face/ref'
    base_mask_fpath='datasets/celeba/train_1'
    novel_feature_fpath='Features/face/query'
    novel_gt_mask_fpath='datasets/celeba/test'
    save_p='Result/face'
    if not os.path.exists(save_p):
        os.makedirs(save_p)
    base_file_name='152.png'
    kk=9  
    dino_min_remain_ratio=0.85
    sd_min_remain_ratio=0.85
    cidp='Select_IDs/face/face_dino_'+str(dino_min_remain_ratio)+'_sd_'+str(sd_min_remain_ratio)
    channels=torch.load(cidp) 
    novel_file_names=os.listdir(novel_gt_mask_fpath)

    nn=0
    for novel_file_name in novel_file_names:
        if novel_file_name[-4:]=='.npy':
            continue
        print("-----num:",nn+1,'-----') 
        get_novel_sigle_mask(save_p,kk,base_feature_fpath,base_mask_fpath,novel_feature_fpath,
                            novel_gt_mask_fpath,base_file_name,novel_file_name,channels)
        nn+=1

            
if __name__ =='__main__':

    get_masks()

  
        
        

        
