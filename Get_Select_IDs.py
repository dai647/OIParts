import torch
import numpy as np
import cv2
import os
def IoU(prediction, mask):
    intersection = prediction * mask
    union = prediction + mask - intersection
    return intersection.sum() / (union.sum() + 1e-7)

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y)) 
    return torch.stack(result_list, dim=2)  


def selec_ch(fpath,mpath,selec_num,feature_size=(60,60)):
    with torch.no_grad():

        img_feature=torch.load(fpath)[0,0]
        mask1=np.load(mpath)
        class_channels=[]

        for i in range(len(torch.tensor(mask1).unique())):

            mask=np.load(mpath)
            if i==0:
                mask[mask!=0]=1

            else:
                mask[mask!=i]=0
                mask[mask!=0]=1
            resize_base_mask = cv2.resize(mask.astype(np.float64), 
                                            dsize=feature_size).astype(int)

            Score=torch.zeros(img_feature.shape[1]).to(img_feature.device)
                                
            va=torch.zeros(img_feature.shape[1]).to(img_feature.device)
            for ii in range(img_feature.shape[1]):
                va[ii]+=(torch.var(img_feature[:,ii][resize_base_mask.reshape(-1)==1])+
                        torch.var(img_feature[:,ii][resize_base_mask.reshape(-1)==0]))/2
                
            Score -= ((va-va.min())/(va.max()-va.min()))

            select_channes=torch.topk(Score,selec_num)[1]
            class_channels.append(select_channes)

        return class_channels


def score_cluster(fpath,mpath,chs,cid,num):
    with torch.no_grad():

        feature_size=(60,60)
        img_feature=torch.load(fpath)

        mask=np.load(mpath)
        if cid==0:
            mask[mask!=0]=1

        else:
            mask[mask!=cid]=0
            mask[mask!=0]=1
        resize_base_mask = cv2.resize(mask.astype(np.float64), 
                                        dsize=feature_size).astype(int).reshape(-1)
        
        resize_base_mask=torch.tensor(resize_base_mask).to(img_feature.device)
        mm=resize_base_mask.reshape(-1)
        ids=chs[cid]
        ff=img_feature[:,:,:,ids[:num]]
        backf=ff[:,:,mm==0,:]
        foref=ff[:,:,mm==1,:]
        
        back_av=backf.mean(dim=2,keepdim=True)
        fore_av=foref.mean(dim=2,keepdim=True)
        
        sim_b=chunk_cosine_sim(back_av, ff)[0,0]
        sim_f=chunk_cosine_sim(fore_av, ff)[0,0]
        
        sim=torch.cat((sim_b,sim_f),dim=0)
        pmask=torch.argmax(sim,dim=0)

        loss=0-IoU(pmask,resize_base_mask)
        return loss
    
    
def adaptive_select(dino_min_remain_ratio,sd_min_remain_ratio):

    id2name={0:'Background',1:'Face',2:'Eye',3:'Mouth',4:'Nose',5:'Eyebrow',6:'Ear',7:'Neck',8:'Cloth',9:'Hair'}
    class_num=len(id2name.keys())

    fpath4dino='Features/face/ref/152_dino.pth'
    fpath4sd='Features/face/ref/152_sd.pth'
    mpath='datasets/celeba/train_1/152.npy'
    
    chs2sd=selec_ch(fpath4sd,mpath,1024)
    chs2dino=selec_ch(fpath4dino,mpath,768)
    
    ##############for sd
    res2sd=[1e10 for i in range(class_num)]
    idxs2sd=[1024 for i in range(class_num)]
    for i in range(0,1024):
        ii=1024-i
        for cid in range(class_num):
            score1=score_cluster(fpath4sd,mpath,chs2sd,cid,ii)
            if res2sd[cid]>score1:
                res2sd[cid]=score1
                idxs2sd[cid]=ii
        if ii==int(1024*sd_min_remain_ratio):
            break
        
    ##############for dino
    res2dino=[1e10 for i in range(class_num)]
    idxs2dino=[768 for i in range(class_num)]
    for i in range(0,768):
        ii=768-i
        for cid in range(class_num):
            score1=score_cluster(fpath4dino,mpath,chs2dino,cid,ii)
            if res2dino[cid]>score1:
                res2dino[cid]=score1
                idxs2dino[cid]=ii
        if ii==int(768*dino_min_remain_ratio):
            break

    saves=[]
    for cid in range(class_num):
        saves.append(torch.cat([chs2sd[cid][:idxs2sd[cid]],1024+chs2dino[cid][:idxs2dino[cid]]],dim=0))
    print('dino:',idxs2dino,'sd:',idxs2sd)
    return saves

if __name__=='__main__':
    
    dataset_type='face'
    dino_min_remain_ratio=0.85
    sd_min_remain_ratio=0.85

    print(dataset_type,'dino:'+str(dino_min_remain_ratio),'sd:'+str(sd_min_remain_ratio))
    saves=adaptive_select(dino_min_remain_ratio,sd_min_remain_ratio)
     
    saves_path='Select_IDs/'+dataset_type+'/'+dataset_type+'_dino_'+str(dino_min_remain_ratio)+'_sd_'+str(sd_min_remain_ratio)
    if not os.path.exists('Select_IDs/'+dataset_type):
        os.makedirs('Select_IDs/'+dataset_type)
    torch.save(saves,saves_path)
    print('save to: ',saves_path)
    print('---------------Select Ids Have saved--------------------')
