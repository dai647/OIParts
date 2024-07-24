from Get_features import SD_DINO 
import torch
import os

def save_base_f(model,img_f_path,img_file_name,prompt,save_path):
    save_f_path=save_path+'/ref'
    if not os.path.exists(save_f_path):
        os.makedirs(save_f_path)
    img_path=img_f_path+'/'+img_file_name
    sd_f=model.get_sd_f(img_path,prompt)
    dino_f=model.get_dino_f(img_path)
    fuse_f=model.get_fuse_f(sd_f,dino_f)
    save_path_dino=save_f_path+'/'+img_file_name[:-4]+'_dino.pth'
    save_path_sd=save_f_path+'/'+img_file_name[:-4]+'_sd.pth'
    save_path_fuse=save_f_path+'/'+img_file_name[:-4]+'_fuse.pth'
    torch.save( dino_f,save_path_dino)
    torch.save( sd_f,save_path_sd)
    torch.save( fuse_f,save_path_fuse)

def save_novel_f(model,img_f_path,img_file_name,prompt,save_path):
    save_f_path=save_path+'/query'
    if not os.path.exists(save_f_path):
        os.makedirs(save_f_path)
    img_path=img_f_path+'/'+img_file_name
    sd_f=model.get_sd_f(img_path,prompt)
    dino_f=model.get_dino_f(img_path)
    fuse_f=model.get_fuse_f(sd_f,dino_f)
    save_path_dino=save_f_path+'/'+img_file_name[:-4]+'_dino.pth'
    save_path_sd=save_f_path+'/'+img_file_name[:-4]+'_sd.pth'
    save_path_fuse=save_f_path+'/'+img_file_name[:-4]+'_fuse.pth'

    torch.save( dino_f,save_path_dino)
    torch.save( sd_f,save_path_sd)
    torch.save( fuse_f,save_path_fuse)
    

def save_all(img_path,save_path,model,prompt,only_base=False):
    base_img_path=img_path+'/train_1'
    bases=os.listdir(base_img_path)
    nn=0
    for base_file_name in bases:
        if base_file_name[-4:] == '.npy':
            continue
        print("---------ref No:",nn+1,'----------')
        save_base_f(model,base_img_path,base_file_name,prompt,save_path)
        nn+=1
    if only_base:
        return 
    novel_img_path=img_path+'/test'
    novels=os.listdir(novel_img_path)
    nn=0
    for novel_file_name in  novels:
        if novel_file_name[-4:] == '.npy':
            continue
        print("---------query No:",nn+1,'----------')
        save_novel_f(model,novel_img_path,novel_file_name,prompt,save_path)
        nn+=1

if __name__ =='__main__':
    
    with torch.no_grad():
        model=SD_DINO()
        img_path='datasets/celeba'
        save_path='Features/face'
        prompt='a photo of face'
        save_all(img_path,save_path,model,prompt,only_base=False)