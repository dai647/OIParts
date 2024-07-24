import torch
import numpy as np
from PIL import Image
from sd_dino.extractor_sd import load_model, process_features_and_mask
from sd_dino.utils.utils_correspondence import resize
from sd_dino.extractor_dino import ViTExtractor


class SD_DINO():
    def __init__(self):
        self.MASK = True
        self.VER = "v1-5"
        self.PCA = False
        self.CO_PCA = True
        self.PCA_DIMS = [256, 256, 256]
        self.SIZE =960
        self.EDGE_PAD = False

        self.FUSE_DINO = 1
        self.ONLY_DINO = 0
        self.DINOV2 = True
        self.MODEL_SIZE = 'base'
        self.DRAW_DENSE = 1
        self.DRAW_SWAP = 1
        self.TEXT_INPUT = False
        # self.SEED = 42
        self.TIMESTEP = 100
        self.real_size=960
        self.img_size = 840 
        self.DIST = 'l2' if self.FUSE_DINO and not self.ONLY_DINO else 'cos'
        if self.ONLY_DINO:
            self.FUSE_DINO = True
        # np.random.seed(self.SEED)
        # torch.manual_seed(self.SEED)
        # torch.cuda.manual_seed(self.SEED)
        # torch.backends.cudnn.benchmark = True

        
        model_dict={'small':'dinov2_vits14',
                    'base':'dinov2_vitb14',
                    'large':'dinov2_vitl14',
                    'giant':'dinov2_vitg14'}

        model_type = model_dict[self.MODEL_SIZE] if self.DINOV2 else 'dino_vits8'
        self.layer = 11 if self.DINOV2 else 9
        if 'l' in model_type:
            self.layer = 23
        elif 'g' in model_type:
            self.layer = 39
        self.facet = 'token' if self.DINOV2 else 'key'
        stride = 14 if self.DINOV2 else 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = ViTExtractor(model_type, stride, device=self.device)
        patch_size = self.extractor.model.patch_embed.patch_size[0] if self.DINOV2 else self.extractor.model.patch_embed.patch_size
        self.num_patches = int(patch_size / stride * (self.img_size // patch_size - 1) + 1)
        
        self.model, self.aug = load_model(diffusion_ver=self.VER, image_size=self.SIZE, num_timesteps=self.TIMESTEP)
        
    def get_sd_f(self,img_path,input_text = "a photo of a "):
        img1 = Image.open(img_path).convert('RGB')
        img1_input = resize(img1, self.real_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        img1_desc_SD = process_features_and_mask(self.model, self.aug, img1_input, input_text=input_text, mask=False, pca=self.PCA).reshape(1,1,-1, self.num_patches**2).permute(0,1,3,2)

        return img1_desc_SD
    
    def get_dino_f(self,img_path):
        img1 = Image.open(img_path).convert('RGB')
        img1 = resize(img1, self.img_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        img1_batch = self.extractor.preprocess_pil(img1)
        img1_desc_dino = self.extractor.extract_descriptors(img1_batch.to(self.device), self.layer, self.facet)

        return img1_desc_dino

    def get_fuse_f(self,img1_desc_SD,img1_desc_dino):

        img1_desc_SD2 = img1_desc_SD / img1_desc_SD.norm(dim=-1, keepdim=True)
        img1_desc_dino2 = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
        img1_desc_fuse = torch.cat((img1_desc_SD2, img1_desc_dino2), dim=-1)
        
        return img1_desc_fuse
        


