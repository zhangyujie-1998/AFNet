import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image


class MyDataset(data.Dataset):
    def __init__(self, data_dir_texture, data_dir_depth, data_dir_mask, datainfo_path, transform, patch_num, crop_size = 224, img_length_read = 6, is_train = True):
        super(MyDataset, self).__init__()
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
        self.ply_name = dataInfo[['name']]
        self.ply_mos = dataInfo['mos']
        self.crop_size = crop_size
        self.patch_num = patch_num
        self.data_dir_texture = data_dir_texture
        self.data_dir_depth = data_dir_depth
        self.data_dir_mask = data_dir_mask
        self.transform = transform
        self.img_length_read = img_length_read
        self.transform_mask = transforms.RandomCrop(crop_size)
        self.is_train = is_train
        
        img_name = self.ply_name.iloc[:,0]
        img_mos = self.ply_mos
        sample = []
        mos = []

        for i, item in enumerate(img_name):
            for aug in range(patch_num):
                sample.append(item)
        for i, item in enumerate(img_mos):
            for aug in range(patch_num):
                mos.append(item)
        self.sample = sample
        self.mos = mos
        self.length = len(self.sample)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_name = self.sample[idx]
        img_name_parts = img_name.split('.')
        img_prefix = img_name_parts[0]
                 
        texture_dir = self.data_dir_texture
        depth_dir = self.data_dir_depth
        mask_dir = self.data_dir_mask

        img_channel = 5
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size
        img_length_read = self.img_length_read       
        transformed_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        
        img_read_index = 0
        
        for i in range(img_length_read):
  
            imge_name = os.path.join(texture_dir, img_prefix + '_view_' + str(i) + '.png')
            depth_name = os.path.join(depth_dir, img_prefix + '_view_' + str(i) + '.png')
            mask_name = os.path.join(mask_dir, img_prefix + '_view_' + str(i) + '.png')
            if os.path.exists(imge_name):
                read_texture = Image.open(imge_name)
                read_texture = read_texture.convert('RGB')
                read_depth = Image.open(depth_name)
                read_mask = Image.open(mask_name)
                
                read_texture = transforms.ToTensor()(read_texture)
                read_depth = transforms.ToTensor()(read_depth)
                read_mask = transforms.ToTensor()(read_mask)
   
                read_texture_depth_mask = torch.cat([read_texture,read_depth, read_mask], dim=0)
                read_texture_depth_mask = self.transform(read_texture_depth_mask)

                transformed_img[i] = read_texture_depth_mask

                img_read_index += 1
            else:
                print(imge_name)
                print('Image do not exist!')

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                transformed_img[j] = transformed_img[img_read_index-1]

        mos = self.mos[idx] 
        mos = torch.FloatTensor(np.array(mos))
 
        return transformed_img, mos
    
    
    
