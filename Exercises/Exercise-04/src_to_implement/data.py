import torch            #type: ignore
import numpy as np
import os
import socket

from pathlib            import Path                 #type: ignore
from torchvision        import transforms as tv     #type: ignore
from torch.utils.data   import Dataset              #type: ignore
from skimage.io         import imread               #type: ignore
from skimage.color      import gray2rgb             #type: ignore

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std  = [0.16043035, 0.16043035, 0.16043035]

environment_type = os.getenv('ENVIRONMENT_TYPE')

ROOT = '/home/jovyan/work/exercise4_material/src_to_implement/' if environment_type == 'cuda-env' else './'

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        
        if self.mode == 'train':
            self._transform = tv.Compose([
                tv.ToPILImage(),
                tv.ToTensor(),
                tv.Normalize(mean = train_mean, std = train_std)  
            ])
        if self.mode == 'val':
            self._transform = tv.Compose([
                tv.ToPILImage(),
                tv.ToTensor(),
                tv.Normalize(mean = train_mean, std = train_std)
            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        curr_data = self.data.iloc[index] 
        
        image_path = os.path.join(ROOT, curr_data['filename'])
        image = imread(image_path, as_gray = True)
        image = gray2rgb(image) 
        image = self._transform(image)
        
        label = np.asarray([curr_data['crack'], curr_data['inactive']])
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    
    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, transform):
        self._transform = transform