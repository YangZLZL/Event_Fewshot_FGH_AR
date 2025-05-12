import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader
import random
import logging
import time
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]='0'



class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=64):
        self.root_dir = root_dir

        self.num_frames = num_frames
        self.video_names = [os.path.splitext(filename)[0] for filename in os.listdir(root_dir)]



    def __len__(self):
        return len(self.video_names)
    
    def sort_key(self,file_name):
        return int(os.path.splitext(file_name)[0])
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.root_dir, video_name)
        files = os.listdir(video_path)
        sorted_files = sorted(files, key=self.sort_key)

        frames = []
        framedir = []
        for frame_name in sorted_files:
            frame_path = os.path.join(video_path, frame_name)
            frame = Image.open(frame_path).convert("RGB")
            ##print(frame_path)
            frames.append(ToTensor()(frame))
            framedir.append(str(frame_path))
        frame_count = len(frames)

        selected_indices = torch.linspace(0, len(frames) - 1, steps=self.num_frames).long()
        frames_tensor = torch.stack([frames[i] for i in selected_indices]).float()
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # Rearrange dimensions: (frames, channels, height, width) -> (channels, frames, height, width)
        split_frames = []
        for i in range(0, frames_tensor.shape[1], 8):
            split_frames.append(frames_tensor[:, i:i+8, :, :].data)

        
        return split_frames, video_name, video_name

class ImprovedMLP(nn.Module):
    def __init__(self):
        super(ImprovedMLP, self).__init__()
        self.fc = nn.Linear(5, 40)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x


class VideoFeatureExtractor(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        base_model = models.video.r3d_18(pretrained=False)

        for param in base_model.parameters():
            param.requires_grad = False

        self.base_features = nn.Sequential(*list(base_model.children())[:-2])

        print(self.base_features)
    def forward(self, x):
        x = self.base_features(x)  

        return x.squeeze(2)
@torch.no_grad()
def encode(model, dataloader,  device, output_dir,pretrained_weights):

    state_dict = pretrained_weights['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items() if "eventencoder" in k}
    #state_dict = pretrained_weights
    new_state_dict = {k.replace('eventencoder.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    print('load success')

    model.eval()

    for i, data in enumerate(tqdm(dataloader), 0):

        print("{} / {}".format(i, len(dataloader)))
        framess, global_feature, name = data
        if(os.path.exists(output_dir+ '/'+name[0]+'.pt')):
            print('exist')
            continue
        for i in range(len(framess)):
            framess[i] = framess[i].to(device)

        encoded_splits = [model(frames.to(device)).float() for frames in framess]
        local_features = torch.cat(encoded_splits, dim=0)
        print('local_features.shape: ', local_features.shape)
        torch.save(local_features, output_dir+ '/'+name[0]+'.pt')


    

import torch.multiprocessing as mp
import clip
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dir = 'YOUR_pt_FILE'
    output_dir = "./encoded/"+dir.split('/')[-2]+dir.split('/')[-1].split('.')[0]+'_'
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    root_dir = "YOUR_PROJECT_ROOT_DIR"


    train_dataset = VideoDataset(root_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    model = VideoFeatureExtractor().float().to(device)
    use = 0
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1 and use == 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model, device_ids=[0, 1])  # Specify the GPU IDs here

    a = torch.load(dir)
    encode(model, train_dataloader, device, output_dir,pretrained_weights=a)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
 