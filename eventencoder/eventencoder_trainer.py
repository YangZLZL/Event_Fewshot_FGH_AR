import sys
sys.path.append('core')
from matplotlib import pyplot as plt
import numpy as np
import torch.distributed as dist
import os
#os.environ["CUDA_VISIBLE_DEVICES"]='0'
import cv2
import shutil
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
#from videoflow import inference
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader
import random
import logging
import time
import math
from draw_loss import plot_loss_graphs_combined
from torchvision import transforms
import torch.nn.functional as F
import re


import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from videoflow.core.utils.misc import process_cfg
#from utils import flow_viz

from videoflow.core.Networks import build_network

from videoflow.core.utils import frame_utils
from videoflow.core.utils.utils import InputPadder, forward_interpolate
from unet.unet_parts import *
from unet.unet_model import *
import itertools
import imageio




def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoDataset(Dataset):
    def __init__(self, root_dir, feature_dir,rgbdir,flow_dir, sample_num, num_frames=64, train=True):
        self.root_dir = root_dir
        self.feature_dir = feature_dir
        self.num_frames = num_frames
        self.rgbdir = rgbdir
        self.flow_dir = flow_dir
        self.video_names = os.listdir(root_dir)

        random.seed(12345)
        self.video_names = random.sample(self.video_names, sample_num)

        total_num = len(self.video_names)
        print('total_num:', total_num)
        #train_num = int(0.8 * total_num)

        #if train:
        #   self.video_names = self.video_names[:train_num]  # Use the first 80% of the videos for training
        #else:
         #   self.video_names = self.video_names[train_num:]  # Use the remaining 20% for validation

    def __len__(self):
        return len(self.video_names)
    
    def load_and_stack_images(self, directory):
        files = os.listdir(directory)
        
        regex = re.compile(r'(\d+)-(\d+)\.jpg')
        image_files = [f for f in files if regex.match(f)]
        
        
        image_files.sort(key=lambda x: [int(i) for i in regex.match(x).groups()])
        #print("Sorted files:", image_files)
        images = [Image.open(os.path.join(directory, f)).convert('RGB') for f in image_files]
        tensors = [torch.tensor(np.array(img)) for img in images]
        tensors = [t.permute((2,0,1)) for t in tensors]
    
        stacked_tensor = torch.stack(tensors,dim=0)  
        #stacked_tensor = stacked_tensor.permute(3, 0, 1, 2)  # Rearrange dimensions: (64, 224, 224, 3) -> (channels, frames, height, width)
        stacked_tensor = stacked_tensor/255
        stacked_tensor = torch.round(stacked_tensor)
        int_stacked_tensor = 1 - stacked_tensor
        return int_stacked_tensor

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        #print('loading {}'.format(video_name))
        video_path = os.path.join(self.root_dir ,video_name)
        feature_path = os.path.join(self.feature_dir, f"{video_name}.pt")
        rgbs_path = os.path.join(self.rgbdir, video_name)
        flow_path = os.path.join(self.flow_dir, f"{video_name}.pt")
        stacked_tensor = self.load_and_stack_images(video_path).float()
        stacked_tensor = stacked_tensor.permute(1, 0, 2, 3)
        #print(stacked_tensor)

        rgb_feature = torch.load(feature_path).float()
        flow_feature = torch.load(flow_path).float()

        return stacked_tensor.data, rgb_feature.data, rgbs_path, flow_feature.data



class VideoFeatureExtractor(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        base_model = models.video.r3d_18(pretrained=True)

        for param in base_model.parameters():
            param.requires_grad = False

        self.base_features = nn.Sequential(*list(base_model.children())[:-2])
        
        for param in self.base_features[-1].parameters():
            param.requires_grad = True



    def forward(self, x):
        x = self.base_features(x)  
        feature5121414 = x

        x = self.avgpool(x)
        x = x.squeeze()  

        return x, feature5121414.squeeze()

def batch_sinkhorn(batch_out):
    b = batch_out.shape[0]
    batch_Q = torch.empty_like(batch_out)
    
    for i in range(b):
        #print(batch_out[i])
        batch_Q[i] = sinkhorn_stable(batch_out[i])
        
    return batch_Q

def rowwise_kl_divergence_nn(matrix1, matrix2):
    """
    Compute the row-wise Kullback-Leibler divergence between two matrices using PyTorch's nn.KLDivLoss.
    This function allows backpropagation for optimization.
    Parameters:
        matrix1, matrix2: PyTorch tensors of shape (n, m) representing the matrices. 
                           These tensors should require gradients if you want to backpropagate the loss.
    Returns:
        A scalar tensor representing the sum of row-wise KL divergence between matrix1 and matrix2.
    """
    total_kl_div = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    for row1, row2 in zip(matrix1, matrix2):
        kl_row = F.kl_div(row1, row2, reduction='batchmean')
        total_kl_div = total_kl_div + kl_row
    return total_kl_div


def sinkhorn_stable(out):
    Q = torch.exp(out / torch.max(out)).t()  # Subtracting the max value for numerical stability
    #Q = torch.exp(out).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q = Q / sum_Q.detach()

    for it in range(10):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q = Q / sum_of_rows.detach()
        Q = Q / K

        # normalize each column: total weight per sample must be 1/B
        Q = Q / torch.sum(Q, dim=0, keepdim=True)
        Q = Q / B

    Q = Q * B  # the colomns must sum to 1 so that Q is an assignment

    return Q.t()

import torch.nn.init as init
class FlowAlignModule(nn.Module):
    def __init__(self):
        super(FlowAlignModule, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=512, 
                                         out_channels=128, 
                                         kernel_size=3, 
                                         stride=2, 
                                         padding=1, 
                                         output_padding=1).float()
        #init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        #if self.deconv.bias is not None:
            #init.zeros_(self.deconv.bias)
        
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.GELU()
        self.unet = UNet().float()
        self.unet = self.replace_relu_with_gelu(self.unet)
        self.layer_norm = nn.LayerNorm(784).float()

    def replace_relu_with_gelu(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.GELU())
            else:
                self.replace_relu_with_gelu(child)

        return model

    def slide_avg(self, input_tensor):
       # print('input_tensor.shape: ', input_tensor.shape)
        if input_tensor.shape[1] == 8:
            avg_frames = torch.stack([
                torch.mean(input_tensor[:, 0:4, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 2:6, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 4:8, :, :, :], dim=1, keepdim=True)
            ], dim=1).squeeze(2)
        elif input_tensor.shape[1] == 6:
            avg_frames = torch.stack([
                torch.mean(input_tensor[:, 0:2, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 2:4, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 4:6, :, :, :], dim=1, keepdim=True)
            ], dim=1).squeeze(2)
        else:
            raise ValueError("Input tensor must have 6 or 8 frames in the second dimension")

        return avg_frames

    def forward(self, x):
        b, _, _, h, w = x.size()
        x = x.view(-1, 512, h, w)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        _, _, h_out, w_out = x.size()
        #print('xzize:',x.size())
        x = x.view(-1,8, 128, h_out, w_out)
        x = self.slide_avg(x)
        x = x.view(-1,128, h_out, w_out)
        x = self.unet(x)
        x = x.view(b,3,128, h_out, w_out)
        x_reshaped = x.view(b, 3, 128, -1)
        normalized_x = self.layer_norm(x_reshaped)
        normalized_x = normalized_x.view(b, 3, 128, 28, 28)

        #print(normalized_x)
        return normalized_x
    


class CompleteModel(nn.Module):
    def __init__(self):
        super(CompleteModel, self).__init__()
        self.eventencoder = VideoFeatureExtractor().float()
        self.flowalignmodule = FlowAlignModule().float()
        self.layer_norm = nn.LayerNorm(512).float()
        self.initialize_flowalignmodule_kaiming(self.flowalignmodule)
        for param in self.flowalignmodule.parameters():
            param.requires_grad = True

    def initialize_flowalignmodule_kaiming(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            # Add other layer types here if needed
    def slide_avg(self, input_tensor):
        if input_tensor.shape[1] == 8:
            avg_frames = torch.stack([
                torch.mean(input_tensor[:, 0:4, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 2:6, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 4:8, :, :, :], dim=1, keepdim=True)
            ], dim=1).squeeze(2)
        elif input_tensor.shape[1] == 6:
            avg_frames = torch.stack([
                torch.mean(input_tensor[:, 0:2, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 2:4, :, :, :], dim=1, keepdim=True),
                torch.mean(input_tensor[:, 4:6, :, :, :], dim=1, keepdim=True)
            ], dim=1).squeeze(2)
        else:
            raise ValueError("Input tensor must have 6 or 8 frames in the second dimension")

        return avg_frames
    
    def forward(self, framess, flow_feature):
        framess = framess.cuda()
  
        split_tensors = torch.split(framess, 8, dim=2)
        cnt = 0
        for tensor in split_tensors:
            output,feature5121414 = self.eventencoder(tensor) #4,512
            output = output.unsqueeze(1) # 4 1 512
            feature5121414 = feature5121414.unsqueeze(1)
            if cnt == 0:
                output_ind = output
            else:
                output_ind = torch.cat((output_ind, output), 1)
            if cnt == 0:
                feature5121414_ind = feature5121414
            else:
                feature5121414_ind = torch.cat((feature5121414_ind, feature5121414), 1)
            cnt += 1
        output_ind = self.layer_norm(output_ind)
        afterflowalignmodule = self.flowalignmodule(feature5121414_ind)
        avg_flow_feature = self.slide_avg(flow_feature)

        return output_ind, afterflowalignmodule, avg_flow_feature
    


def train(completemodel, dataloader, val_dataloader, device, output_dir, attention, num_epochs, pretrained_weights=None,mlp = None, alpha=None,beta = None,gamma = None, rgbs_dir = None):
    completemodel.train()
    #optimizer = optim.SGD(completemodel.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    #optimizer = optim.SGD(completemodel.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0001)

    optimizer = optim.AdamW(completemodel.parameters(), lr=0.001, weight_decay=0.01)
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    softmax = nn.Softmax(dim=-1)
    log_softmax = nn.LogSoftmax(dim=-1)

    start_epoch = 0

    if pretrained_weights is not None:
        checkpoint = pretrained_weights
        completemodel.load_state_dict(checkpoint['model_state_dict'])
        print('checkpoint: ', checkpoint['epoch'])
        start_epoch = checkpoint['epoch']
    
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.INFO)
    train_log_file = os.path.join(output_dir, 'train_log.txt')
    train_file_handler = logging.FileHandler(train_log_file)
    train_logger.addHandler(train_file_handler)

    val_logger = logging.getLogger('validation')
    val_logger.setLevel(logging.INFO)
    val_log_file = os.path.join(output_dir, 'val_log.txt')
    val_file_handler = logging.FileHandler(val_log_file)
    val_logger.addHandler(val_file_handler)

   
    for epoch in range(0, num_epochs):
        running_loss = 0.0
        sum_loss0 = 0.0
        sum_loss1 = 0.0
        avg_loss0 = 0.0
        avg_loss1 = 0.0
        for i, data in enumerate(dataloader, 0):
            print("{} / {}".format(i, len(dataloader)))
            framess, rgb_feature, rgbs_path, flow_feature = data
            output_ind, afterflowalignmodule, avg_flow_feature = completemodel(framess,flow_feature)

            transpose_tensor_rgb_feature = rgb_feature.transpose(1, 2)
            rgb_feature_selfcorelation = torch.bmm(rgb_feature, transpose_tensor_rgb_feature)
            sinkhorn_rgb_feature_selfcorelation = batch_sinkhorn(rgb_feature_selfcorelation)

            output_ind_transpose_tensor = output_ind.transpose(1, 2)
            output_ind_selfcorelation = torch.bmm(output_ind, output_ind_transpose_tensor)
            logsm_output_ind_selfcorelation = log_softmax(output_ind_selfcorelation)

            pool_avg_flow_feature = log_softmax(F.adaptive_avg_pool3d(avg_flow_feature, (128,1, 1)).squeeze(3).squeeze(3))
            pool_afterflowalignmodule = softmax(F.adaptive_avg_pool3d(afterflowalignmodule, (128,1, 1)).squeeze(3).squeeze(3))

            loss0 = rowwise_kl_divergence_nn(logsm_output_ind_selfcorelation,sinkhorn_rgb_feature_selfcorelation)*alpha        
            loss1 = rowwise_kl_divergence_nn(pool_avg_flow_feature, pool_afterflowalignmodule) * beta

            sum_loss0 += loss0
            sum_loss1 += loss1
            print('loss0: ', loss0)
            print('loss1: ', loss1)
            with open("./mid_level_action/model_parameters.txt", "w") as file:
                for name, param in completemodel.flowalignmodule.named_parameters():
                    file.write(f"Layer: {name}\n")
                    file.write(f"Requires Gradient: {param.requires_grad}\n")
                    file.write(f"Gradient: {param.grad}\n")
                    file.write(f"Parameters: \n{100*param.data}\n\n")
            with open('./mid_level_action/gradients.txt', 'w') as file:
                for name, module in completemodel.flowalignmodule.named_children():
                    for param_name, param in module.named_parameters():
                        file.write(f"Gradient of {name}.{param_name}: {param.grad}\n")
            #print(completemodel.eventencoder.base_features[-1].weight.grad)
            loss = loss0 + loss1
            loss.backward(retain_graph=False)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        avg_loss0 = sum_loss0 / len(dataloader)
        avg_loss1 = sum_loss1 / len(dataloader)
 
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
        current_lr = optimizer.param_groups[0]['lr']
        train_logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Learning Rate: {current_lr}, loss0: {avg_loss0}, loss1: {avg_loss1}")
        
        val = False
        if val:
            # Validate the model
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                sum_val_loss = 0.0
                sum_val_loss0 = 0.0
                sum_val_loss1 = 0.0
                val_avg_loss0 = 0.0
                val_avg_loss1 = 0.0

                for i, data in enumerate(val_dataloader, 0):
                    optimizer.zero_grad()
                    print("{} / {}".format(i, len(val_dataloader)))
                    framess, rgb_feature, video_name, rgbs_path, flow_feature = data
                    framess = framess.to(device)
                    rgb_feature = rgb_feature.to(device)
                    ###############################################################################
                    transpose_tensor = rgb_feature.transpose(1, 2)
                    rgb_feature_selfcorelation = torch.bmm(rgb_feature, transpose_tensor)
                    sinkhorn_rgb_feature_selfcorelation = batch_sinkhorn(rgb_feature_selfcorelation)
                    ###############################################################################
                    split_tensors = torch.split(framess, 8, dim=2)
                    cnt = 0
                    for tensor in split_tensors:
                        output,feature5121414 = model(tensor) #4,512
                        output = output.unsqueeze(1) # 4 1 512
                        feature5121414 = feature5121414.unsqueeze(1)
                        if cnt == 0:
                            output_ind = output
                        else:
                            output_ind = torch.cat((output_ind, output), 1)
                        if cnt == 0:
                            feature5121414_ind = feature5121414
                        else:
                            feature5121414_ind = torch.cat((feature5121414_ind, feature5121414), 1)
                        cnt += 1
                    output_ind = layer_norm(output_ind)
                    feature5121414_ind = transposeconv(feature5121414_ind)
                    feature5121414_ind_list = [feature5121414_ind[i:i+1] for i in range(feature5121414_ind.shape[0])]
                    output_ind_transpose_tensor = output_ind.transpose(1, 2)
                    output_ind_selfcorelation = torch.bmm(output_ind, output_ind_transpose_tensor)
                    logsm_output_ind_selfcorelation = log_softmax(output_ind_selfcorelation)

                    ###############################################################################
                    afterflowalignmodule_list = []
                    for i in range(len(feature5121414_ind_list)):
                        afterflowalignmodule = MOF_inference(flowmodel, feature5121414_ind[i], rgbs_path[i])
                        afterflowalignmodule_list.append(afterflowalignmodule.unsqueeze(0))
                    afterflowalignmodule = torch.cat(afterflowalignmodule_list, dim=0)

                    pred_prob = F.softmax(afterflowalignmodule, dim=1)
                    target_log_prob = F.log_softmax(flow_feature, dim=1)

                    b = flow_feature.shape[0]
                    pred_prob = pred_prob.view(b, 6, -1) 
                    target_log_prob = target_log_prob.view(b, 6, -1)  

                    val_loss1 = F.kl_div(target_log_prob, pred_prob, reduction='batchmean') * beta     
                    val_loss0 = rowwise_kl_divergence_nn(logsm_output_ind_selfcorelation,sinkhorn_rgb_feature_selfcorelation)*alpha  
                
                    print('val_loss0: ', val_loss0)
                    print('val_loss1: ', val_loss1)
                    val_loss = val_loss0.item() + val_loss1.item()
                    sum_val_loss += val_loss
                    sum_val_loss0 += val_loss0.item()
                    sum_val_loss1 += val_loss1.item()

                epoch_val_loss = sum_val_loss / (i + 1)
                val_avg_loss0 = sum_val_loss0 / (i + 1)
                val_avg_loss1 = sum_val_loss1 / (i + 1)

                print(f"Validation Loss after Epoch {epoch + 1}: {epoch_val_loss}")
                val_logger.info(f"Validation Loss after Epoch {epoch + 1}: {val_loss}, Learning Rate: {current_lr}, val_loss0: {val_avg_loss0}, val_loss1: {val_avg_loss1}")

        # Save the model and the validation loss
        model_path = os.path.join(output_dir, f"event3dencoder_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': completemodel.state_dict(),
            #'transposeconv_state_dict': transposeconv.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
        }, model_path)
            
        plot_loss_graphs_combined(output_dir)


    print("Finished Training")
    logging.info("Finished Training")

    logging.info(f"Saved model: {model_path}")


def build_network(cfg):
    name = cfg.network 
    if name == 'MOFNetStack':
        from videoflow.core.Networks.MOFNetStack.network import MOFNet as network

    return network(cfg[name])



import torch.multiprocessing as mp
import clip
import argparse
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #transposeconv = TransposeConv2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0).to(device)

    num_epochs = 201
    bs = 32
    sample_num = 99999
    num_frames = 64
    alpha = 1
    beta = 0.1
    gamma = 1
    output_dir = "./event3dencoder/r3d_18_{}vid_{}frames_bs{}_sotatry_{}_{}".format(sample_num, num_frames, bs, alpha, beta)
    os.makedirs(output_dir)
    
    # Save a copy of this script in output_dir
    shutil.copy(__file__, output_dir)

    # Setup logging
    #setup_logging(output_dir)
    root_dir = "PATH_TO_YOUR_ROOT_DIR"
    feature_dir = "PATH_TO_YOUR_RGB_FEATURE_DIR"
    rgbs_dir = "PATH_TO_YOUR_RGB_DIR"
    flow_dir = "PATH_TO_YOUR_FLOW_FEATURE_DIR"
    
    train_dataset = VideoDataset(root_dir, feature_dir, rgbs_dir,flow_dir, sample_num=sample_num, num_frames=num_frames, train=True)
    val_dataset = VideoDataset(root_dir, feature_dir, rgbs_dir, flow_dir,sample_num=sample_num, num_frames=num_frames, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4) 

    completemodel = CompleteModel().float().cuda()
    # Use DataParallel for multi-GPU training
    use = 1
    if torch.cuda.device_count() > 1 and use == 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model, device_ids=[0, 1])  # Specify the GPU IDs here
    pretrained_weights = torch.load('./event3dencoder_50.pt')
   # pretrained_weights = None
    
    train(completemodel, train_dataloader, val_dataloader, device, output_dir=output_dir,attention =None, num_epochs=num_epochs, pretrained_weights=pretrained_weights,mlp = None,alpha=alpha, beta=beta, gamma=gamma, rgbs_dir=rgbs_dir)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
 