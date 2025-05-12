import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils.misc import process_cfg
from utils import flow_viz

from core.Networks import build_network

from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
import itertools
import imageio

def prepare_image(seq_dir):
    print(f"preparing image...")
    print(f"Input image sequence dir = {seq_dir}")

    images = []

    image_list = sorted(os.listdir(seq_dir))

    for fn in image_list:
        img = Image.open(os.path.join(seq_dir, fn))
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
    
    return torch.stack(images)

def vis_pre(flow_pre, vis_dir, name, beforflowhead):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    print(os.path.join(vis_dir+'/beforflowhead',name+'.pt'))
    torch.save(beforflowhead.detach(),os.path.join(vis_dir+'/beforflowhead',name+'.pt'))
    print('beforflowheadsaveshaope:',beforflowhead.shape)

@torch.no_grad()
def MOF_inference(model, dir):

    model.eval()

    input_images = prepare_image(dir)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _, beforflowhead = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()
    print('flow_pre', flow_pre.shape ) #([12, 2, 224, 224])
    return flow_pre, beforflowhead

@torch.no_grad()
def BOF_inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg.seq_dir)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # 基本目录路径
    base_dir = "/home/ubuntu/allrgbs"
    dst_dir = "/media/ubuntu/eventencoder"
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')
    
    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    print(cfg)
    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    

    with torch.no_grad():
        if args.mode == 'MOF':
            from configs.multiframes_sintel_submission import get_cfg
            for folder in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder)
                if os.path.isdir(folder_path):
                    flow_pre, beforflowhead = MOF_inference(model.module, folder_path)
                    vis_pre(flow_pre, dst_dir, folder, beforflowhead)
    
    



