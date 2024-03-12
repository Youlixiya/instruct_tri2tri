import os
import json
import random
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from typing import Any, Optional, Tuple, Union
from functools import partial
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from torch import distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from instruct_tri2tri.tsr.system import InstructTri2Tri, TSR

class Image2Triplane(Dataset):
    def __init__(self,
                 data_path):
        super().__init__()
        datas = json.load(open(data_path))
        self.datas = datas

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        image_name = data['image']
        image_name = f'data/{image_name}'
        instruct_image_name = f'data/{instruct_image_name}'
        image = Image.open(image_name)
        instruct_image = Image.open(instruct_image_name)
        return image, instruct_image, image_name, instruct_image_name
        
def collate_fn(batch):

    images = []
    instruct_images = []
    image_names = []
    instruct_image_names = []

    for image, instruct_image, image_name, instruct_image_name in batch:
        images.append(image)
        instruct_images.append(instruct_image)
        image_names.append(image_name)
        instruct_image_names.append(instruct_image_name)
    
    return images, instruct_images, image_names, instruct_image_names

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/objaverse/cap3d_automated_objaverse_highquality_instruct_550k.json", help='root path of dataset')
    parser.add_argument('--model_path', type=str, default="instruct_tri2tri/tsr/instruct_tri2tri_config", help='root path of model')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--ckpt', type=str, default='', help='model pretrained ckpt')

    # multi gpu settings
    parser.add_argument("--local-rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=1, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default=".", help='root path')
    parser.add_argument('--work_dir', type=str, default="checkpoints", help='work directory')
    parser.add_argument('--save_dir', type=str, default="instruct_tri2tri", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=2000, help='save iterations')

    args = parser.parse_args()
    return args

def get_optimizer(args, model):
    if args.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)
    
if __name__ == "__main__":
    args = parse_option()
    torch.cuda.set_device(args.local_rank)
    
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device

    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
    
    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
        
    
    model = TSR.from_pretrained(
        'stabilityai/TripoSR',
        'config.yaml',
        'model.ckpt'
    )
    model.requires_grad_(False)
    
    train_dataset = Image2Triplane(args.dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn)
    
    model.to(device=device)
    model, train_loader = accelerator.prepare(model, train_loader)
    loss_fn = torch.nn.MSELoss()
    total_iters = 0
    triplane_save_path = 'data/objaverse/triplanes'
    instruct_triplane_save_path = 'data/objaverse/instruct_triplanes'
    os.makedirs(triplane_save_path, exist_ok=True)
    os.makedirs(instruct_triplane_save_path, exist_ok=True)
    for images, instruct_images, image_names, instruct_names in tqdm(train_loader):
        with torch.no_grad():
            triplane = model(images, device)
            insturct_triplnae = model(instruct_images, device)
            triplane_names = image_names.replace('images', 'triplanes').replace('jpg', 'pt')
            instruct_triplane_names = image_names.replace('images', 'instruct_triplanes').replace('jpg', 'pt')
            torch.save(triplane.cpu(), triplane_names)
            torch.save(insturct_triplnae.cpu(), instruct_triplane_names)