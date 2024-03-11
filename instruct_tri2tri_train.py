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
from transformers import get_cosine_schedule_with_warmup, AutoProcessor, CLIPModel, CLIPProcessor
from tensorboardX import SummaryWriter

from torch import distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from instruct_tri2tri.tsr.system import InstructTri2Tri, TSR

class InstructTri2TriDataset(Dataset):
    def __init__(self,
                 data_path):
        super().__init__()
        datas = json.load(open(data_path))[:200]
        self.datas = datas

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        image_name = data['image']
        instruct_image_name = data['instruct_image']
        image = Image.open(f'data/{image_name}')
        instruct_image = Image.open(f'data/{instruct_image_name}')
        # image_name = image_name.split('/')[-1]
        instruct = data['instruct']
        return image, instruct_image, instruct
        
def collate_fn(batch):

    images = []
    instruct_images = []
    instructs = []

    for image, instruct_image, instruct in batch:
        images.append(image)
        instruct_images.append(instruct_image)
        instructs.append(instruct)
    
    return images, instruct_images, instructs

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/objaverse/cap3d_automated_objaverse_highquality_instruct_550k.json", help='root path of dataset')
    parser.add_argument('--model_path', type=str, default="instruct_tri2tri/tsr/instruct_tri2tri_config", help='root path of model')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
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
    
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2)
    accelerator = Accelerator(mixed_precision='fp16',  deepspeed_plugin=deepspeed_plugin)
    device = accelerator.device
    # device = torch.device('cuda', args.local_rank)
    # dtype = torch.float32
    # torch.distributed.init_process_group(backend='nccl')
    
    # file folder creating
    # if args.local_rank == 0:
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
    tsr = TSR.from_pretrained(
        'stabilityai/TripoSR',
        'config.yaml',
        'model.ckpt'
    )
    model = InstructTri2Tri.from_pretrained(
        args.model_path,
        'config.yaml',
        'model.ckpt'
    )
    model.load_state_dict(tsr.state_dict(), False)
    del tsr
    # model.instruction_converter.load_state_dict(model.backbone.state_dict())
    model.requires_grad_(False)
    model.instruction_converter.requires_grad_(True)
    # model.image_tokenizer.requires_grad_(False)
    # model.tokenizer.requires_grad_(False)
    # model.text_encoder.requires_grad_(False)
    # model.backbone.requires_grad_(False)
    # model.post_processor.requires_grad_(False)
    # model.decoder.requires_grad_(False)
    # model.renderer.requires_grad_(False)
    
    train_dataset = InstructTri2TriDataset(args.dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)
    
    model.to(device=device)
    # model.half()
    optimizer = get_optimizer(args, model.instruction_converter)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_loader) * args.epochs) // args.gradient_accumulation_steps,
    )
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader)
    loss_fn = torch.nn.MSELoss()
    total_iters = 0
    os.makedirs(os.path.join(args.root_path, args.work_dir), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        # new epoch
        # if args.local_rank == 0:
        accelerator.print("------start epoch {}------".format(epoch))
        # train_sampler.set_epoch(epoch)
        # training
        model.train()
        
        for batch_idx, (images, instruct_images, instructs) in enumerate(train_loader):
            total_iters += 1
            samples = len(images)

            with torch.no_grad():
                target_tokens = model.module.forward_tsr(instruct_images, device, False)
            pred_tokens = model(images, instructs, device, False)
            dim = target_tokens.shape[2]
            index = torch.randint(0, pred_tokens.shape[1], (32, ))
            loss = loss_fn(pred_tokens.reshape[:, index, :](-1, dim), target_tokens[:, index, :].reshape(-1, dim))
            accelerator.backward(loss)
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                    epoch, batch_idx // args.gradient_accumulation_steps, len(train_loader) // args.gradient_accumulation_steps,
                        100. * batch_idx / len(train_loader), loss.item()))
            
            # save model
            if total_iters % (args.save_iters * args.gradient_accumulation_steps) == 0 and total_iters != 0:
                save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + '.ckpt')
                accelerator.print("save model to {}".format(save_path))
                unwrap_model = accelerator.unwrap_model(model)
                torch.save(unwrap_model.state_dict(), save_path)

    unwrap_model = accelerator.unwrap_model(model)
    torch.save(unwrap_model.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.ckpt"))
    
    