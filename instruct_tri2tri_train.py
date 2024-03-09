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
from triplane_clip import TriPlaneCLIPModel, TriPlaneCLIPConfig

from torch import distributed as dist
from accelerate import Accelerator
from instruct_tri2tri.tsr.system import TSR, InstructTri2Tri

class InstructTri2TriDataset(Dataset):
    def __init__(self,
                 data_path,
                 num_chunks,
                 chunk_index):
        super().__init__()
        datas = json.load(open(data_path))
        self.datas = datas

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        image_name = data['image']
        instruct_image_name = data['insturct_image']
        image = Image.open(f'data/{image_name}')
        instruct_image = Image.open(f'data/{instruct_image_name}')
        image_name = image_name.split('/')[-1]
        instruct = data['instruct']
        return image, instruct_image, instruct
        
    
def collate_fn(batch, processor):

    images = []
    texts = []

    for image, text in batch:
        images.append(image)
        texts.append(text)
    
    inputs = processor(text=texts,
                       return_tensors="pt",
                       padding=True,
                       truncation=True)
    inputs['images'] = images
    
    return inputs

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/objaverse/Cap3D_imgs_view0", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='gradient accumulation steps')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=1, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default=".", help='root path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpts", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=10000, help='save iterations')

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

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
    
if __name__ == "__main__":
    args = parse_option()
    torch.cuda.set_device(args.local_rank)
    accelerator = Accelerator()
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
    triplaneclip_config = TriPlaneCLIPConfig()
    model = TriPlaneCLIPModel(triplaneclip_config)
    clip = CLIPModel.from_pretrained("ckpts/clip-vit-large-patch14")
    
    model.text_model = clip.text_model
    model.visual_projection = clip.visual_projection
    model.text_projection = clip.text_projection
    model.vision_model.pre_layrnorm = clip.vision_model.pre_layrnorm
    model.vision_model.encoder = clip.vision_model.encoder
    model.vision_model.post_layernorm = clip.vision_model.post_layernorm
    
    model.vision_model.pre_layrnorm.requires_grad_(False)
    model.vision_model.encoder.requires_grad_(False)
    model.vision_model.post_layernorm.requires_grad_(False)
    model.text_model.requires_grad_(False)
    model.image2triplane_model.requires_grad_(False)
    model.visual_projection.requires_grad_(False)
    model.text_projection.requires_grad_(False)
    model.image2triplane_model.requires_grad_(False)
    
    del clip
    
    processor = CLIPProcessor.from_pretrained("ckpts/clip-vit-large-patch14")
    # processor.tokenizer.add_tokens(['<ref>', '</ref>', '<box>', '</box>'], special_tokens=True)
    # model.resize_token_embeddings(len(processor.tokenizer))
    train_dataset = TriPlaneCLIPDataset(args.dataset_path)
    # train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=partial(collate_fn, processor=processor))
    # if args.local_rank == 0:
        # writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))
    # model.vision_model.requires_grad_(False)
    
    model.to(device=device)
    clip_vision.to(device=device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
     # optimizer and scheduler
    # optimizer = get_optimizer(args, model.module.vision_model)
    optimizer = get_optimizer(args, model.vision_model.embeddings)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_loader) * args.epochs) // args.gradient_accumulation_steps,
    )
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader)
    loss_fn = torch.nn.MSELoss()
    total_iters = 0
    
    for epoch in range(1, args.epochs + 1):
        # new epoch
        # if args.local_rank == 0:
        accelerator.print("------start epoch {}------".format(epoch))
        # train_sampler.set_epoch(epoch)
        # training
        model.train()
        for batch_idx, inputs in enumerate(train_loader):
            total_iters += 1
            samples = len(inputs['images'])
            for key, value in inputs.items():
                if type(value) == torch.Tensor:
                    inputs[key] = value.to(device=device)
                    # if key == 'pixel_values':
                    #     clip_inputs[key] = value.to(device=device)
                    # else:
                    #     clip_inputs[key] = value.to(device=device)
            data['return_loss'] = True
            optimizer.zero_grad()
            out = model(**inputs)
            loss = out.loss
            # print(loss.item())
            # loss = reduce_mean(output.loss, dist.get_world_size())
            # loss.backward()
            accelerator.backward(loss)
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLM Loss: {:.6f}'.format(
                    epoch, batch_idx // args.gradient_accumulation_steps, len(train_loader) // args.gradient_accumulation_steps,
                        100. * batch_idx / len(train_loader), loss.item()))
            # optimizer.step()
            
            # if is master process
            # if args.local_rank == 0:
            # print training info
            # if (batch_idx + 1) % args.print_iters == 0:
                # accelerator.print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLM Loss: {:.6f}'.format(
                #     epoch, batch_idx * samples * dist.get_world_size(), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), loss.item()))
                # writer.add_scalar("LM loss", loss.item(), total_iters)
            
            # save model
            if (total_iters // args.gradient_accumulation_steps) % args.save_iters == 0:
                # save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters))
                accelerator.print("save model to {}".format(save_path))
                # torch.save(model.module.state_dict(), save_path)
                accelerator.unwrap_model(model).save_pretrained(save_path)
                processor.save_pretrained(save_path)

                # evaluation
                '''
                if total_iters % args.eval_iters == 0:
                    test_loss = test(args, model, val_loader)
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)
                '''

        # dist.barrier()
        # scheduler.step()

    # save final model
    # if args.local_rank == 0:
        # torch.save(model.module.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        # model.module.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
    accelerator.unwrap_model(model).save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final"))
    processor.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final"))
    # writer.close()

    
    