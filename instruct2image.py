import os
import json
import random
import rembg
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from instruct_tri2tri.tsr.utils import remove_background, resize_foreground

class Instruct2ImageDataset(Dataset):
    def __init__(self,
                 data_path,
                 num_chunks,
                 chunk_index):
        super().__init__()
        datas = json.load(open(data_path))
        self.datas = datas
        data_len = len(datas)
        chunks = data_len // num_chunks
        if (chunk_index + 1) != num_chunks:
            self.datas = datas[chunk_index*chunks: chunk_index*chunks + chunks]
        else:
            self.datas = datas[chunk_index*chunks:]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        image_name = data['image']
        image = Image.open(f'data/{image_name}')
        image_name = image_name.split('/')[-1]
        instruct = data['instruct']
        return image_name, image, instruct

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/objaverse/cap3d_automated_objaverse_highquality_instruct_550k.json", help='root path of dataset')
    parser.add_argument('--save_path', type=str, default="data/objaverse/instruct_images", help='root path of dataset')
    # multi gpu settings
    parser.add_argument("--local-rank", type=int, default=-1)
    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--num_chunks', type=int, default=4, help='num chunks')
    parser.add_argument('--chunk_index', type=int, default=0, help='chunk index')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_option()
    torch.cuda.set_device(args.local_rank)
    device = 'cuda'

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    model_id = "ckpts/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    train_dataset = Instruct2ImageDataset(args.dataset_path, args.num_chunks, args.chunk_index)
    rembg_session = rembg.new_session()
    os.makedirs(args.save_path, exist_ok=True)
    for image_name, image, instruct in tqdm(train_dataset):
        image = pipe(instruct, image=image, num_inference_steps=10, image_guidance_scale=1).images[0]
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, 0.85)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        image.save(f'{args.save_path}/{image_name}.jpg')

    
    