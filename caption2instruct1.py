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
import transformers
from tensorboardX import SummaryWriter

from torch import distributed as dist
# from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from instruct_tri2tri.tinyllama_ft.conversation import conv_llama_2

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class Caption2InstructDataset(Dataset):
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
        # self.image_names = os.listdir(data_path)
        # self.image_paths = [os.path.join(data_path, image_name) for image_name in self.image_names]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        # try:
        data = self.datas[index]
        caption = data['conversations'][1]['value']
        return data, caption
        
    
def collate_fn(batch, tokenizer):
    datas = []
    captions = []

    for data, caption in batch:
        captions.append(caption)
        datas.append(data)
    
    # inputs = tokenizer(text=captions,
    #                    return_tensors="pt")
    
    return datas, captions

def get_input_prompt(prompt):
    conv = conv_llama_2.copy()
    roles = conv_llama_2.roles
    conv.append_message(conv_llama_2.roles[0], prompt)
    conv.append_message(conv_llama_2.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/objaverse/cap3d_automated_objaverse_highquality_550k.json", help='root path of dataset')
    parser.add_argument('--save_path', type=str, default="data/objaverse/cap3d_automated_objaverse_highquality_instruct_550k", help='root path of dataset')
    # multi gpu settings
    parser.add_argument("--local-rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--num_chunks', type=int, default=8, help='num chunks')
    parser.add_argument('--chunk_index', type=int, default=0, help='chunk index')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_option()
    torch.cuda.set_device(args.local_rank)
    # accelerator = Accelerator()
    # device = accelerator.device
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
    model_name_or_path = 'checkpoints/tinyllama_caption2instruct_ft'
    model = transformers.LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.float16,
        ).cuda()
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    train_dataset = Caption2InstructDataset(args.dataset_path, args.num_chunks, args.chunk_index)
    # train_sampler = DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    # if args.local_rank == 0:
        # writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))
    # model.vision_model.requires_grad_(False)
    
    model.to(device=device)
    # model, train_loader = accelerator.prepare(model, train_loader)
    target_jsons = []
    for data, caption in tqdm(train_dataset):
        prompt = get_input_prompt(caption)
        inputs = tokenizer(prompt, return_tensors='pt')
        for key, value in inputs.items():
            if type(value) == torch.Tensor:
                inputs[key] = value.to(device=device)
        output = model.generate(**inputs)
        instruct = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        tmp_dict = {}
        print(instruct)
        # caption = data['conversations'][1]['value']
        tmp_dict['image'] = data['image']
        tmp_dict['caption'] = caption
        tmp_dict['instruct'] = instruct
        # print(tmp_dict)
        target_jsons.append(tmp_dict)
    with open(f'{args.save_path}_{args.chunk_index}_{args.num_chunks}.json', 'w') as f:
        json.dump(target_jsons, f)
    
    
    
    # for datas, captions in tqdm(train_loader):
    #     for i in range(len(datas)):
    #         data = datas[i]
    #         caption = captions[i]
    #         prompt = get_input_prompt(caption)
    #         inputs = tokenizer(prompt, return_tensors='pt')
    #         for key, value in inputs.items():
    #             if type(value) == torch.Tensor:
    #                 inputs[key] = value.to(device=device)
    #         output = model.module.generate(**inputs)
    #         instruct = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    #         tmp_dict = {}
    #         # caption = data['conversations'][1]['value']
    #         tmp_dict['image'] = data['image']
    #         tmp_dict['caption'] = caption
    #         tmp_dict['instruct'] = instruct
    #         print(tmp_dict)
    #         target_jsons.append(tmp_dict)
        
    #     # model.module.save_pretrained(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
    # with open(args.save_path, 'w') as f:
    #     json.dump(target_jsons, f)

    
    