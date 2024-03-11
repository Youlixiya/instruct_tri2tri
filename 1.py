import os
import rembg
from PIL import Image
import numpy as np
from tqdm import tqdm
from instruct_tri2tri.tsr.utils import remove_background, resize_foreground

image_names = os.listdir('data/Cap3D_imgs_view0')
save_path = 'data/images'
foreground_ratio = 0.85
rembg_session = rembg.new_session()
os.makedirs(save_path, exist_ok=True)
for image_name in tqdm(image_names):
    image = Image.open(os.path.join('data/Cap3D_imgs_view0', image_name))
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, foreground_ratio)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    image.save(f'{save_path}/{image_name}.jpg')