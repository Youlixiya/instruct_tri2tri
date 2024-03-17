import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

from instruct_tri2tri.tsr.pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
from instruct_tri2tri.tsr.system import TSR
from instruct_tri2tri.tsr.utils import (
    remove_background,
    resize_foreground,
    to_gradio_3d_orientation,
    ImagePreprocessor,
    find_class,
)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

image_preprocessor = ImagePreprocessor()
latent_image_tokenizer_cls = "tsr.models.tokenizers.latentimage.DINOLatentImageTokenizer"
latent_image_tokenizer_dict = {}
latent_image_tokenizer = find_class(latent_image_tokenizer_cls)(latent_image_tokenizer_dict)
latent_image_tokenizer.load_state_dict(torch.load('checkpoints/instruct_tri2tri/latent_image_tokenizer.ckpt'))
latent_image_tokenizer = latent_image_tokenizer.to(device)
latent_image_tokenizer.requires_grad_(False)
# del latent_image_tokenizer.vae

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.requires_grad_(False)
# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(4096)
del model.image_tokenizer
model.to(device)

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained('ckpts/instruct-pix2pix', torch_dtype=torch.float16, safety_checker=None)
pipe.to(device)

rembg_session = rembg.new_session()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, instruct):
    image = Image.fromarray(image)
    edited_image = image
    with torch.no_grad():
        if instruct:
            pipe_output = pipe(instruct, image=image.resize((512, 512)), num_inference_steps=10, image_guidance_scale=2)
            latent_images = pipe_output.latents
            edited_image = pipe_output.images[0]
            input_image_tokens = latent_image_tokenizer(latent_images = latent_images)
        else:
            image_pt = image_preprocessor(image, 512).permute(0, 3, 1, 2).to(device)
            input_image_tokens = latent_image_tokenizer(images=image_pt)
        scene_codes = model(input_image_tokens=input_image_tokens, device=device)
        # scene_codes = model(image, device=device)
        mesh = model.extract_mesh(scene_codes)[0]
        mesh = to_gradio_3d_orientation(mesh)
        mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        mesh.export(mesh_path.name)
        return mesh_path.name, edited_image


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name = generate(preprocessed)
    return preprocessed, mesh_name


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # TripoSR Demo
    [TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image, collaboratively developed by [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/).
    
    **Tips:**
    1. If you find the result is unsatisfied, please try to change the foreground ratio. It might improve the results.
    2. You can disable "Remove Background" for the provided examples since they have been already preprocessed.
    3. Otherwise, please disable "Remove Background" option only if your input image is RGBA with transparent background, image contents are centered and occupy more than 70% of image width or height.
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
                edited_image = gr.Image(label="Edited Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    instruct = gr.Text(
                        label='Edit Instruct'
                    )
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
        with gr.Column():
            with gr.Tab("Model"):
                output_model = gr.Model3D(
                    label="Output Model",
                    interactive=False,
                )
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/hamburger.png",
                "examples/poly_fox.png",
                "examples/robot.png",
                "examples/teapot.png",
                "examples/tiger_girl.png",
                "examples/horse.png",
                "examples/flamingo.png",
                "examples/unicorn.png",
                "examples/chair.png",
                "examples/iso_house.png",
                "examples/marble.png",
                "examples/police_woman.png",
                "examples/captured.jpeg",
            ],
            inputs=[input_image],
            outputs=[processed_image, edited_image, output_model],
            cache_examples=False,
            fn=partial(run_example),
            label="Examples",
            examples_per_page=20,
        )
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, instruct],
        outputs=[output_model, edited_image],
    )

demo.queue(max_size=1)
demo.launch()
