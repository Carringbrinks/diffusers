from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
import torch

base_model_path = "/home/scb123/huggingface_weight/stable-diffusion-3-medium-diffusers"
controlnet_path = "/home/scb123/huggingface_weight/contronlnet_sd3_medium_custom/"

controlnet = SD3ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet
)
pipe.to("cuda", torch.float16)


control_image = load_image("/home/scb123/PyProject/DeepData/fill50k_src/source/14.png").resize((1024, 1024))
prompt = "beige circle with yellow green background"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, control_image=control_image
).images[0]
image.save("./output.png")