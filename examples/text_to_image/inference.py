import os
import json
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def full_test(model_path, test_jsonl_path, save_result_dir):
    os.makedirs(save_result_dir, exist_ok=True)
    prompts = read_jsonl(test_jsonl_path)

    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")

    for prompt in prompts:
        image = pipe(prompt=prompt["text"]).images
        for j, img in enumerate(image):
            # img = img.resize((197,100))
            file_name = os.path.basename(prompt["file_name"])
            file_path = os.path.join(save_result_dir, file_name.split(".png")[0]+"_"+str(j)+".png")
            img.save(file_path)

def ckpt_test(ckpt_suffix, model_path, initial_model, test_jsonl_path, save_result_dir, unet_type="unet"):

    model_path =model_path + "/checkpoint-{}/" + unet_type
    for ckpt in ckpt_suffix:
        model_path = model_path.format(ckpt)
        unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(initial_model, unet=unet, torch_dtype=torch.float16)
        pipe.to("cuda")
        prompts = read_jsonl(test_jsonl_path)
        for prompt in prompts:
            image = pipe(prompt=prompt["text"]).images
            for j, img in enumerate(image):
                # img = img.resize((197,100))
                file_name = os.path.basename(prompt["file_name"])
                save_dir = save_result_dir+"/checkpoint-{}-{}/".format(ckpt, unet_type)
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, file_name.split(".png")[0]+"_"+str(j)+".png")
                img.save(file_path)


if __name__ == "__main__":

    # 自己训练的模型保存的根目录，和训练脚本的output_dir保持一致即可
    full_model_path = "./weights/stable-diffusion-2-full-v2-v3"

    # 官方的预训练权重，和训练脚本的pretrained_model_name_or_path保持一直即可
    initial_model_path = "/home/scb123/HuggingfaceWeight/stable-diffusion-2"

    # 自己定义的测试json，和训练的matedata.jsonl格式保持一致即可
    test_data_path = "./test_data/test_v3.jsonl"

    # 自己定义的图片保存路径根目录
    save_dir = "./test_result/v2-v3"

    # 测试最后一次权重的效果
    full_test(model_path=full_model_path, test_jsonl_path=test_data_path, save_result_dir=save_dir)

    # 测试保存的checkpoint的效果，这里需要指定checkpoint的后缀
    ckpt_suffixs= ["800", "1600"]

    # 测试unet和unet_ema两种格式权重
    unet_type = "unet"
    ckpt_test(ckpt_suffix=ckpt_suffixs, model_path=full_model_path, initial_model=initial_model_path, 
              test_jsonl_path=test_data_path, save_result_dir=save_dir, unet_type=unet_type)