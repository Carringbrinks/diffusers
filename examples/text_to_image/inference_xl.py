import os
import json
import torch
from diffusers import DiffusionPipeline


def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def full_test(model_path, test_jsonl_path, save_result_dir):

    os.makedirs(save_result_dir, exist_ok=True)
    prompts = read_jsonl(test_jsonl_path)

    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")

    for prompt in prompts:
        image = pipe(prompt=prompt["text"], num_inference_steps=30, guidance_scale=7.5).images
        for j, img in enumerate(image):
            # img = img.resize((197,100))
            file_name = os.path.basename(prompt["file_name"])
            file_path = os.path.join(save_result_dir, file_name.split(".png")[0]+"_"+str(j)+".png")
            img.save(file_path)


if __name__ == "__main__":

    # 自己训练的模型保存的根目录，和训练脚本的output_dir保持一致即可
    full_model_path = "./weights/sdxl-base-model"

    # 自己定义的测试json，和训练的matedata.jsonl格式保持一致即可
    test_data_path = "./test_data/test_v2.jsonl"

    # 自己定义的图片保存路径根目录
    save_dir = "./test_result/sdxl-base-model"

    # 测试最后一次权重的效果
    full_test(model_path=full_model_path, test_jsonl_path=test_data_path, save_result_dir=save_dir)

    