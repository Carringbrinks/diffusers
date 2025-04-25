import re
import cv2
import os
import json
from tqdm import tqdm

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def scale_coordinates(text, factor=1.5):
    def repl(match):
        x = float(match.group(1)) * factor
        y = float(match.group(2)) * factor
        return f"({x:.2f}, {y:.2f})"

    return re.sub(r"\(([\d.]+),\s*([\d.]+)\)", repl, text)


def resize_image(image_dir, save_image_dir):
    for img_file_name in tqdm(os.listdir(image_dir)):
        # print(img_file_name)
        img_path= os.path.join(image_dir, img_file_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (768, 768))
        img_file_name = scale_coordinates(img_file_name)
        # print(os.path.join(save_image_dir, img_file_name))
        cv2.imwrite(os.path.join(save_image_dir, img_file_name), img)



if __name__ == '__main__':

    img_dir = "/home/scb123/PyProject/diffusers/examples/text_to_image/train_data/data_v3/images"
    save_img_dir = "/home/scb123/PyProject/diffusers/examples/text_to_image/train_data/data_v3_768/images"
    os.makedirs(save_img_dir, exist_ok=True)
    matedata =  os.path.join(os.path.dirname(img_dir), "metadata.jsonl")
    new_matedata = os.path.join(os.path.dirname(save_img_dir), "metadata.jsonl")

    # resize_image(img_dir, save_img_dir)
    json_data = read_jsonl(matedata)
    new_datas = []
    for data in json_data:
        text = scale_coordinates(data["text"])
        file_name = scale_coordinates(data["file_name"])
        new_data = {"file_name":data["file_name"], "text":text.replace("512x512", "768x768")}
        new_datas.append(new_data)
    save_jsonl(new_datas, new_matedata)
