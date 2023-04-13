import base64
import json
import pathlib

import numpy as np
from PIL import Image
from io import BytesIO


def convert(path_img:pathlib.Path) -> dict:
    label = path_img.parent.name  # Directory name
    mean, std =  mean_std(path_img)

    with open(path_img, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    d = {
        "piece":label,
        "characters":None,
        "font":None,
        "mean":mean,
        "std":std,
        "image":img_base64
    }
    return d


def mean_std(path_img):
    im = Image.open(path_img)
    img = np.array(im)  # (64, 64, 3)
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]
    img_mean = [np.mean(img_r)/255, np.mean(img_g)/255, np.mean(img_b)/255]
    img_std = [np.std(img_r)/255, np.std(img_g)/255, np.std(img_b)/255]
    return img_mean, img_std



def read_json(path_json:pathlib.Path) -> dict:
    with open(path_json, "r") as f:
        j = json.load(f)


    print("0000000000000000000000000000000000000000")

    img_base64 = j[0]["image"]
    # print(img_base64)
    print(len(j))

    img_raw = base64.b64decode(img_base64)

    print(type(img_raw))

    im = Image.open(BytesIO(img_raw))
    print(im.format, im.size, im.mode)


    img = np.array(im)  # (64, 64, 3)
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]

    img_mean = [np.mean(img_r)/255, np.mean(img_g)/255, np.mean(img_b)/255]
    img_std = [np.std(img_r)/255, np.std(img_g)/255, np.std(img_b)/255]

    print(img_mean)
    print(img_std)





if __name__ == '__main__':
    path_current_dir = pathlib.Path(__file__).parent

    d_list = []


    path_img = path_current_dir.joinpath("train_images", "hi", "1_10.png")

    d = convert(path_img)
    # print(d)

    d_list.append(d)

    with open('tmp.json', 'wt') as f:
        json.dump(d_list, f, indent=2, ensure_ascii=False)


    read_json(path_current_dir.joinpath("tmp.json"))


        