import torch
import os
from tqdm import tqdm
import torchvision

PATH_TO_MLR3TORCH = "."
cache_dir = "cache"

path_to_melanoma_train = os.path.join(PATH_TO_MLR3TORCH, cache_dir, "train")
path_to_melanoma_test = os.path.join(PATH_TO_MLR3TORCH, cache_dir, "ISIC_2020_Test_Input")

path_to_output_train = os.path.join(PATH_TO_MLR3TORCH, cache_dir, "hf_dataset", "train")
path_to_output_test = os.path.join(PATH_TO_MLR3TORCH, cache_dir, "hf_dataset", "ISIC_2020_Test_Input")

os.makedirs(path_to_output_train)
os.makedirs(path_to_output_test)

tx = torchvision.transforms.Resize((128, 128))

for f in tqdm(os.listdir(path_to_melanoma_train)):
    img = torchvision.io.read_image(os.path.join(path_to_melanoma_train, f))
    small_img = tx(img.float() / 255)
    torchvision.utils.save_image(small_img, os.path.join(path_to_output_train, f))

for f in tqdm(os.listdir(path_to_melanoma_test)):
    if f.endswith(".jpg"):
        img = torchvision.io.read_image(os.path.join(path_to_melanoma_test, f))
        small_img = tx(img.float() / 255)
        torchvision.utils.save_image(small_img, os.path.join(path_to_output_test, f))
