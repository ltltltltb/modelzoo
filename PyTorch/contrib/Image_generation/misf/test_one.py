# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import torch
from src.config import Config
from src.models import InpaintingModel
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', required=True, type=str, default='./data/image/10.jpg', help='image path')
parser.add_argument('--mask_path', required=True, type=str, default='./data/mask/10_mask.png', help='mask path')
parser.add_argument('--model_path', required=True, type=str, default='./checkpoints/places2_InpaintingModel_gen.pth', help='pretrained model')
args = parser.parse_args()

print(args.img_path)
print(args.mask_path)
print(args.model_path)



def resize(img, height=256, width=256, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = cv2.resize(img, dsize=(height, width))

    return img


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def to_img(data):
  data = data * 255.0

  data = data.detach().cpu().permute(1, 2, 0).numpy()
  data = np.clip(data, 0, 255)
  data = data.astype(np.uint8)
  return data

config_path = './checkpoints/config.yml'
# load config file
config = Config(config_path)

inpaint_model = InpaintingModel(config)

data = torch.load(args.model_path, map_location=lambda storage, loc: storage)
inpaint_model.generator.load_state_dict(data['generator'])
print('the model is loaded')

mask = Image.open(args.mask_path)
mask = np.array(mask)
mask = resize(mask)
mask = (mask > 0).astype(np.uint8) * 255
mask = to_tensor(mask).unsqueeze(dim=0)

img = Image.open(args.img_path)
img = np.array(img)
img = resize(img)
img = to_tensor(img).unsqueeze(dim=0)

if torch.cuda.is_available():
    img = img.cuda()
    mask = mask.cuda()
    inpaint_model = inpaint_model.cuda()

img_masked = img * (1 - mask)
input = torch.cat((img_masked, mask), dim=1)
output = inpaint_model.generator(input)

output = to_img(output[0])
output = Image.fromarray(output)
output.save('./data/result/result.png')

print('the result is saved in ./data/result')