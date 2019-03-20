from PIL import Image
import torchvision.utils as utils
from torchvision import transforms
import torch
import numpy as np
import cv2
from pathlib import Path

def Unsqueezer(img):
    return img.unsqueeze(0)

Tfms = [transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        Unsqueezer]

ImageTransforms = transforms.Compose(Tfms)

Model = torch.load("Models/BestModel.pth")
Model.eval()


def Forward(inputImgName: str):
    with torch.no_grad():
        outputImgName = inputImgName.with_name(str(inputImgName.stem) + "-seg.pbm")

        img = Image.open(inputImgName)
        img_size = img.size

        img = ImageTransforms(img).cuda()

        res = torch.sigmoid(Model(img))
        res = res.argmax(dim=1)

        to_pil = transforms.ToPILImage()
        p_img = to_pil(res.detach().cpu().type(torch.ByteTensor))

        p_img = p_img.resize(img_size)
        p_img.save(outputImgName)

        return outputImgName
