from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import fastai.vision as fv
import fastai.basics as fai

to_pil = transforms.ToPILImage()
Learn = fai.load_learner("./Models", "HairLearner.pkl")

def Forward(inputImgName: str, Color):
    outputImgName = inputImgName.with_name(str(inputImgName.stem) + "-seg.png")

    Img = fv.open_image(inputImgName)
    originalSize = Img.size
    Img = Img.resize(500)
    Res = Learn.predict(Img)[0]

    # Colorization
    Mask = (Res.data == 255)
    R, G, B, A = [torch.zeros((1, 500, 500), dtype=torch.uint8) for _ in range(4)]
    R[Mask], G[Mask], B[Mask] = Color
    A[Mask] = 255
    ColorMask = fv.Image(torch.cat([R, G, B, A]))

    Pil_Img = to_pil(ColorMask.data.detach().cpu().type(torch.ByteTensor))
    Pil_Img = Pil_Img.resize(originalSize[::-1])
    Pil_Img.save(outputImgName)

    return outputImgName
