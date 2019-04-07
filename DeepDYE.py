#coding=utf-8
# Vistas de la interfaz de Voluntario
import os, sys, uuid, argparse
from pathlib import Path
from shutil import copyfile
import PIL
from Models import NeuralModel, Soft_Light

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('input_image', metavar='Image', type=Path,
                    help='The Image to Dye')
parser.add_argument('color', metavar='Color', type=str,
                    help='The color to dye the image')
parser.add_argument('-o', "--output", default="generated.png",
                    help='Output the image')
parser.add_argument("--opacity", default=0.7, type=float,
                    help='Opacity for the blend mode')

args = parser.parse_args()

COLORES = {"red":[255.,0.,0.], "green":[0.,255.,0.], "blue" : [0.,0.,255.],
          "pink":[255.,64.,195.], "white":[255.,255.,255.]}

GENERATED = Path("./Generated")
if not GENERATED.exists():
    GENERATED.mkdir()

def MainProgram(input_image, color, output_image, opacity):
    UniqueID = uuid.uuid4().hex
    ImgSaveFilename = (GENERATED/UniqueID).with_suffix(input_image.suffix)

    try:
        Img = PIL.Image.open(input_image)
        Img.save(ImgSaveFilename)

        ColorRGB = COLORES.get(color, None) 
        if ColorRGB is None:
            raise KeyError

        OutFilename = ProcessImage(ImgSaveFilename, ColorRGB, opacity)
        copyfile(str(OutFilename), output_image)

    except Exception as e:
        print("Program terminated due to")
        print(e)

def ProcessImage(ImageFileName, Color, opacity):
    # Pass through Neuro
    OutMask = NeuralModel.Forward(ImageFileName, Color)
    OutFileName = ImageFileName.with_name(str(ImageFileName.stem) + "Processed.png")

    # Color
    Soft_Light.ChangeColor(str(ImageFileName), str(OutMask), str(OutFileName), opacity)

    return OutFileName

if __name__ == "__main__":
    args = parser.parse_args()
    MainProgram(args.input_image, args.color, args.output, args.opacity)
