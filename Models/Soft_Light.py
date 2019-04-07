import cv2  # import OpenCV
import numpy as np
from .blend_modes import soft_light

def ChangeColor(in_file_name, mask_file_name, out_file_name, opacity=0.7):
    # Import background image
    background_img_float = cv2.imread(in_file_name, -1)
    background_img_float = cv2.cvtColor(background_img_float, cv2.COLOR_RGB2RGBA).astype(float) 
    # Import mask image
    mask_color_float = cv2.imread(mask_file_name, -1).astype(float)

    # Blurry using 2D Conv
    kernel = np.ones((5, 5), np.float32)/25
    mask_color_float = cv2.filter2D(mask_color_float, -1, kernel)

    # Blend images
    blended_img_float = soft_light(background_img_float, mask_color_float, opacity)
    blended_img_uint8 = blended_img_float.astype(np.uint8)  # Convert image to OpenCV
    cv2.imwrite(out_file_name, blended_img_uint8)
