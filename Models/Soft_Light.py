import cv2  # import OpenCV
import numpy as np
from .blend_modes import soft_light

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])

	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def ChangeColor(in_file_name, mask_file_name, desired_color, out_file_name, opacity=0.7, gamma = 1.):
    D_color = desired_color[::-1]
    # Import background image
    background_img_float = cv2.imread(in_file_name, -1)
    background_img_float = cv2.cvtColor(background_img_float, cv2.COLOR_RGB2RGBA).astype(float) 
    # Import mask image
    mask_color_float = cv2.imread(mask_file_name, -1)
    mask_color_float = cv2.cvtColor(mask_color_float, cv2.COLOR_GRAY2RGBA).astype(float)

    shape = mask_color_float.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.array_equal(mask_color_float[i, j, 0:3], [255., 255., 255.]):
                mask_color_float[i, j, 0:3] = D_color

    kernel = np.ones((5, 5), np.float32)/25
    mask_color_float = cv2.filter2D(mask_color_float, -1, kernel)

    blended_img_float = soft_light(background_img_float, mask_color_float, opacity)
    blended_img_uint8 = blended_img_float.astype(np.uint8)  # Convert image to OpenCV
    if gamma != 1.0:
        blended_img_uint8 = adjust_gamma(blended_img_uint8, gamma)
    cv2.imwrite(out_file_name, blended_img_uint8)


if __name__ == "__main__":
    in_file = "Background.jpg"
    in_mask = "Foreground.pbm"
    out_file = "out.png"
    DESIRED_COLOR = [255., 0., 0.]
    ChangeColor(in_file, in_mask, DESIRED_COLOR, out_file)
