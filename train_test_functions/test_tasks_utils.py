import cv2
import numpy as np

def get_psnr_rmse(img1, img2, mask=None):
    if mask is not None:
        rmse=np.sqrt(np.mean(mask*img1 - mask*img2) ** 2)
    else:
        rmse=np.sqrt(np.mean(img1 - img2) ** 2)
    if rmse == 0:
        return 100,rmse
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / rmse),rmse

def affine_transform(original_image, angle=0,shear=0,translation=0,type_border=cv2.BORDER_CONSTANT):

    color_border = (0, 0, 0)
    rows, cols, ch = original_image.shape

    # First: Necessary space for the rotation
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cos_part = np.abs(M[0, 0])
    sin_part = np.abs(M[0, 1])
    new_cols = int((rows * sin_part) + (cols * cos_part))
    new_rows = int((rows * cos_part) + (cols * sin_part))

    # Second: Necessary space for the shear
    new_cols += (shear * new_cols)
    new_rows += (shear * new_rows)

    # Calculate the space to add with border
    up_down = int(np.ceil((new_rows - rows) / 2))
    left_right = int(np.ceil((new_cols - cols) / 2))

    final_image = cv2.copyMakeBorder(original_image, up_down, up_down, left_right, left_right, type_border,value=color_border)
    rows, cols, ch = final_image.shape

    # Application of the affine transform.
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    translat_center_x = -(shear * cols) / 2
    translat_center_y = -(shear * rows) / 2

    M = M_rot + np.float64([[0, shear, translation + translat_center_x], [shear, 0, translation + translat_center_y]])
    M_inv=np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:-1,:]
    final_image = cv2.warpAffine(final_image, M_inv, (cols, rows), borderMode=type_border, borderValue=color_border, flags=cv2.WARP_INVERSE_MAP)
    return final_image, M
