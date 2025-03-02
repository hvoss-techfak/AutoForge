import cv2


def resize_image(img, max_size):
    h_img, w_img, _ = img.shape

    # Compute the scaling factor based on the larger dimension
    if w_img >= h_img:
        scale = max_size / w_img
    else:
        scale = max_size / h_img

    # Compute new dimensions and round them
    new_w = int(round(w_img * scale))
    new_h = int(round(h_img * scale))

    # Resize the image with an appropriate interpolation method
    img_out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_out
