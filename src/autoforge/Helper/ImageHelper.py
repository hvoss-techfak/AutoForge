import cv2


def resize_image(img, max_size):
    h_img, w_img, _ = img.shape
    if w_img >= h_img:
        new_w = max_size
        new_h = int(max_size * h_img / w_img)
    else:
        new_h = max_size
        new_w = int(max_size * w_img / h_img)
    img_out = cv2.resize(img, (new_w, new_h))
    return img_out
