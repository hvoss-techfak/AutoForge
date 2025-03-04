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
    img_out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # INTER_CUBIC is actually not a good choice for shrinking images. Except in our case.
    # As we shrink the image, we lose information and the image becomes blurry with INTER_AREA. This is detrimental for the solver. If you want to try it out be my guest.
    # INTER_AREA actually destroys a lot of the colors of our solver.
    return img_out
