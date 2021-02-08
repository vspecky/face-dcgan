import os
import cv2 as cv
from tqdm import tqdm

this_path = os.path.dirname(__file__)
img_path = os.path.join(this_path, "faces", "faces")

imgs = os.listdir(img_path)

tqdm_imgs = tqdm(imgs, unit="Images")

for img_p in tqdm_imgs:
    path = os.path.join(img_path, img_p)
    big_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    small_img = cv.resize(big_img, (64, 64))
    cv.imwrite(path, small_img)
