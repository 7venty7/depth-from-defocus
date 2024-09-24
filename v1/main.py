from src.image_align import align_images
from src.fourier import fourier
from src.compute_var import compute_var

import sys
import cv2
import matplotlib.pyplot as plt

im1 = cv2.imread("./src/dataset/05/01.JPG", cv2.IMREAD_COLOR)
im2 = cv2.imread("./src/dataset/05/53.JPG", cv2.IMREAD_COLOR)

# align the 2nd image with the first
print("aligning")
aligned2, h = align_images(im2, im1)

# compute the fourier transformation of the images
print("computing fourier transforms")
if1 = fourier(im1)
if2 = fourier(aligned2)

print("computing variance")
var_r = compute_var(if2, if1)
print(var_r)
M, N = var_r.shape

#map = cv2.applyColorMap(var_r, cv2.COLORMAP_JET)
#cv2.imwrite("map.jpg", map)

for x in range(0, M):
    for y in range(0, N):
        var_r[x][y] /= var_r.max()

plt.imshow(var_r)
plt.title("variance")
plt.show()
