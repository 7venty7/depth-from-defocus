import cv2
import numpy as np


def fourier(im):
    # convert the image to greyscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # compute the discrete fourier transformation of the image
    dft = cv2.dft(np.float32(im_gray), flags=cv2.DFT_COMPLEX_OUTPUT)

    # shift the 0 frequency component to the center
    shifted = np.fft.fftshift(dft)

    # compute the magnitude of the fourier transform
    magnitude = 20 * np.log(cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1]))

    return magnitude


if __name__ == '__main__':
    im01_file = "./dataset/05/01.JPG"

    im01 = cv2.imread(im01_file, cv2.IMREAD_COLOR)

    im_reg = fourier(im01)
