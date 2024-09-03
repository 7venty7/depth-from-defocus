import numpy as np
import cv2
import math


def compute_var(if1, if2):
    if (if1.shape != if2.shape):
        return

    M, N = if1.shape

    var_r[M][N] = [[0]]

    for i in range(M):
        for j in range(N):
            var_r[i][j] = -(m**2 * n**2) / (math.pi**2 * (N**2 * i**2 + M**2 * j**2))
            var_r[i][j] *= np.log(if1[i][j] / if2[i][j])
