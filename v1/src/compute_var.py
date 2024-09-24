import numpy as np
import cv2
import math


def compute_var(if1, if2):
    if (if1.shape != if2.shape):
        return

    M, N = if1.shape

    var_r = np.zeros((M, N))

    for x in range(M):
        for y in range(N):
            count = 0

            for i in range(-1, 2):
                if x + i >= 1 and x + i < M:
                    for j in range(-1, 2):
                        if y + j >= 1 and y + j < N:
                            count += 1
                            var_r[x][y] += -(M**2 * N**2) / (math.pi**2 * (N**2 * (x+i)**2 + M**2 * (y+j)**2)) * np.log(if1[x+i][y+j] / if2[x+i][y+j])

            var_r[x][y] /= count

    return var_r
