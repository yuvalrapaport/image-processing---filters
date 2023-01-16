import numpy as np
import cv2


def XFilter2D(img, kernelX):
    kerSize = len(kernelX)-1
    bordered = np.pad(img, pad_width=kerSize//2, mode='constant', constant_values=0)
    rows, cols = bordered.shape[:2]
    grad = np.zeros(img.shape[:2], dtype=img.dtype)

    
    for y in range(round((kerSize)/2), round(rows-(kerSize)/2)):
        for x in range(round((kerSize)/2), round(cols-(kerSize)/2)):
            Gx = 0
            for i in range(round(-kerSize/2), round(kerSize/2)+1):
                for j in range(round(-kerSize/2), round(kerSize//2)+1):
                    val = bordered[y+i, x+j]
                    Gx += kernelX[round(kerSize/2)+i][round(kerSize/2)+j] * val

            grad[round(y-kerSize/2), round(x-kerSize/2)] = Gx
            
    return grad, cv2.absdiff(img, grad)


def filter2D(img, kernelX, kernelY, threshold):
    kerSize = len(kernelX)-1
    bordered = np.pad(img, pad_width=kerSize//2, mode='constant', constant_values=0)
    rows, cols = bordered.shape[:2]
    grad = np.zeros(img.shape[:2], dtype=img.dtype)
    
    for y in range(round((kerSize)/2), round(rows-(kerSize)/2)):
        for x in range(round((kerSize)/2), round(cols-(kerSize)/2)):
            Gx = 0
            Gy = 0
            for i in range(round(-kerSize/2), round(kerSize/2)+1):
                for j in range(round(-kerSize/2), round(kerSize//2)+1):
                    val = bordered[y+i, x+j]
                    Gx += kernelX[round(kerSize/2)+i][round(kerSize/2)+j] * val
                    Gy += kernelY[round(kerSize/2)+i][round(kerSize/2)+j] * val

            calc = np.sqrt(Gx*Gx+Gy*Gy)
            # Thresholding
            if threshold == 2:
                if calc > 100:
                    calc = 255
                else:
                    calc = 0
            elif threshold == 3:
                if calc > 200:
                    calc = 255
                else:
                    calc = 0

            grad[round(y-kerSize/2), round(x-kerSize/2)] = calc
    
    return grad, cv2.absdiff(img, grad)


def calc_sobel_kernel(target_shape: tuple[int, int]):
    kerX = np.zeros(target_shape, dtype=np.float32)
    kerY = np.zeros(target_shape, dtype=np.float32)
    indices = np.indices(target_shape, dtype=np.float32)
    cols = indices[0] - target_shape[0] // 2
    rows = indices[1] - target_shape[1] // 2
    squared = cols ** 2 + rows ** 2
    np.divide(cols, squared, out=kerY, where=squared != 0)
    np.divide(rows, squared, out=kerX, where=squared != 0)
    return kerX, kerY

