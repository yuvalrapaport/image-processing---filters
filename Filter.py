import cv2
import numpy as np
import sys
from HelperFunctions import filter2D, XFilter2D, calc_sobel_kernel


imgPath = sys.argv[1]


# Read the original image
img = cv2.imread(imgPath, 0)
img = np.float64(img)


# Choosing kernel size:
kerSize = int(input("Enter the kernel size from 3,5,7,9 : "))

# Choosing Mode
mode = input(
    "Choose number for mode between \n 1)edge detection \n 2)smoothing\n 3)create kernel\n")

# user kernel
if mode == '3':
    print("Enter " + str(kerSize**2) +
          " numbers in a single line separated by space: ")
    # User input of entries in a
    # single line separated by space
    entries = list(map(int, input().split()))

    # For printing the matrix
    kernel = np.array(entries).reshape(kerSize, kerSize)
    kernel = kernel/np.sum(kernel) if np.sum(kernel) != 0 else kernel
    print("you created kernel: " + str(kernel))
    output, diff = XFilter2D(img, kernel)

    # Display original image
    cv2.imshow('Original', img.astype(np.uint8))

    # write and show filtered image
    cv2.imwrite('user_kernel_img.jpg', output)
    cv2.imshow('user_kernel_img', cv2.imread('user_kernel_img.jpg'))

    # write and show diff image
    cv2.imwrite('difference_user_kernel_img.jpg', diff)
    cv2.imshow('difference_user_kernel_img', cv2.imread(
        'difference_user_kernel_img.jpg'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# smoothing
elif mode == '2':
    kernel = np.ones((kerSize, kerSize))
    kernel = kernel/np.sum(kernel)
    output, diff = XFilter2D(img, kernel)

    # Display original image
    cv2.imshow('Original', img.astype(np.uint8))

    # write and show filtered image
    cv2.imwrite('smoothed_img.jpg', output)
    cv2.imshow('smoothed', cv2.imread('smoothed_img.jpg'))

    # write and show diff image
    cv2.imwrite('difference_smoothed_img.jpg', diff)
    cv2.imshow('difference_smoothed', cv2.imread(
        'difference_smoothed_img.jpg'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Edge detection
elif mode == '1':
    threshold = int(input(
        "Please choose option number:\n1)no threshold\n2)threshold 100\n3)threshold 200\n"))
    if threshold not in [1, 2, 3]:
        print("You chose Wrong!")

    # calculate kernels
    kerX, kerY = calc_sobel_kernel((kerSize, kerSize))
    if kerSize == 3:
        kerX, kerY = kerX*2, kerY*2

    # convolution edge detection
    edge_detection_image, diff = filter2D(img, kerX, kerY, threshold)

    # Display original image
    cv2.imshow('Original', img.astype(np.uint8))

    # write and show filtered image
    cv2.imwrite('edge_detection_image.jpg', edge_detection_image)
    cv2.imshow('Sobel Edge Detection', cv2.imread('edge_detection_image.jpg'))

    # write and show diff image
    cv2.imwrite('difference_edge_detection_image.jpg', diff)
    cv2.imshow('difference_sobel Edge Detection', cv2.imread(
        'difference_edge_detection_image.jpg'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Wrong Value")
