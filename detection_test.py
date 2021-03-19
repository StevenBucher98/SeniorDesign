import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('DepthTesting/test/L1.JPG', 0)
imgR = cv2.imread('DepthTesting/test/L2.JPG', 0)

print(imgR.shape[:2])

print(imgL.shape[:2])

# f, axes = plt.subplots(1, 2)

# axes[0].imshow(imgL)
# axes[1].imshow(imgR)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
