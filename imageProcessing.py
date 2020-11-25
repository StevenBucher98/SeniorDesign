import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = '/Users/stevenbucher/Documents/SCU/SeniorDesign/HomeFlowerImages/flower1.jpeg'
img = cv2.imread(img_path)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# H S V
hsv_color1 = np.asarray([0, 0, 255])
hsv_color2 = np.asarray([255, 20, 255])

mask = cv2.inRange(hsv_img, hsv_color1, hsv_color2)
res = cv2.bitwise_and(img, img, mask=mask)

f, axes = plt.subplots(2, 3)

edges = cv2.Canny(img, 100, 200)
gray_edges = cv2.Canny(gray_img, 100, 200)

# img = cv.imread('j.png',0)
kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

axes[0][0].imshow(img)
axes[0][1].imshow(hsv_img)
axes[0][2].imshow(gray_img)
axes[1][0].imshow(res)
axes[1][1].imshow(edges)
axes[1][2].imshow(opening)

# axes[1].colorbar()
plt.show()
# cv2.imshow("flower",  hsv_img)
# cv2.waitKey(0)
