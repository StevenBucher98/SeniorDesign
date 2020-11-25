import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = '/Users/stevenbucher/Documents/SCU/SeniorDesign/HomeFlowerImages/flower1.jpeg'
img = cv2.imread(img_path)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# H S V
hsv_color1 = np.asarray([0, 0, 255])
hsv_color2 = np.asarray([255, 20, 255])

mask = cv2.inRange(hsv_img, hsv_color1, hsv_color2)

f, axes = plt.subplots(1, 3)

axes[0].imshow(img)
axes[1].imshow(hsv_img)
axes[2].imshow(mask)
# axes[1].colorbar()
plt.show()
# cv2.imshow("flower",  hsv_img)
# cv2.waitKey(0)
