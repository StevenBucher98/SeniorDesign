import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg') # Makes you able to use png's with OPENCV

# img_path = 'HomeFlowerImages/flower1.jpeg'
# img_path = 'flowers2.jpg'
img_path = 'Test_Depths/test_5_2021-02-03_23:29:49.png'
# img_path = 'Test_Depths/test_8_2021-02-03_22:55:35.png'


# img = cv2.imread(img_path)
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # USE FOR PNG

# Convert to HSV and Greyscale respectively
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# HSV value range
hsv_color1 = np.asarray([0, 0, 255])
hsv_color2 = np.asarray([255, 20, 255])

# Filter out the specific colors from above
mask = cv2.inRange(hsv_img, hsv_color1, hsv_color2)
res = cv2.bitwise_and(img, img, mask=mask)

# Kernals for morphological transformations
kernel = np.ones((5, 5), np.uint8)
big_kernal = np.ones((50, 50), np.uint8)

# initally opening of the result of the filtered colors
opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

# dilation with small kernal
dilation = cv2.dilate(opening, kernel, iterations=10)

# bigger kernal closing 
final = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, big_kernal)

# convert to greyscale for finding contours
final_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
im2, contours, hierarchy = cv2.findContours(final_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(img, contours, -1, (0,0,0), 10)
print("contor len", len(contours))

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    print("X Y W H")
    print(x, y, w, h)
    print("________________")
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 10)

f, axes = plt.subplots(1, 1)
axes.imshow(img)
# axes[0].imshow(dilation)
# axes[1].imshow(final)


# axes[0][0].imshow(img)
# axes[0][1].imshow(hsv_img)
# axes[0][2].imshow(gray_img)
# axes[1][0].imshow(res)
# axes[1][1].imshow(dilation)
# axes[1][2].imshow(final_gray)

# axes[1].colorbar()
plt.show()
# cv2.imshow("flower",  hsv_img)
# cv2.waitKey(0)
