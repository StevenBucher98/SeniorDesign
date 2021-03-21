import pyzed.sl as sl
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
from PIL import Image
import cv2

now = datetime.now()

print(now)


zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080 	# 1080p
init_params.camera_fps = 30 							# 30 fps
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER


err = zed.open(init_params)

if err != sl.ERROR_CODE.SUCCESS:
	exit(1)


i = 0
image = sl.Mat()
runtime_parameters = sl.RuntimeParameters()

image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
depth_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

	zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
	#zed.retrieve_image(image_zed, sl.VIEW.DEPTH)
	zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
	image_ocv = image_zed.get_data()
	depth_ocv = depth_zed.get_data()
	print(type(depth_ocv))
	print(depth_ocv.shape)
	np.savetxt("depth.csv", depth_ocv, delimiter=",") 
#	cv2.imshow("Depth", depth_ocv)
#	cv2.imshow("Image", image_ocv)
	cv2.imwrite('test.png', image_ocv)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()

'''while i < 50:
    # Grab an image, a RuntimeParameters object must be given to grab()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns ERROR_CODE.SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image

        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp
        print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(), timestamp.get_milliseconds()))
        i = i + 1'''

zed.close()
