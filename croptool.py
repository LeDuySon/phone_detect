import argparse
import cv2
import numpy as np
import os

global boxes # the list of selected regions
global cropping # boolean indicating whether cropping is being performed or not
global image 
global clone # clone of the image
boxes = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# if the left mouse button was clicked, record the starting (x, y) coordinates 
	# and indicate that cropping is being performed
	if event == cv2.EVENT_LBUTTONDOWN:
		boxes.append((x, y, 0, 0)) # ending coordinates not specified yet, default to 0
		cropping = True
		# visualize reference boxes to crop more precisely
		cv2.rectangle(clone, (x,y), (x+80,y+80), (0, 200, 0), 2)
		cv2.rectangle(clone, (x,y), (x+120,y+120), (0, 200, 0), 2)
		cv2.rectangle(clone, (x,y), (x+200,y+200), (0, 200, 0), 2)
		cv2.imshow("image", clone)
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates to the last box(newly appended to the list)
		# and indicate that the cropping operation is finished
		x1 = boxes[-1][0]
		y1 = boxes[-1][1]
		boxes[-1] = (x1,y1,x,y)
		cropping = False
		# show the chosen regions
		displayROIs()

def displayROIs():
	# assign original image to clone
	np.copyto(clone, image)
	# draw selected boxes on clone, orginal image remained unchanged
	for box in boxes:
		x1,y1,x2,y2 = box
		cv2.rectangle(clone, (x1,y1), (x2,y2), (0, 255, 0), 2)
	cv2.imshow("image", clone)

def create_project_dir(directory):
    if not os.path.exists(directory):
        print('Creating directory ' + directory)
        os.makedirs(directory)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="path to folder contains images")
ap.add_argument("-i", "--image", help="Path to the image")
ap.add_argument("-dst", "--destination", required=True, help="destination where the cropped images are stored")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])
# print(image)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

#create folder if not exists
create_project_dir(args["destination"])
dst = args["destination"]

images_link = os.listdir(args["folder"])
print(images_link)
# keep looping until the 'q' key is pressed
for index, img in enumerate(images_link[15:]):
	path2img = os.path.join(args["folder"], img)
	print("{}- Crop Image {}".format(index, path2img))
	image = cv2.imread(path2img)
	if(image.shape[0] > 2500 or image.shape[1] > 2500):
		image = cv2.resize(image, (1024, 1024))
	clone = image.copy()
	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", clone)
		key = cv2.waitKey(1) & 0xFF
		# if the 'r' key is pressed, remove the last region
		if key == ord("r") and len(boxes) > 0:
			boxes.pop()
			displayROIs()
		# if the 'c' key is pressed, break from the loop, to save the cropped regions
		elif key == ord("c"):
			break
		# if the 'e' key is pressed, quit
		elif key == ord("e"):
			cv2.destroyAllWindows()
			quit()
	if(len(boxes) > 0):
		for i in range(len(boxes)):
			try:
				x1,y1,x2,y2 = boxes[i]
				roi = image[y1:y2, x1:x2]
				cv2.imwrite(dst+"/"+str(i+index).zfill(3)+".png", roi)
			except:
				print("Some error occurs!!")
				break
	boxes = []

# close all open windows
cv2.destroyAllWindows()