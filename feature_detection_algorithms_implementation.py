import numpy as np
import cv2
from matplotlib import pyplot as plt

def orb_feature_detection(filename):
	'''
	Takes the filepath for an image and returns another image of that image, but 
	with orb features overlayed

		input: 
			filename: the filepath for a givenimage
	'''

	img = cv2.imread(filename,0)
	
	# Initiate ORB detector
	orb = cv2.ORB_create()

	# find the keypoints and feature descriptors with ORB
	kp = orb.detect(img,None)
	kp, des = orb.compute(img, kp)

	# draw only keypoints location,not size and orientation
	img2 = cv2.drawKeypoints(img, kp, None, color = (0, 255, 0), 
	                    flags = 0)

	#Draw the images
	plt.imshow(img2)
	plt.show()

def fast_feature_detection(filename):
	'''
	fast_feature_detection takes some filepaath for some image named filename
	and returns an image with the FAST keypoints overlayed over it.

		input: 
			filename: the filepath for the givenimage
	'''

	#load the file into an image obj
	img = cv2.imread(filename,0)

	#create the FAST Feature Detection, and keypoints 
	fast = cv2.FastFeatureDetector_create()
	kp = fast.detect(img, None)


	# draw only keypoints location,not size and orientation
	img2 = cv2.drawKeypoints(img, kp, None, color = (0, 255, 0), 
                    flags = 0)

	#Draw the images
	plt.imshow(img2)
	plt.show()

def sift_feature_detection(filename):
	'''
	Takes an image file given by filename which is the path to this file,
	and returns an image with sift keypoints overlayed over it.

		input:
			filename: the filepath for the image that is being SIFTED
	'''

	#create the img object alongside a grayscale version of it
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#initialize the sift object, get the keypoints, and then draw the keypoints
	#onto the image
	sift = cv2.SIFT_create()
	kp = sift.detect(gray,None)
	img = cv2.drawKeypoints(gray,kp,img)


	#Display our image
	plt.imshow(img)
	plt.show()

