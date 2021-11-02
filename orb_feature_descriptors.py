import numpy as np 
import cv2 
from matplotlib import pyplot as plt 


def orb_return_features(filename):
	'''
	Takes some filename which is a path to some image file 
	and returns the list of ORB feature descriptors 

	inputs:
		filename: a string which is a filepath to some image file, the 
		one being used for analysis 

	'''
	
	#generate the image
	img = cv2.imread(filename,0)

	# Initiate ORB detector
	orb = cv2.ORB_create()

	# find the keypoints and feature descriptors with ORB
	kp = orb.detect(img,None)
	kp, des = orb.compute(img, kp)

	return (kp,des)


