import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from feature_detection_algorithms_implementation import *
from orb_feature_descriptors import orb_return_features

'''
Goal: download the real and fake image datasets and do bruteforce
comparison on both of them, find how the deepfake images relate/ differ
from the real images and then do some analysis on such images.
'''

def brute_force_orb_comparison(real_filename, fake_filename):
	'''
	This function takes two filenames (real_filename and fake_filename)
	and then does bruteforce comparison of there keypoint descriptor image's 
	corresponding keypoint descriptors.

	inputs:
		real_filename: a string which is the path to the real image
		fake_filename: a string which is the path to the fake image 
	'''

	#load the images.....
	real_image = cv2.imread(real_filename,0)
	fake_image = cv2.imread(fake_filename,0)

	#Two corresponding lists of descriptors and keypoints for each image
	real_image_kp = orb_return_features(real_filename)[0]
	real_image_des = orb_return_features(real_filename)[1]
	fake_image_kp = orb_return_features(fake_filename)[0]
	fake_image_des = orb_return_features(fake_filename)[1]

	#create brute_force matcher object 
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	#Match Descriptors 
	matches = bf.match(real_image_des,fake_image_des)

	#Sort them in order of distance 
	matches = sorted(matches, key = lambda x: x.distance)

	#Draw first 10 matches 
	img3 = cv2.drawMatches(real_image,real_image_kp,
		fake_image,fake_image_kp,
		matches[:10],None,flags=2)

	plt.imshow(img3),plt.show()


def return_brute_force_orb_matches(real_filename,fake_filename):
	'''
	This function takes two filenames (real_filename) and (fake_filename)
	and returns a list of the matching objects

	inputs:
		real_filename: a string which is the path to the real image
		fake_filename: a string which is the path to the fake image 
	'''

	#Two corresponding lists of descriptors and keypoints for each image
	real_image_kp = orb_return_features(real_filename)[0]
	real_image_des = orb_return_features(real_filename)[1]
	fake_image_kp = orb_return_features(fake_filename)[0]
	fake_image_des = orb_return_features(fake_filename)[1]

	#create brute_force matcher object 
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	#Match Descriptors 
	matches = bf.match(real_image_des,fake_image_des)

	#Sort them in order of distance 
	matches = sorted(matches, key = lambda x: x.distance)

	#return the matches object
	return matches


#Testing section of the code 
brute_force_orb_comparison(
	'training_fake/easy_100_1111.jpg',
	'training_fake/easy_101_0010.jpg'
	)

#Print Matches 
print(return_brute_force_orb_matches(
	'training_fake/easy_100_1111.jpg',
	'training_fake/easy_101_0010.jpg'
	))


	