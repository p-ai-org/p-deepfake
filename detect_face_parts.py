from imutils import face_utils
import numpy as np 
import argparse
import imutils 
import dlib 
import cv2


'''
CODE SOURCE: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
'''

def extract_facial_regions(image):
	'''
	Takes some image file and then extracts it into subimages containing the follow facial regions:
	mouth, right eyebrow, left eyebrow, right eye, left eye, nose jaw
	'''

	#a dict of the images
	region_images = {}

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(image)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

				# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
			#append this image to the list of images
			region_images[str(name)] = roi
			# show the particular face part
			cv2.imshow("ROI", roi)
			cv2.imshow("Image", clone)
			cv2.waitKey(0)
		# visualize all facial landmarks with a transparent overlay
		output = face_utils.visualize_facial_landmarks(image, shape)
		cv2.imshow("Image", output)
	cv2.waitKey(0)

	#return the list of regions
	return region_images


#Testing if ran directly
if __name__ == "__main__":
	image = 'training_fake/easy_100_1111.jpg'
	regions = extract_facial_regions(image)
	for image in regions.values():
		cv2.imshow("Region", image)
		cv2.waitKey(0)
	
