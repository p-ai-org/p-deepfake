import cv2

def orb_return_features_2(image):
	'''
	Takes some image object 
	and returns the list of ORB feature descriptors 

	inputs:
		image: some image object..

	'''
	
	#generate the image
	img = image

	# Initiate ORB detector
	orb = cv2.ORB_create()

	# find the keypoints and feature descriptors with ORB
	kp = orb.detect(img,None)
	kp, des = orb.compute(img, kp)

	return (kp,des)


def sift_return_features(image):
	'''
	Takes some image object 
	and returns the list of sift feature descriptors 

	inputs:
		image: some image object..

	'''
	
	#generate the image
	img = image

	# Initiate SIFT detector
	sift = cv2.SIFT_create()
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# find the keypoints and features and return them....
	kp = sift.detect(gray, None)
	des = sift.compute(gray, kp)[1]

	return (kp,des)





if __name__ == "__main__":
	#Testing to see if the algorithms work....
	img1 = cv2.imread('training_fake/easy_100_1111.jpg') 
	print(
		str(sift_return_features(img1)[0]) + '\n'*10 + str(sift_return_features(img1)[1])
		)
	print(len(sift_return_features(img1)[1]))
