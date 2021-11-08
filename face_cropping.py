import dlib
from PIL import Image 
from skimage import io 
import matplotlib.pyplot as plt 


def detect_images(image):
	'''
	This function takes in some image file path and then returns a list of coordinates
	for the all faces in the image......

	inputs:
		image: the filepath to the image wanting the image to be extracted from.....
	outputs: 
		detect_faces(image): a list of tuple coordinates giving the faces....
	'''

	#initialilze the face detecting algorithim of dlib
	face_detector = dlib.get_frontal_face_detector()
	#get the faces in the face
	detected_faces = face_detector(image, 1)
	#format the data to be the a list of coordinates corresponding to the frames
	face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]

	#return this result
	return face_frames 


def get_face_images(image):
	'''
	This function takes in some image file path and then returns a tuple of images of all
	the faces in the image ..... 

	inputs: 
		image: image for which we are trying to extract faces from
	outputs:
		get_face_images(image): the tuple of image objects
	'''

	#initialize the list of faces
	face_list = []

	#get the image frames in the picture
	image_path = image
	image = io.imread(image_path)
	detected_faces = detect_images(image)
	

	#crop faces and plot 
	for n, face_rect in enumerate(detected_faces):
		face = Image.fromarray(image).crop(face_rect)
		face_list.append(face)
		plt.subplot(1,len(detected_faces), n+1)
		plt.axis('off')
		plt.imshow(face)
		plt.show()

	#return the list of images
	return face_list 


if __name__ == "__main__":
	#load image 
	image_path = 'training_fake/easy_100_1111.jpg'
	print(get_face_images(image_path))

