import os 
from orb_feature_descriptors import orb_return_features
import csv

'''
	GOAL: Take each file in training_fake and create a csv file
	list with the [file name, descriptors]
'''

def generate_orb_feature_data(foldername):
	'''
	Takes a folder of images in the relative directory and
	generates a csv file in the form filename: keypoints
	'''

	#open the file as foldername_data.csv
	with open(foldername + '_data.csv', 'w', newline='') as csv_file:

		#create a csvwriter object
		csvwriter = csv.writer(csv_file)

		#Format this into a table that is needed
		csvwriter.writerow(['file_name','descriptors'])

		#Iterate through each file in training_fake
		for path in os.listdir(foldername):
			csvwriter.writerow(
				[path, str(orb_return_features(foldername + '/' + path)[1])]
				)

		csv_file.close()


generate_orb_feature_data('training_fake')
generate_orb_feature_data('training_real')