import csv
from get_feature_descriptors import orb_return_features_2 as orb
from detect_face_parts import extract_facial_regions
import os


def generate_facial_region_data(real_dict, fake_dict, algo=orb):
	'''
	This function takes two dictionaries of the form {facial_region: image of that region}
	and adds generates corresponding csv files for each facial region, facial_region.csv, and 
	then if the file is non-empty it appends a new column with the list [real_keypoints, real_desc,
	fake_keypoints, fake_desc] for each region...

	inputs:
		real_dict, fake_dict (dicts of the form {name of facial region: image obj of that region})
		algo, the feature description algorithm which will be used for this analysis
	output:
		appends that data to the corresponding facial_region.csv files....
	'''

	#gets the names of all the different facial regions..
	facial_regions = real_dict.keys()

	#iterate through each facial region appending to the region file ...
	for region_string in facial_regions:
		with open(f'{region_string}.csv', 'a', newline='') as region_file:
			csvwriter = csv.writer(region_file)
			#If the file is empty  ..... 
			if os.stat(f'{region_string}.csv').st_size == 0:
				#add the column names in the first column...
				csvwriter.writerow(['real kp', 'real desc', 'fake kp', 'fake desc', 'algo'])

			#a tuple with real kp, real desc as the elems
			real_orb = orb(real_dict[region_string])
			#a tuple with fake kp, fake desc as the elems 
			fake_orb = orb(fake_dict[region_string])
			#write to the row.
			csvwriter.writerow([real_orb[0],real_orb[1],fake_orb[0],fake_orb[1], str(algo)])

			#After we append close the file and then go through the next region....
			region_file.close()


if __name__ == "__main__":
	#if the file is ran directly lets get to testing :)
	real_file = 'training_real/real_00001.jpg'
	fake_file = 'training_fake/easy_100_1111.jpg'
	real_dict = extract_facial_regions(real_file)
	fake_dict = extract_facial_regions(fake_file)
	generate_facial_region_data(real_dict,fake_dict)


