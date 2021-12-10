from data_generation import *

from detect_face_parts import *

import json

import os 

from os import path

'''
This is the script that we will use to generate the 
csv files needed........
'''

#Below we define the file paths used...
cwd = os.getcwd()
json_file_path = path.join(cwd, 'link00JSON.json')  
dataset_file_dir = path.join(cwd, 'SET_00_OUTPUT_TRAIN_SAMPLE_VIDEOS')

#load the json object into a dictionary which we will use.
decoder = json.JSONDecoder() #init the decoder
json_file = open(json_file_path, 'r') #open the file in read only
json_file.readline()
json_file_header = json_file.readline()
#Now here we load the linker into here..
meta_data_linker = decoder.decode(json_file.readline())

#in this block of code we clean the dict and test with a print
meta_data_linker = {path.splitext(x)[0]:path.splitext(y)[0] for (x,y) in meta_data_linker.items()}
print(meta_data_linker)

#Here we will then go through the json_file_path items...
for fake, original in meta_data_linker.items():
  #open up the folders to there corresponding file paths....
  fake = path.join(dataset_file_dir, fake)
  real = path.join(dataset_file_dir, original)
  #Assume that fake and real have the same number of files...
  if len(fake) == len(real):
    for file_number in len(fake):
      fake_file = path.join(fake, os.listdir(fake)[file_number])
      real_file = path.join(fake, os.listdir(real)[file_number])
      fake_dict = extract_face_regions(fake_file)
      real_dict = extract_face_regions(real_file)
      generate_facial_region_data(real_dict, fake_dict)
  else:
    print('Folder Mismatch')




