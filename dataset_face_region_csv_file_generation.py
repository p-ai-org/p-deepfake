from data_generation import *

from detect_face_parts import *

import json

import os 

from os import path

import cv2


'''
This is the script that we will use to generate the 
csv files needed........
'''

#Below we define the file paths used...
cwd = os.getcwd()
folder = os.path.dirname(cwd)
json_file_path = path.join(cwd, 'link00JSON.json')  
dataset_file_dir = path.join(folder, 'data\\cropped_data')
#load the json object into a dictionary which we will use.
decoder = json.JSONDecoder() #init the decoder
json_file = open(json_file_path, 'r') #open the file in read only
json_file.readline()
json_file_header = json_file.readline()
#Now here we load the linker into here..
meta_data_linker = decoder.decode(json_file.readline())

#in this block of code we clean the dict and test with a print
meta_data_linker = {path.splitext(x)[0]:path.splitext(y)[0] for (x,y) in meta_data_linker.items()}



#Here we will then go through the json_file_path items...
for fake, original in meta_data_linker.items():
  #open up the folders to there corresponding file paths....
  fake_folder = path.join(dataset_file_dir, fake)
  real_folder = path.join(dataset_file_dir, original)
  #Assume that fake and real have the same number of files...
  for real_index in range(min(len(os.listdir(real_folder)), len(os.listdir(fake_folder)))):
    if str(os.listdir(real_folder)[real_index]) == str(os.listdir(fake_folder)[real_index]):
      fake_file = path.join(fake_folder, str(os.listdir(fake_folder)[real_index]))
      real_file = path.join(real_folder, str(os.listdir(real_folder)[real_index]))
      fake_dict = extract_facial_regions(fake_file)
      real_dict = extract_facial_regions(real_file)
      generate_facial_region_data(real_dict, fake_dict)





