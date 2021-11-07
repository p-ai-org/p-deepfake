import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os

# SIFT and SURF implementation
def analyze_regions(regions, algo):
  keypoints_by_region = []
  for region in regions:
    if (algo == "SIFT"):
      gray = cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
      sift = cv2.xfeatures2d.SIFT_create()
      kp = sift.detect(gray,None)
      keypoints_by_region.append(kp) 
    else:
      surf = cv2.xfeatures2d.SURF_create(400)
      kp = surf.detect(region,None)
      keypoints_by_region.append(kp) 
    
  return keypoints_by_region