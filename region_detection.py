import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os

# peforms given algo on region 
def analyze_regions(regions, algo):
  keypoints_by_region = []
  for region in regions:
    # SIFT
    if (algo == "SIFT"):
      gray = cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
      sift = cv2.xfeatures2d.SIFT_create()
      kp = sift.detect(gray,None)
      keypoints_by_region.append(kp) 
    # ORB
    else if (algo == "ORB"):
      orb = cv2.ORB_create()
      kp = orb.detect(region,None)
      keypoints_by_region.append(kp) 
    # FAST 
    else if (algo == "FAST"):
      fast = cv2.FastFeatureDetector_create()
	    kp = fast.detect(region, None)
      keypoints_by_region.append(kp)
    # SURF 
    else if (algo == "SURF"):
      surf = cv2.xfeatures2d.SURF_create(400)
      kp = surf.detect(region,None)
      keypoints_by_region.append(kp) 
    else:
      print("Invalid algo")
      raise
    
  return keypoints_by_region