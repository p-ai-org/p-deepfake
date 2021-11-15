import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import matplotlib.pyplot as plt

import csv
import os

# input:
# eyes: image
# Write to .csv file


def analyze_images(real, fake):

    for key in real:
        f = open(f + ".csv", "a")
        writer = csv.writer(f)
        img_real = real[key]
        img_fake = fake[key]

        data = []
        if (algo == "SIFT"):
            gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp_real, des_real = sift.detectAndCompute(gray, None)

            gray = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp_fake, des_fake = sift.detectAndCompute(gray, None)
        # ORB
        else if (algo == "ORB"):
            orb = cv2.ORB_create()
            kp_real, des_real = orb.detectAndCompute(img_real, None)

            orb = cv2.ORB_create()
            kp_fake, des_fake = orb.detectAndCompute(img_fake, None)
        # FAST
        else if (algo == "FAST"):
            fast = cv2.FastFeatureDetector_create()
            kp_real, des_real = fast.detectAndCompute(img_real, None)

            fast = cv2.FastFeatureDetector_create()
            kp_fake, des_fake = fast.detectAndCompute(img_fake, None)
        # SURF
        else if (algo == "SURF"):
            surf = cv2.xfeatures2d.SURF_create(400)
            kp_real, des_real = surf.detectAndCompute(img_real, None)

            surf = cv2.xfeatures2d.SURF_create(400)
            kp_fake, des_fake = surf.detectAndCompute(fake, None)
        else:
            print("Invalid algo")
            raise

        data = [kp_real, des_real, kp_fake, des_fake]
        writer.writerow(data)


# peforms given algo on region
def analyze_regions(regions, algo):
    keypoints_by_region = []
    for region in regions:
        # SIFT
        if (algo == "SIFT"):
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray, None)
            keypoints_by_region.append(kp)
        # ORB
        else if (algo == "ORB"):
            orb = cv2.ORB_create()
            kp = orb.detect(region, None)
            keypoints_by_region.append(kp)
        # FAST
        else if (algo == "FAST"):
            fast = cv2.FastFeatureDetector_create()
            kp = fast.detect(region, None)
            keypoints_by_region.append(kp)
        # SURF
        else if (algo == "SURF"):
            surf = cv2.xfeatures2d.SURF_create(400)
            kp = surf.detect(region, None)
            keypoints_by_region.append(kp)
        else:
            print("Invalid algo")
            raise

    return keypoints_by_region
