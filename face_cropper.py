import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob 

dataset = "C:\Documents (Austin)\P-ai\Kaggle\SET_00_OUTPUT_TRAIN_SAMPLE_VIDEOS"

def clear_folder():
  mydir = "C:\Documents (Austin)\P-ai\Kaggle\cropped-faces"
  filelist = [ f for f in os.listdir(mydir) if f.endswith(".jpg") ]
  for f in filelist:
    os.remove(os.path.join(mydir, f))



def detect_faces(image):
    face_detector = dlib.get_frontal_face_detector()

    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    return face_frames

df = pd.DataFrame(columns=['filename', 'faces'])
all_filename = []
all_faces = []

individual_imagesets = os.listdir(dataset)

#print(individual_imagesets)

for filenames in individual_imagesets:
    
    for img in os.listdir(dataset + "\\" + filenames):
       
        if img.endswith(".jpg"):
            img_path = dataset + "\\" + filenames + "\\" + img
            image = io.imread(img_path)
            detected_faces = detect_faces(image)
            print(detected_faces)
            
            if len(detected_faces) == 1:
                face = Image.fromarray(image).crop(detected_faces[0])
                # plt.subplot(1, )
                plt.axis('off')
                new_path = dataset + "\\" + filenames + "\\" + img
                plt.imshow(face)
                plt.savefig(new_path, dpi=300, bbox_inches='tight')

            else: 
                os.remove(img_path)

            # all_filename.append(img_path)
            # all_faces.append(len(detected_faces))

            # for n, face_rect in enumerate(detected_faces):
            #     face = Image.fromarray(image).crop(face_rect)
            #     plt.subplot(1, len(detected_faces), n+1)
            #     plt.axis('off')
            #     new_path = "C:\Documents (Austin)\P-ai\Kaggle\cropped" + str(counter) + "-" + filename 
            #     counter += 1;  
            #     plt.imshow(face)
            #     plt.savefig(new_path, dpi=300, bbox_inches='tight')
 
# df['filename'] = all_filename
# df['faces'] = all_faces

# df.to_csv('all-data.csv')