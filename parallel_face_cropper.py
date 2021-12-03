import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob 
import multiprocessing as mp

dataset = "C:\Documents (Austin)\P-ai\Kaggle\Danica batch\\batch_austin"

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




def crop_faces(all_data):

    df = pd.DataFrame(columns=['filename', 'faces'])



    for filenames in all_data:
        
        for img in os.listdir(dataset + "\\" + filenames):
            print(img)
        
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


if __name__ == "__main__":

    individual_imagesets = os.listdir(dataset) #Global

    pool = mp.Pool(mp.cpu_count())

    pool.map(crop_faces(individual_imagesets))

    pool.close()
