import json
import os


metadata_file = "C:\Documents (Austin)\P-ai\Kaggle/Set 00/metadata.json"

link_file = "C:\Documents (Austin)\P-ai\Kaggle/link00DICT.txt"


with open(metadata_file) as json_file:
    metadata = json.load(json_file)


f = open(link_file, "w")
f.write("Link file" + "\n" + "Fake : Original" + "\n")

link_dict = dict()

for video in metadata:
    if metadata[video]["label"] == "FAKE":
        print(video)
        print(metadata[video]["original"])

        link_dict[video] = metadata[video]["original"]

print(link_dict)
       

json.dump(link_dict, f)



f.close()
    
