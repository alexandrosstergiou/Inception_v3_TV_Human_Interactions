import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,glob


i = 0

os.chdir("tv_human_interactions_videos")
for file in glob.glob("*.avi"):
    videoFile = os.path.basename(file)
    imagesFolder = videoFile.split(".")[0]
    f1 = 'frames/'+imagesFolder.split("_")[0]+"/"+imagesFolder+"_"+str(i)+'.png'
    if (i < 1):
        print(f1)
    else:
        break
    i += 1