import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,glob


os.chdir("tv_human_interactions_videos")
for file in glob.glob("*.avi"):
    videoFile = os.path.basename(file)

    imagesFolder = videoFile.split(".")[0]

    cap = cv2.VideoCapture(videoFile)
    pos = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    i = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        ret, frame2 = cap.read()

        f1 = 'frames/'+imagesFolder.split("_")[0]+"/"+imagesFolder+"_"+str(i)+'.png'

        cv2.imwrite(f1,frame2)

        i = i + 1
        pos = pos + 5
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
        print("Created image: "+f1)
        if (pos >= length):
            break
        prvs = next
    cap.release()
    cv2.destroyAllWindows()
