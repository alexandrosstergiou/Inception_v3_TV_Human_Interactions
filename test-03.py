import cv2
import os
import glob
import numpy as np 
count = 0
os.chdir("tv_human_interactions_videos")
if not os.path.isdir('test_dir'):
    os.mkdir('test_dir')
#for file in glob.glob("*.avi"):
    #if count > 5:
        #break
#videoFile = os.path.basename(file)
videoFile = 'handShake_0001.avi'
imagesFolder = videoFile.split(".")[0]

print(imagesFolder)

cap = cv2.VideoCapture(videoFile)
pos = 0
cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
i = 0
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#count += 1
while(cap.isOpened()):
    ret, frame2 = cap.read()
    #print(frame2, ret)
    class_folder = 'test_dir/'+imagesFolder.split("_")[0]
    frame_name = imagesFolder+"_"+str(i)+'.png'
    if not os.path.isdir(class_folder):
        os.mkdir(class_folder)
    f1 = os.path.join(class_folder, frame_name)
    print('writing dir:',f1)
    s = cv2.imwrite(f1,frame2)
    print('write result: ', s)

    i = i + 1
    pos = pos + 5
    cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
    #print("Created image: "+f1, )
    if (pos >= length):
        break
    prvs = next