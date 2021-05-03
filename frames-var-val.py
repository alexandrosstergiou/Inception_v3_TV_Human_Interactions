import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,glob

pos_list = [1, 2, 3, 4, 6, 7, 8, 9, 10]
os.chdir("tv_human_interactions_videos")

for pos_val in pos_list:
    frame_dir = 'frames_pos_' + str(pos_val)
    if not os.path.isdir(frame_dir):
        os.mkdir(frame_dir)
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
            class_folder = os.path.join(frame_dir, imagesFolder.split("_")[0])
            frame_name = imagesFolder+"_"+str(i)+'.png'
            if not os.path.isdir(class_folder):
                os.mkdir(class_folder)
            f1 = os.path.join(class_folder, frame_name)
            try:
                s = cv2.imwrite(f1,frame2)
            except:
                print(f'Writing frame {pos} of video {videoFile} failed!')
            if (s == False):
                print(f'Writing frame {pos} of video {videoFile} failed!')
            i = i + 1
            pos = pos + pos_val
            cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
            if (pos >= length):
                break
            prvs = next
        cap.release()
        cv2.destroyAllWindows()



