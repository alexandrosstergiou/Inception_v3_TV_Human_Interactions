from PIL import Image
import os, sys

path = "tv_human_interactions_videos/frames"
i = 0

for subpath in os.listdir(path):
    if i >= 1:
        break
    for item in os.listdir(path+"/"+subpath):
        components = item.split("_")
        if (i < 1):
            print('components: ', components)
            print('save path: ',path+"/"+subpath+"/"+str(i)+"_"+components[1]+"_"+components[2])
            print('remove path: ',path+"/"+subpath+"/"+item)
        else:
            break
        i += 1