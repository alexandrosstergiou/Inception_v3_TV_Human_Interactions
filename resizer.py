from PIL import Image
import os, sys

path = "tv_human_interactions_videos/frames"
i = 0
for subpath in os.listdir(path):
    for item in os.listdir(path+"/"+subpath):
        im = Image.open(path+"/"+subpath+"/"+item)
        imResize = im.resize((300,300), Image.ANTIALIAS)
        components = item.split("_")
        imResize.save(path+"/"+subpath+"/"+str(i)+"_"+components[1]+"_"+components[2] , 'PNG')
        os.remove(path+"/"+subpath+"/"+item)
    i = i+1
