import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

# style your matplotlib
mpl.style.use("seaborn-darkgrid")
# run this block

from tqdm import tqdm

files=os.listdir("dataset")
print(files)

image_array=[]  # it's a list later i will convert it to array
label_array=[]

path="dataset/"
# loop through each sub-folder in train
for i in range(len(files)):
    # files in sub-folder
    file_sub=os.listdir(path+files[i])

    for k in tqdm(range(len(file_sub))):
        try:
            img=cv2.imread(path+files[i]+"/"+file_sub[k])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(64,64))
            image_array.append(img)
            label_array.append(i)
        except:
            pass