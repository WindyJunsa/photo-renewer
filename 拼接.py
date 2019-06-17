import cv2
import numpy as np
import glob
import os
os.environ['CUDA_VISABLE_DEVICES'] = '1'

image_all = glob.glob('*.jpg')
image_gray = glob.glob('*.jpggray.jpg')

image_ori = list(set(image_all)^set(image_gray))

for image in image_ori:
    out=np.concatenate((cv2.imread(image),cv2.imread(image+"gray.jpg")),1)
    cv2.imwrite('gray_train/'+image+"out.jpg",out)
