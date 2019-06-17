import cv2
import numpy as np
import glob
import os
os.environ['CUDA_VISABLE_DEVICES']='1'
from tqdm import tnrange,tqdm

images = glob.glob('train/*.jpg')

for image in tqdm(images,desc="calcing"):
    img = cv2.imread(image)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(image+'gray.jpg',gray)
    
