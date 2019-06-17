import cv2
import numpy as np
import glob
import os
os.environ['CUDA_VISABLE_DEVICES']='1'
from tqdm import tnrange,tqdm

images = glob.glob('*.jpg')

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
for image in tqdm(images,desc="calcing"):
    img = cv2.imread(image)
    edge = cv2.filter2D(img,-1,kernel)
    #cv2.imwrite('image_edge.jpg',edges)
    #img = cv2.imread('image_edge.jpg')
    

    gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)

    dst1 = 255 - gray
    cv2.imwrite(image+'dst1.jpg',dst1)
    
