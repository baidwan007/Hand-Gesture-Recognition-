import cv2
import os.path
import imutils
import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from scipy.cluster.vq import whiten,vq, kmeans

print ("dictForHisto has been imported")
def dictCreator():
  MIN_MATCH_COUNT = 10
  sift = cv2.xfeatures2d.SIFT_create()
  list=np.zeros((1,128))
  f=open("desDict.pkl","wb")
  for i in range(98):
    print("A",i)
    try:
      os.chdir
      img= cv2.imread('./../resized_foreground_extracted_images/A/A'+str(i+1)+'.jpg',0)
      gray = imutils.resize(img, width = 400)
      kp,des = sift.detectAndCompute(gray,None)
      img=cv2.drawKeypoints(gray,kp,0)
      key="a"+str(i+1)
      desdic={key:des} 
      """check if the dict in python requires initialisation in python"""
      pickle.dump(desdic,f)
    except AttributeError:
      print ("a",i,"fuck")
  for i in range(102):
    print("B",i)
    try:
      img= cv2.imread('./../resized_foreground_extracted_images/B/B'+str(i+1)+'.jpg',0)
      gray = imutils.resize(img, width = 400)
      kp,des = sift.detectAndCompute(gray,None)
      img=cv2.drawKeypoints(gray,kp,0)
      key="b"+str(i+1)
      desdic={key:des} 
      """check if the dict in python requires initialisation in python"""
      pickle.dump(desdic,f)
    except AttributeError:
      print ("b",i,"fuck")
  f.close() 
  
dictCreator()