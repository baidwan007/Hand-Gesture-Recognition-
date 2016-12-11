import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from scipy.cluster.vq import whiten,vq, kmeans

desDic={}
print ("dictForHisto has been imported")
def RAMdictCreator():
  f=open("desDict.pkl","rb")
  #desDic=[]
  for i in range(200):
    print(i)
    try:
      a=pickle.load(f)
      print('yes')
      print(a)
      desDic.update(a)
      #desDic.append(a)
    except EOFError:
      print ("no")
      pass
  f.close()

  print (desDic)
  print ("bitch")
  print (desDic['a1'])

RAMdictCreator()
