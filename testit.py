import cv2
import os.path
import imutils
import numpy as np
from numpy import histogram, array
from matplotlib import pyplot as plt
import time
import pickle
from scipy.cluster.vq import whiten,vq, kmeans
import codebook_loader_f
from sklearn import svm

codebook=codebook_loader_f.constant_codebook()
s=open("svm.pkl","rb")
clf2 = pickle.load(s)
s.close()

"""img = cv2.imread('./testit2.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (20,1,80,90)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imwrite('extracted3.jpg',img)"""



print ("dictForHisto has been imported")

sift = cv2.xfeatures2d.SIFT_create()
for i in range(101):
  try:
    img = cv2.imread('./../minor_dataset/Training/A/A-train'+str(i+1)+'.jpg')
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,1,80,90)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    #cv2.imwrite('extracted3.jpg',img)
    #img= cv2.imread('./../imwritereground_extracted_images//B'+str(i+1)+'.jpg',0)
    cv2.imwrite('./../test_write1'+str(i)+'.jpg',img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./../test_write'+str(i)+'.jpg',img)
    gray = imutils.resize(img, width = 400)
    kp,des = sift.detectAndCompute(gray,None)
    #img=cv2.drawKeypoints(gray,kp,0)
    whitened=whiten(des)
    #print ("mai bhi chal ra hu")
    code, dist = vq(whitened, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              density=True)
    result=clf2.predict(histogram_of_words)
    if(result==1):
      print (result,str(i),"The symbol input means: Hi")
    if(result==2):
      print (result,str(i),"The symbol input means: How are you?")
  except AttributeError:
    pass

  


