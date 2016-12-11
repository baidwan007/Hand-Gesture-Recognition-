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


img = cv2.imread('./testit2.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (20,1,80,90)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imwrite('extracted3.jpg',img)



print ("dictForHisto has been imported")

sift = cv2.xfeatures2d.SIFT_create()
img= cv2.imread('./extracted.jpg',0)
gray = imutils.resize(img, width = 400)
kp,des = sift.detectAndCompute(gray,None)
img=cv2.drawKeypoints(gray,kp,0)
  
  
whitened=whiten(des)
 
codebook=codebook_loader_f.constant_codebook()

#print ("mai bhi chal ra hu")
code, dist = vq(whitened, codebook)
histogram_of_words, bin_edges = histogram(code,
                                            bins=range(codebook.shape[0] + 1),
                                            density=True)


s=open("svm.pkl","rb")
clf2 = pickle.load(s)
s.close()
result=clf2.predict(histogram_of_words)
if(result==1):
  print (result,"The symbol input means: Hi")
if(result==2):
  print (result,"The symbol input means: Hi")