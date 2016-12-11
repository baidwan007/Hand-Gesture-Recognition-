import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import time,math
import pickle
from scipy.cluster.vq import whiten,vq, kmeans
import dictForHisto_f


def codebook_creator():
  print (dictForHisto_f)
  dictForHisto_f.RAMdictCreator()
  #MIN_MATCH_COUNT = 10
  #sift = cv2.xfeatures2d.SIFT_create()
  print (cv2)
  list=np.zeros((1,128))
  count=0
  for key,value in sorted(dictForHisto_f.desDic.items()):
    #print(i)
    #img= cv2.imread('./proto/A-train'+str(i+1)+'.jpg',0)
    #gray = imutils.resize(img, width = 400)
    #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #kp,des = sift.detectAndCompute(gray,None)
    #img=cv2.drawKeypoints(gray,kp,0)#flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img1=cv2.drawKeypoints(gray,kp,0,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('skinSift_B/sift_b'+str(i+1)+'.jpg',img)
    #cv2.imwrite('skinSift_B/sift_be'+str(i+1)+'.jpg',img1)
    #print ("for image kp")
    #print (kp)
    #f=open("testFeature.pkl","rb")
    #f.write(str(des))
    #print(type(des))
    #print (len(kp))
    #print (len(des))
    #for j in value:
      #print (j)
    #print ("yup yahi ka hu",type(value))
    #print (value)
    #print (key)
    try:
        #a=pickle.load(f)
        #b=np.array(a)
      #print ("for key ",key," count= ",count)
      if(count==0):
        #print ("initialising list")
        list=value
        #print ("length of list now is",len(list))
      else:
        #print ("concatenating bc")
        #a=pickle.load(f)e
        #print (value)
        list=np.concatenate((list,value),axis=0)
        #print ("length of list now is",len(list))
    except EOFError:
      pass
      #f.write(str(kp)+"\ n")
      #f.write("\n \n \n"+ str(des[i])+"\n") 
    count=count+1
    #return list
  
  #f.close()
  print (list[24])
  f=open("list.txt","w")
  for i in list:
    f.write(str(i))
  f.close()

  whitened = whiten(list)
  codes=int(math.sqrt(len(list)))
  codebook,distortion=kmeans(whitened,codes)

  print (type(codebook))
  #print (whitened)
  #print (codebook)
  f=open("codebook.txt","w")
  print (len(codebook))

  for i in range(len(codebook)):
    # print ("dumbassdivya "+str(i))
    f.write(str(codebook[i]))
  f.close()
  
  f=open("cbook.pkl","wb")
  pickle.dump(codebook,f)
  f.close()
  print (codes)
  print (len(list))
codebook_creator()  