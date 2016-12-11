from numpy import array, histogram 
from scipy.cluster.vq import vq 
import codebook_loader_f
import dictForHisto_f
import numpy as np
import pickle

codebook=codebook_loader_f.constant_codebook()
#print (dictForHisto.desDic)
print ("mai chal ra hu")
def computeHistograms(codebook, descriptors):
  #print ("mai bhi chal ra hu")
  code, dist = vq(descriptors, codebook)
  histogram_of_words, bin_edges = histogram(code,
                                            bins=range(codebook.shape[0] + 1),
                                            density=True)
  return histogram_of_words


#print (codebook)
#print (type(codebook))
print ("hi")
print (dictForHisto_f.desDic)
list=np.zeros((1,94))
count=0
#a=[1,2,3,4]
#b=np.array(a)
print("**********************************************************")
for key,value in sorted(dictForHisto_f.desDic.items()):
  #print ("hiiii")
  #print (type(value))
  print (key)
  histogram_word_pic =computeHistograms(codebook,value)
  print (histogram_word_pic)
  if (count==0):
    #print ("in if")
    list=histogram_word_pic
  else:
    #print ("in else:  ")
    list=np.vstack((list,histogram_word_pic))
  #print (list)
  print (type(histogram_word_pic))
  count=count+1
  #print (histogram_word_pic)
  #print (histogram_word_pic.sum())
print ("hi",list)

f=open("histogram.pkl","wb")
pickle.dump(list,f)
f.close()