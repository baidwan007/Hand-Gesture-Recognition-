import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from scipy.cluster.vq import whiten,vq, kmeans

def constant_codebook():
  f=open("cbook.pkl","rb")
  codebook=pickle.load(f)
  #print (codebook)
  return codebook

#constant_codebook()