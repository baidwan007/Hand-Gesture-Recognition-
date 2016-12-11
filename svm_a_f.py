from sklearn import svm
import numpy as np
import histoBuild_f
import pickle

#lin_clf
y=np.array([])
for i in range(130):
  if (i<69):
    y=np.append(y,[1])
  else:
    y=np.append(y,[2])
print (y)

lin_clf = svm.LinearSVC()
lin_clf.fit(histoBuild_f.list, y)
#dec = lin_clf.decision_function(x)
#print (dec)
#print(dec.shape[1])
print(lin_clf.predict(histoBuild_f.list))
s=open("svm.pkl",'wb')
pickle.dump(lin_clf,s)
s.close()
  