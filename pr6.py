import pandas as pd
import numpy as np
import seaborn as sea

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df=load_iris()
X=df.data
y=df.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42
)

model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average="macro")
recall=recall_score(y_test,y_pred,average="macro")
error=1-accuracy
print("\nAccuracy:", accuracy)
print("Error Rate:", error)
print("Precision:", precision)
print("Recall:", recall)


for i in range(len(cm)):
    tp=cm[i][i]
    fp=sum(cm[:,i])-tp
    fn=sum(cm[i,:])-tp
    tn=sum(sum(cm))-(tp+fp+fn)
    
    print(f"Class {i}:")
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print()