# Fertility Prediction using Random Forest Classification in Python

import pandas as pd
full=pd.read_csv("C:/Users/E002891/Desktop/DayWiseTracker/Programming Concepts/Data Science/DataSets/fertility.csv",sep=";")
full.head(5)

full.describe()

full.isna().any()
mybins=range((full.Age*100).astype("int").min()-10,(full.Age*100).astype("int").max()+10,10)
full.Age=pd.cut((full.Age*100).astype("int"),bins=mybins)
full.head(3)

bins1=range((full["Number of hours spent sitting"]*10).astype("int").min(),(full["Number of hours spent sitting"]*10).astype("int").max(),3)
full["Number of hours spent sitting"]=pd.cut((full["Number of hours spent sitting"]*10).astype("int"),bins1)
# full["Number of hours spent sitting"].value_counts()

full.Output=full.Output.factorize()[0]
full.head(3)

full.Output.value_counts()

# Fitting models without treating categorical variables¶
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
log=LogisticRegression(penalty="l2",C=1)
log.fit(full.iloc[:,0:9],full["Output"])
classification_report(full["Output"],log.predict(full.iloc[:,0:9]))
#Error: float() argument must be a string or a number, not 'pandas._libs.interval.Interval'

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
log=RandomForestClassifier()
log.fit(full.iloc[:,0:9],full["Output"])
classification_report(full["Output"],log.predict(full.iloc[:,0:9]))
#Error: float() argument must be a string or a number, not 'pandas._libs.interval.Interval'

full1=full

# Binning and Converting to dummies¶
for i in range(len(full)):
    if((str(full.iloc[i,1])=='(80, 90]') | (str(full.iloc[i,1])=='(90, 100]')):
        full.loc[i,"NewAge"]='(80,100]'
    else:
        full.loc[i,"NewAge"]=full1.loc[i,"Age"]

for i in range(len(full)):
    if((str(full.loc[i,"NewAge"])=='(40, 50]') | (str(full.loc[i,"NewAge"])=='(50, 60]')):
        full.loc[i,"NewAge1"]='(40-60]'
    else:
        full.loc[i,"NewAge1"]=full.loc[i,"NewAge"]
		
for i in range(len(full)):
    if(str(full.loc[i,'NewAge1'])=='(40-60]'):
        full.loc[i,'Age']='(40, 60]'
    else:
        full.loc[i,'Age']=full.loc[i,'NewAge1']
		
full.drop(columns=["Age","NewAge"],inplace=True)

full["Age"]=full["NewAge1"]
full.drop(columns=["NewAge1"],inplace=True)
full.head(3)

full=full1

full=pd.get_dummies(data=full,columns=["Season","Childish diseases","Accident or serious trauma","Surgical intervention","High fevers","Frequency of alcohol","Smoking habit","Number of hours spent sitting","Age"],drop_first=True)

# Converting Strings to numbers¶
import numpy as np
full2=full1
full2["Frequency of alcohol"]=np.where(((full2["Frequency of alcohol"]==0.2) | (full2["Frequency of alcohol"]==0.4)),1.0,full2["Frequency of alcohol"])
full2["Frequency of alcohol"].value_counts()

# PCA
from sklearn.decomposition import PCA
X=full.iloc[:,1:len(full.columns)]
y=full["Output"]
pca=PCA(n_components=2)
X_new=pca.fit_transform(X)
y.value_counts()


# Cross Validation and model building
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
log=LogisticRegression(penalty="l2",C=1, class_weight="balanced")
kfold=RepeatedKFold(n_splits=20,n_repeats=5)
f1_scores=cross_val_score(estimator=log,X=X,y=y,n_jobs=-1,cv=kfold,verbose=True,scoring="recall")
f1_scores

kfold=KFold(n_splits=3,random_state=123)
# f1Scores=cross_val_score(X=X,y=y,estimator=log,cv=kfold,scoring="f1",n_jobs=-1)
preds=cross_val_predict(X=X,y=y,estimator=log,cv=kfold,n_jobs=-1,method="predict_proba")

# pos_pred=preds[]
pos_preds=[i[1] for i in preds]
pos_preds
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y,pos_preds))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
log=RandomForestClassifier(n_estimators=1000)
kfold=KFold(n_splits=3,random_state=123)
# f1Scores=cross_val_score(X=X,y=y,estimator=log,cv=kfold,scoring="f1",n_jobs=-1)
preds=cross_val_predict(X=X,y=y,estimator=log,cv=kfold,n_jobs=-1,method="predict_proba")

# Treating class imbalance problem: Upsampling
from sklearn.utils import resample
full_majority=full.loc[full.Output==0,:]
full_minority=full.loc[full.Output==1,:]
full_minority_new = resample(full_minority,n_samples=88,replace=True,random_state=123)
full_minority_new.Output.value_count()

full_new=pd.concat([full_majority,full_minority_new])
len(full_new)

X=full_new.iloc[:,1:len(full_new)]
y=full_new.iloc[:,0]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_new=pca.fit_transform(X)
print(X_new[1:5])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

clf=RandomForestClassifier()
kfold = KFold(n_splits=3)
preds = cross_val_predict(X=X_new,y=y,cv=kfold,n_jobs=-1,estimator=clf)
preds

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y,preds))
print(classification_report(y,preds))

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
rclf=RandomForestClassifier(n_estimators=1000)
rKfold=RepeatedKFold(n_splits=5,n_repeats=5)
f1Scores = cross_val_score(X=X_new,y=y,cv=rKfold,n_jobs=-1,estimator=rclf,scoring="f1")
print(f1Scores)
print("avarage f1:",f1Scores.mean())







