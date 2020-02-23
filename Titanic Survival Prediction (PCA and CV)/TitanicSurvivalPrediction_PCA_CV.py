# Titanic Survival analysis using Python (Including PCA and Cross Validation)

import pandas as pd
full=pd.read_csv("C:/Users/***/DataSets/titanic.csv")
X=full.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12,13]]
y=full.iloc[:,1]
X.head()

# Imputation¶
full.isna().sum()

full.age.fillna(full.age.mean(),inplace=True)
full.fare.fillna(full.fare.mean(),inplace=True)

full["HasCabin"]="F"
for i in range(len(full.cabin)):
    if pd.isna(full.loc[i,"cabin"]):
        full.loc[i,"HasCabin"]=0
    else:
        full.loc[i,"HasCabin"]=1
		
full=full.drop(columns="cabin")
full["embarked"].fillna(full.embarked.mode()[0],inplace=True)
full=full.drop(columns=["boat","body","home_dest"])
full=full.drop(columns=["ticket"])

# Binning
mybins=range(0,full.age.astype("int").max()+1,10)
full["Age_Bucket"]=pd.cut(full.age,bins=mybins)
full.drop(columns="age",inplace=True)

# Convert strings to numbers¶
full=pd.get_dummies(data=full,columns=["pclass"],drop_first=True)
full=pd.get_dummies(data=full,columns=["HasCabin"],drop_first=True)
for i in range(0,len(full)):
    if (full.sibsp[i]>2):
        full.sibsp[i]="3+"

for i in range(len(full)):
    if full.parch[i]>2:
        full.parch[i]="3+"
		
print(full.parch.value_counts())
print(full.sibsp.value_counts())

full=pd.get_dummies(data=full,columns=["sibsp","parch"],drop_first=True)
full=pd.get_dummies(data=full,columns=["sex","embarked"],drop_first=True)
full.Age_Bucket=pd.factorize(full.Age_Bucket)[0]
full.drop(columns="name",inplace=True)
full.head()

# Outlier Management¶
import numpy as np
np.sqrt(full.fare).hist()
full.fare=np.sqrt(full.fare)
from matplotlib import pyplot as pt
pt.boxplot(full.fare,showfliers=True)

full.loc[full.fare>9,"fare"]=9
from matplotlib import pyplot as pt
pt.boxplot(full.fare,showfliers=True)

#Scaling¶
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
full_scaled=min.fit_transform(full)
full_scaled=pd.DataFrame(full_scaled, columns=full.columns, index=full.index)
full_scaled

# PCA¶
from sklearn.decomposition import PCA
X=full_scaled.iloc[:,1:len(full_scaled.columns)]
y=full_scaled.iloc[:,0]
pca=PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(X_reduced)

# Cross Validation & Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
log=LogisticRegression(penalty="l2",C=1)
kfold=KFold(n_splits=10,random_state=123)
f1Scores=cross_val_score(X=X,y=y,estimator=log,cv=kfold,scoring="accuracy",n_jobs=-1)
f1Scores








