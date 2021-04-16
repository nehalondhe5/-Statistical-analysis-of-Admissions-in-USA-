# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 05:38:31 2020

@author: dell
"""


# Import the required library to run the algorithm 

import os
import numpy as np
import pandas as pd
import scipy as scipy
from pandas import DataFrame , Series
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import preprocessing
from statsmodels.graphics.mosaicplot import mosaic 
import matplotlib.pyplot as plt

# The data is stored in D directory ,reading the xlsx file  
# The data is in uncoded format 

os.chdir('D:')
admission=pd.read_excel('naruto.xlsx')
#data preprossesing
print(admission.count().sort_values())
df=admission
print(df.shape)

# To remove the null values if any in the given dataset 
df.dropna(how='any')
print(df)
print(df.dtypes)


gre1=np.ones((400,1),dtype=str)
gre=df["gre"]

#division of gre in categories under 0 to 440 , 440 to 580, then greater than 580 
for i in range(0,len(gre),1) :
    if gre[i]>=0 and gre[i]<440 :
         gre1[i]="Low"
    if gre[i] >=440 and gre[i]<580 : 
         gre1[i]="Medium"
    if gre[i]>=580 :
         gre1[i]="High"    
#print(gre1)  

df=df.drop("gre",axis=1)     
df.insert(1,"gre1",gre1)
print(df)
print(df.shape)



# to check if there are outliers for numeric data 
X=df[["gpa"]]
sns.boxplot(df[["gpa"]])
print("getting boxplot of GPA")

# to remove outlier
f=df.loc[289,"gpa"]
print(f)
print(df["gpa"].min())
df=df.drop(index=290)
#print(df.shape)


# box plot of numeric variable gpa
X=df[["gpa"]]
sns.boxplot(df[["gpa"]])

# To detect the outlier we use z score method 

from scipy import stats
z=np.abs(stats.zscore(df._get_numeric_data()))
print(z)
threshold=3
print(np.where(z>3))
print(df.shape)
print(df)

# To get the association between cateforical variables we use mosaic plot

mosaic(df,["admit","gre1","rank"])
plt.show()

# To code the data with numeric form 

df["Gender"].replace({"Male":1,"Female":0},inplace=True)
df["rank"].replace({"High":1,"Medium":2,"Low":3,"Poor":4},inplace=True)
df["gre1"].replace({"H":1,"M":2,"L":3},inplace=True)
df["ses"].replace({"High":1,"Medium":2,"Low":3},inplace=True)
df["admit"].replace({"Yes":1,"No":0},inplace=True)
df["Race"].replace({"Hispanic":1,"Asian":2,"AAfrican":3},inplace=True)
print(df)
print(df.shape)


# exploratory data analysis

# To get dummy variables k-1 should be there  
categorical_columns=["gre1","ses","Gender","Race","rank"]   
df=pd.get_dummies(df,columns=categorical_columns)
df.head()
print(df)


# for normalizing the data Set 
from sklearn import preprocessing 
scaler=preprocessing.MinMaxScaler()
scaler.fit(df)
df=pd.DataFrame(scaler.transform(df),index=df.index,columns=df.columns)
print(df.iloc[0:20])


# For feature selecttion we use selectKBest method 
from sklearn.feature_selection import SelectKBest, chi2
X=df.loc[:,df.columns!="admit"]
y=df[["admit"]]
selector=SelectKBest(chi2,k=16)
selector.fit(X,y)
X_new=selector.transform(X)
print(selector.pvalues_)
scores= -np.log10(selector.pvalues_)
print(X.columns[selector.get_support(indices=True)])
print(scores)
print(len(X))


# Defining a list of predictors

s=["gpa","gre1_1","gre1_2","gre1_3","ses_1","ses_2","ses_3","Gender_0","Gender_1",
   "Race_1","Race_2","Race_3","rank_1",
   "rank_2","rank_3","rank_4"]
print(len(s))
print(len(scores))

#plotting of scores 
y_pos=np.arange(len(s))
plt.bar(y_pos ,scores)
plt.xticks(y_pos,s,rotation="vertical")
#plt.barh(range(len(s)), color='b', align='center')
plt.xlabel("Predictors")
plt.ylabel("Scores")
plt.show()


import time 
t0=time.time()

#x=df[["gre1_3","rank_1","rank_4","gre1_1","Race_1","Race_2","rank_3","rank_2","gre1_3"]]
# Defining the predictors and responses 

x=df[["gre1_1","gre1_3","rank_1","rank_3","rank_4"]]
y=df[["admit"]]


# Using Logistic Regression
print("Logistic regression algorithm")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Dividing the data into train and test splitting 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf_logreg=LogisticRegression(random_state=0)
x_train.shape


#building the model using training set 
log_model=clf_logreg.fit(x_train,y_train)
y_pred=clf_logreg.predict(x_test)
print(y_test)
print(y_pred)
print(log_model)
score=accuracy_score(y_test,y_pred)
print("score using logistic regression",score)
print("time taken using logistic regression classifier",time.time()-t0)


#Confusion matrix 
from sklearn import metrics 
cm=metrics.confusion_matrix(y_test,y_pred)
print("The confusion matrix for logistic classifier is given as ")
print(cm)


#plotting of ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve ,auc
log_roc=roc_auc_score(y_test,log_model.predict(x_test))
log_fpr , log_tpr ,threshold =roc_curve(y_test ,log_model.predict_proba(x_test)[:,1])
log_auc=auc(log_fpr , log_tpr)
print(log_auc)

plt.figure()
plt.plot(log_fpr,log_tpr)
plt.xlabel("false positive rate")
plt.ylabel("True positive rate")
plt.show()


# using the decision tree classifier
print("decision tree algorithm")


from sklearn.externals.six import StringIO 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from IPython.display import SVG
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import Image , display
import pydotplus

# splitting the data into train and test      

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf_dt=DecisionTreeClassifier(random_state=0)

#building the model using training set 
dec_tree_model=clf_dt.fit(x_train,y_train)
print(dec_tree_model)
y_pred=clf_dt.predict(x_test)
score=accuracy_score(y_test,y_pred)
print("Score using Decision tree classifier ",score)
print("time taken by decision tree classifier ",time.time()-t0)


import sklearn.metrics as metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print("Accuracy using decision tree classifier : ",metrics.accuracy_score(y_test, y_pred))

from sklearn import metrics 
cm=metrics.confusion_matrix(y_test,y_pred)
print("The confusion matrix for decision tree classifier is given as ")
print(cm)


class_name = ["No","Yes"]
print(class_name)

#plot of decision tree
graph = Source(tree.export_graphviz(dec_tree_model,out_file=None,feature_names = x_train.columns,class_names = class_name, filled=True))
display(SVG(graph.pipe(format="svg")))


#Roc for dec tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve ,auc 
dec_tree_roc=roc_auc_score(y_test,dec_tree_model.predict(x_test))
dec_tree_fpr ,dec_tree_tpr ,threshold =roc_curve(y_test ,dec_tree_model.predict_proba(x_test)[:,1])
dec_tree_auc=auc(dec_tree_fpr,dec_tree_tpr)
print(dec_tree_auc)

plt.figure()
plt.plot(dec_tree_fpr,dec_tree_tpr)
plt.xlabel("false positive rate")
plt.ylabel("True positive rate")
plt.show()



#Using Random forest 
print("Random forest algorithm")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.tree import export_graphviz
from IPython.display import display, Image
import pydotplus

# Spitting the data into train and test 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf_rf=RandomForestClassifier()


#building the model using training set 
rf_model=clf_rf.fit(x_train,y_train)
rf_model.estimators_[0]
len(rf_model.estimators_)
feature_name=x
class_name=y

y_pred=rf_model.predict(x_test)
score=accuracy_score(y_test,y_pred)
print("Score using Random Forest ",score)
print("time taken",time.time()-t0)

import sklearn.metrics as metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print("Accuracy using Random forest: ",metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix

from sklearn import metrics 
cm=metrics.confusion_matrix(y_test,y_pred)
print(cm)



# Using SVM
print("Suport vector machine algorithm")
from sklearn import svm
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf_svc=svm.SVC(kernel="linear")

#building the model using training set 
svm_model=clf_svc.fit(x_train,y_train)
y_pred=clf_svc.predict(x_test)
score=accuracy_score(y_test,y_pred)
print("Score using SVM ",score)
print("time taken",time.time()-t0)

import sklearn.metrics as metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print("Accuracy using SVM : ",metrics.accuracy_score(y_test, y_pred))

# Confusion matrix

from sklearn import metrics 
cm=metrics.confusion_matrix(y_test,y_pred)
print("The confusion matrix for decision tree classifier is given as ")
print(cm)

#plotting of ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve ,auc
#log_roc=roc_auc_score(y_test,model.predict(x_test))
svm_fpr ,svm_tpr ,threshold =roc_curve(y_test ,y_pred)
svm_auc=auc(svm_fpr , svm_tpr)

print(svm_auc)

plt.figure()
plt.plot(svm_fpr,svm_tpr)
plt.xlabel("false positive rate")
plt.ylabel("True positive rate")
plt.show()



