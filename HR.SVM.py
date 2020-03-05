import numpy as np
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
os.chdir('E:\\Phython Sets\\HR Analysis')
train=pd.read_csv('train_LZdllcl.csv')
test=pd.read_csv('test_2umaH9m.csv')
#check missing valuse
train_missing=train.isnull().sum()
#imputing missing values
train['education'].value_counts()
train['education']=np.where(train['education'].isnull(),"Bachelor's",
                              train['education'])
train['education'].isnull().sum()
for col in train.columns:
    print(col)
train['previous_year_rating'].value_counts().plot(kind='bar')
train['previous_year_rating'].plot.hist()
train['previous_year_rating']=train['previous_year_rating'].fillna(1)
train['previous_year_rating'].isnull().sum()
train_missin_updated=train.isnull().sum()
train['department'].value_counts().plot(kind='bar')
train['region'].value_counts().plot(kind='bar')
train['education'].value_counts().plot(kind='bar')
train['recruitment_channel'].value_counts().plot(kind='bar')
#univarte
train.describe()
#crosstab for chi squre test of catogirical vars
tbl=pd.crosstab(train['education'],train['gender'])
tbl.plot(kind='bar')
from scipy.stats import chi2_contingency
chiqex=chi2_contingency(tbl)
tbl1=pd.crosstab(train['department'],train['is_promoted'])
chiqex1=chi2_contingency(tbl1)
tbl2=pd.crosstab(train['gender'],train['is_promoted'])
chiqex2=chi2_contingency(tbl2)
tbl3=pd.crosstab(train['region'],train['is_promoted'])
chiqex3=chi2_contingency(tbl3)
tbl4=pd.crosstab(train['education'],train['is_promoted'])
chiqex4=chi2_contingency(tbl4)
tbl5=pd.crosstab(train['KPIs_met >80%'],train['is_promoted'])
chiqex5=chi2_contingency(tbl5)
tbl6=pd.crosstab(train['previous_year_rating'],train['is_promoted'])
chiqex6=chi2_contingency(tbl6)
#correlation of contiunous vars
cormat=train.corr()
#derive new var
train['starting_age']=train['age']-train['length_of_service']
train['starting_age'].describe()
train['age'].describe()
train.skew()
train['no_of_trainings'].describe()
train['no_of_trainings'].value_counts()
#encoding the string variables
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['department']=le.fit_transform(train['department'])
train['department'].value_counts()
train['region']=le.fit_transform(train['region'])
train['education']=le.fit_transform(train['education'])
train['gender']=le.fit_transform(train['gender'])
train['recruitment_channel']=le.fit_transform(train['recruitment_channel'])
#splitting Y,X
Y=train['is_promoted']
X=train.iloc[:,1:13]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=123)
import statsmodels.api as sm
logit=sm.Logit(y_train,x_train)
result=logit.fit()
result.summary()
#confusion matrix
from sklearn.metrics import confusion_matrix
smpreds_lr=result.predict(x_train)
smpreds_lr=np.where(smpreds_lr>0.5,1,0)
cm_lr=confusion_matrix(y_train,smpreds_lr)
#calculating f1 score to know how the model performs
from sklearn.metrics import f1_score
f1_score_lr=f1_score(y_train,smpreds_lr)
print(f1_score_lr)
#test the model
preds_sm_test=result.predict(x_test)
preds_sm_test=np.where(preds_sm_test>0.5,1,0)
cm_lr_test=confusion_matrix(y_test,preds_sm_test)
f1_score_lr_tst=f1_score(y_test,preds_sm_test)
print(f1_score_lr_tst)
#####################Decision Tree################
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
preds_dtc=dtc.predict(x_train)
from sklearn.metrics import confusion_matrix
cm_dtc=confusion_matrix(y_train,preds_dtc)
f1_score_lr_dct=f1_score(y_train,preds_dtc)
print(f1_score_lr_dct)
preds_dtc_test=dtc.predict(x_test)
cm_dt_test=confusion_matrix(y_test,preds_dtc_test)
f1_score_lr_dct_tst=f1_score(y_test,preds_dtc_test)
print(f1_score_lr_dct_tst)
##############Random Forest##############
from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier()
rdf.fit(x_train,y_train)
preds_rdf=rdf.predict(x_train)
from sklearn.metrics import confusion_matrix
cm_rdf=confusion_matrix(y_train,preds_rdf)
from sklearn.metrics import f1_score
f1_score_rd=f1_score(y_train,preds_rdf)
print(f1_score_rd)
preds_rdf_test=rdf.predict(x_test)
cm_rdf_test=confusion_matrix(y_test,preds_rdf_test)
f1_score_lr_rdf_tst=f1_score(y_test,preds_rdf_test)
print(f1_score_lr_rdf_tst)
#######################SVM################
from sklearn import svm
SVC=svm.SVC(gamma='auto',C=1.5)
SVC.fit(x_train,y_train)
preds_svm=SVC.predict(x_train)
cm_svc=confusion_matrix(y_train,preds_svm)
f1score_svc=f1_score(y_train,preds_svm)
print(f1score_svc)
preds_svm_test=SVC.predict(x_test)
cm_svm_test=confusion_matrix(y_test,preds_svm_test)
f1_score_svm_tst=f1_score(y_test,preds_svm_test)
print(f1_score_svm_tst)
