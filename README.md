# Python-Project-Code

#importing libraries for data handling and analysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#importing the required packages for -

#1. preprocessing -
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

#2. modelling algorithms -

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


#3. Model building -

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

#importing the dataset
attrition_df = pd.read_csv(r'E:\Imarticus\Project\Project\Final Files\attrition_1.csv')

#displaying the first 5 rows of the data
attrition_df.head()

#displaying the number of rows and columns
attrition_df.shape

#check if missing values are present 
#to display boolean values for missing values.

attrition_df.isnull().sum()

#displaying the count of 'yes' and 'no' values of the target variable
attrition_df['Attrition'].value_counts()

#EDA - visualization
fig=plt.figure(figsize=(18,8))
sns.countplot(x='Age',hue='Attrition',data=attrition_df)
plt.show()

#correlation of attrition with some important features
fig=plt.figure(figsize=(15,8))
sns.countplot(x='DistanceFromHome',hue='Attrition',data=attrition_df)
plt.show()

fig=plt.figure(figsize=(10,8))
sns.countplot(x='JobSatisfaction',hue='Attrition',data=attrition_df)
plt.show()

fig=plt.figure(figsize=(10,8))
sns.countplot(x='PerformanceRating',hue='Attrition',data=attrition_df)
plt.show()

fig=plt.figure(figsize=(10,8))
sns.countplot(x='TrainingTimesLastYear',hue='Attrition',data=attrition_df)
plt.show()

fig=plt.figure(figsize=(15,8))
sns.countplot(x='WorkLifeBalance',hue='Attrition',data=attrition_df)
plt.show()

fig=plt.figure(figsize=(15,6))
sns.countplot(x='YearsAtCompany',hue='Attrition',data=attrition_df)
plt.show()

fig=plt.figure(figsize=(15,6))
sns.countplot(x='YearsInCurrentRole',hue='Attrition',data=attrition_df)
plt.show()

fig=plt.figure(figsize=(10,6))
sns.countplot(x='YearsSinceLastPromotion',hue='Attrition',data=attrition_df)
plt.show()

#relation of attrition variable with categorical variables
total_records= len(attrition_df)
columns = ['Gender','MaritalStatus','OverTime','Department','JobRole','BusinessTravel']

j=0
for i in columns:
    j +=1
    plt.subplot(2,3,j)
    ax1 = sns.countplot(data=attrition_df,x= i,hue="Attrition")
    if(j==4 or j==5 or j==6):
        plt.xticks(rotation=90)
    
plt.subplots_adjust(bottom=1, top=4, right=2.0, wspace = 0.1)
plt.show()

#preprocessing the data
#dropping all fixed and non-relevant variables
attrition_df.drop(['DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate','Over18','PerformanceRating','StandardHours','StockOptionLevel','TrainingTimesLastYear'],axis=1,inplace=True)

#To check the number of rows and columns of the data AFTER dropping the variables
attrition_df.shape

#To check again if missing values are present 
attrition_df.isnull().sum()

#extracting feature and response variables
X=attrition_df.values[:,:-1]
Y=attrition_df.values[:,-1]

X[0,:]

#converting categorical variables into numeric form to make it model ready
cols=[1,2,5,7,10,12,15]
le=LabelEncoder()
for i in cols:
    X[:,i]=le.fit_transform(X[:,i])
Y=le.fit_transform(Y)

X[0,:]

# Using One Hot Encoder on converted categorcial features
ohe = OneHotEncoder(categorical_features=[1,2,5,7,10,12,15])
X = ohe.fit_transform(X).toarray()

#scaling the features
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print(X)

#converting label into numerical form 
Y=Y.astype(int)

#plotting correlation matrix
f,ax = plt.subplots(figsize=(16,14))
corrmat = attrition_df.corr()
sns.heatmap(corrmat, annot= True, 
            xticklabels=corrmat.columns.values,
            yticklabels=corrmat.columns.values)
            
#split the data into train and test
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#building the model using logistic regression
classifier_reg = LogisticRegression()
classifier_reg.fit(X_train,Y_train)

Y_pred_Log=classifier_reg.predict(X_test)

#testing the model
cols = ['Model', 'ROC Score','Precision Score', 'Recall Score','Accuracy Score']
models_report = pd.DataFrame(columns = cols)

tmp1 = pd.Series({'Model': " Logistic Regression ",
                 'ROC Score' : metrics.roc_auc_score(Y_test, Y_pred_Log),
                 'Precision Score': metrics.precision_score(Y_test, Y_pred_Log),
                 'Recall Score': metrics.recall_score(Y_test, Y_pred_Log),
                 'Accuracy Score': metrics.accuracy_score(Y_test, Y_pred_Log)})

Reg_report = models_report.append(tmp1, ignore_index = True)
Reg_report

cfm=confusion_matrix(Y_test,Y_pred_Log)
print(cfm)

#roc_auc score 
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_Log)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

#displaying ROC curve
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

#building model using random forest
classifier_rf = RandomForestClassifier(n_estimators= 1000,
    max_features= 0.3,
    max_depth= 4,
    min_samples_leaf= 2)
classifier_rf.fit(X_train,Y_train)

Y_pred_rf=classifier_rf.predict(X_test)

#testing the model
tmp2 = pd.Series({'Model': " Random Forest ",
                 'ROC Score' : metrics.roc_auc_score(Y_test, Y_pred_rf),
                 'Precision Score': metrics.precision_score(Y_test, Y_pred_rf),
                 'Recall Score': metrics.recall_score(Y_test, Y_pred_rf),
                 'Accuracy Score': metrics.accuracy_score(Y_test, Y_pred_rf)})

rf_report = models_report.append(tmp2, ignore_index = True)
rf_report

cfm=confusion_matrix(Y_test,Y_pred_rf)
print(cfm)

classifier_rf.score(X_train,Y_train)

#roc_auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_rf)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

#plotting roc_auc curve
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

#building model using SVM
classifier_svm = SVC(kernel='rbf')
classifier_svm.fit(X_train,Y_train)

Y_pred_svm=classifier_svm.predict(X_test)

#testing the model
tmp3 = pd.Series({'Model': " SVM ",
                 'ROC Score' : metrics.roc_auc_score(Y_test, Y_pred_svm),
                 'Precision Score': metrics.precision_score(Y_test, Y_pred_svm),
                 'Recall Score': metrics.recall_score(Y_test, Y_pred_svm),
                 'Accuracy Score': metrics.accuracy_score(Y_test, Y_pred_svm)})

svm_report = models_report.append(tmp3, ignore_index = True)
svm_report

cfm=confusion_matrix(Y_test,Y_pred_svm)
print(cfm)

#roc_auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_svm)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

#plotting roc_auc curve
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

#building model using XGboost
classifier_gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
classifier_gbm.fit(X_train, Y_train)

Y_pred_XGboost = classifier_gbm.predict(X_test)

#testing the model
tmp4 = pd.Series({'Model': " XGboost ",
                 'ROC Score' : metrics.roc_auc_score(Y_test, Y_pred_XGboost),
                 'Precision Score': metrics.precision_score(Y_test, Y_pred_XGboost),
                 'Recall Score': metrics.recall_score(Y_test, Y_pred_XGboost),
                 'Accuracy Score': metrics.accuracy_score(Y_test, Y_pred_XGboost)})

XGboost_report = models_report.append(tmp4, ignore_index = True)
XGboost_report

cfm=confusion_matrix(Y_test,Y_pred_XGboost)
print(cfm)

#roc_auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_XGboost)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

#creating ROC curve
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Comparison of the models
cols = ['Model', 'ROC Score', 'Precision Score', 'Recall Score','Accuracy Score']
class_model = pd.DataFrame(columns = cols)
class_model = class_model.append([Reg_report,rf_report,svm_report,XGboost_report])
class_model
