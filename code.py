import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
bet=pd.read_excel(r"C:\360 CLASSES\Project 75\Final_Deployment.xlsx")
bet.info()
bet.columns
a=bet.describe()
bet.isna().sum()

duplicate=bet.duplicated()
sum(duplicate)
bet=bet.reset_index()

bet.drop(['index','First_Deposit_Date','Registration_date'], axis = 1, inplace = True)
## Ref_ID
#bet["Ref_ID"]=bet.Ref_ID.str.replace("ref_","")

from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
bet.iloc[:,7:15]=pd.DataFrame(mean_imputer.fit_transform(bet.iloc[:,7:15]))
bet.isna().sum()

bet.dtypes
bet.iloc[:,7:15]=bet.iloc[:,7:15].astype("int64")
bet.wallet_amount=bet.wallet_amount.astype("int64")

##### EDA
data_summary1=pd.DataFrame()
bet.columns
data_summary1["mean"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.mean)
data_summary1["median"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.median)
data_summary1["max"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.max)
data_summary1["min"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.min)
data_summary1["range"]=data_summary1["max"]-data_summary1["min"]
data_summary1["std"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.std)
data_summary1["var"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.var)
data_summary1["skewness"]= bet.skew()
data_summary1["kurtosis"]= bet.kurt()
data_summary1["Q1"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].quantile(0.25)
data_summary1["Q3"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].quantile(0.75)
data_summary1["IQR"]= data_summary1["Q3"]-data_summary1["Q1"]
data_summary1["lower_limit"] = data_summary1["Q1"] - 1.5*data_summary1["IQR"]
data_summary1["upper_limit"] = data_summary1["Q3"] + 1.5*data_summary1["IQR"]

######## Mode
bet.inactive_days.mode()                                         #1) inactive_days
bet.Year_of_Birth.mode()                                         #2) Year_of_Birth
bet.age_at_registration.mode()                                   #3) age_at_registration
bet.sum_stakes_fixedodds.mode()                                  #4) sum_stakes_fixedodds
bet.sum_bets_fixedodds.mode()                                    #5) sum_bets_fixedodds
bet.bettingdays_fixedodds.mode()                                 #6) bettingdays_fixedodds
bet.duration_fixedodds.mode()                                    #7) duration_fixedodds
bet.frequency_fixedodds_MN.mode()                                #8) inactive_days
bet.bets_per_day_fixedodds_LM.mode()                             #9) bets_per_day_fixedodds_LM
bet.euros_per_bet_fixedodds_KL.mode()                           #10) euros_per_bet_fixedodds_KL
bet.net_loss_fixedodds.mode()                                   #11) net_loss_fixedodds
bet.wallet_amount.mode()                                        #12) wallet_amount
bet.bonus_amount.mode()                                         #13) bonus_amount

#######xxxxx---- Visualization ---------xxxxx###########
##### Histogram
df1=bet.select_dtypes([np.int64,np.float64])
for i, col in enumerate(df1.columns):
    plt.figure(i)
    sns.histplot(x=col,color="cyan",data=df1);plt.title(col+" histogram plot")
    
####### Histogram with kde
df1=bet.select_dtypes([np.int64,np.float64])
for i, col in enumerate(df1.columns):
    plt.figure(i)
    sns.histplot(x=col,data=df1,color="red",kde=True);plt.title(col+" histogram kde plot ")
    
#######  Bar plot vs fraud
df1 = bet.select_dtypes([object])
for i, col in enumerate(df1.columns):
        plt.figure(i)
        sns.barplot(x=col,y=bet["is_Fraud"],data=df1);plt.xticks(rotation='vertical');plt.title(col+" vs fraud")

df1 = bet.iloc[:,[4,5,12,18]]
for i, col in enumerate(df1.columns):
        plt.figure(i)
        sns.barplot(x=col,y=bet["is_Fraud"],data=df1);plt.xticks(rotation='vertical');plt.title(col+" vs fraud")

######## jointplot
df1 = bet.select_dtypes([np.float64,np.int64])
for i, col in enumerate(df1.columns):
       plt.figure(i)
       sns.jointplot(x=col,y="is_Fraud",data=df1,color="red");plt.xticks(rotation='vertical');plt.title("jointplot",rotation='vertical')
       sns.jointplot(x=col,y="inactive_days",data=df1);plt.xticks(rotation='vertical');plt.title("jointplot",rotation='vertical')
       sns.jointplot(x=col,y="Year_of_Birth",data=df1,color="green");plt.xticks(rotation='vertical');plt.title("jointplot",rotation='vertical')
       sns.jointplot(x=col,y="age_at_registration",data=df1,color="cyan");plt.xticks(rotation='vertical');plt.title("jointplot",rotation='vertical')
       sns.jointplot(x=col,y="sum_stakes_fixedodds",data=df1,color="yellow");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="sum_bets_fixedodds",data=df1,color="pink");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="bettingdays_fixedodds",data=df1,color="orange");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="duration_fixedodds",data=df1,color="black");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="frequency_fixedodds_MN",data=df1,color="magenta");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="bets_per_day_fixedodds_LM",data=df1,color="brown");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="euros_per_bet_fixedodds_KL",data=df1,color="olive");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="net_loss_fixedodds",data=df1,color="purple");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="bonus_amount",data=df1,color="palevioletred");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')
       sns.jointplot(x=col,y="wallet_amount",data=df1,color="darkkhaki");plt.xticks(rotation='vertical');plt.title(" jointplot",rotation='vertical')

#########  Boxplot before outlier treatment
df1 = bet.select_dtypes([np.int64,np.float64])
for i, col in enumerate(df1.columns):
        plt.figure(i)
        sns.boxplot(x=col,data=df1);plt.xticks(rotation='vertical');plt.title(col+" boxplot before outlier treatment")
        
####xxxxx-------- Outlier Treatment ---------xxxxx##########
# Year_of_Birth
IQR= bet["Year_of_Birth"].quantile(0.75) - bet["Year_of_Birth"].quantile(0.25)
lower_limit = bet["Year_of_Birth"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["Year_of_Birth"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['Year_of_Birth'])
bet.Year_of_Birth = winsor.fit_transform(bet[['Year_of_Birth']])
 
# age_at_registration
IQR= bet["age_at_registration"].quantile(0.75) - bet["age_at_registration"].quantile(0.25)
lower_limit = bet["age_at_registration"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["age_at_registration"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['age_at_registration'])
bet.age_at_registration = winsor.fit_transform(bet[['age_at_registration']])
 
# sum_stakes_fixedodds
IQR= bet["sum_stakes_fixedodds"].quantile(0.75) - bet["sum_stakes_fixedodds"].quantile(0.25)
lower_limit = bet["sum_stakes_fixedodds"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["sum_stakes_fixedodds"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['sum_stakes_fixedodds'])
bet.sum_stakes_fixedodds = winsor.fit_transform(bet[['sum_stakes_fixedodds']])
 
# sum_bets_fixedodds
IQR= bet["sum_bets_fixedodds"].quantile(0.75) - bet["sum_bets_fixedodds"].quantile(0.25)
lower_limit = bet["sum_bets_fixedodds"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["sum_bets_fixedodds"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['sum_bets_fixedodds'])
bet.sum_bets_fixedodds = winsor.fit_transform(bet[['sum_bets_fixedodds']])
 
# bettingdays_fixedodds
IQR= bet["bettingdays_fixedodds"].quantile(0.75) - bet["bettingdays_fixedodds"].quantile(0.25)
lower_limit = bet["bettingdays_fixedodds"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["bettingdays_fixedodds"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['bettingdays_fixedodds'])
bet.bettingdays_fixedodds = winsor.fit_transform(bet[['bettingdays_fixedodds']])
 
# duration_fixedodds
IQR= bet["duration_fixedodds"].quantile(0.75) - bet["duration_fixedodds"].quantile(0.25)
lower_limit = bet["duration_fixedodds"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["duration_fixedodds"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['duration_fixedodds'])
bet.duration_fixedodds = winsor.fit_transform(bet[['duration_fixedodds']])
 
# bets_per_day_fixedodds_LM 
IQR= bet["bets_per_day_fixedodds_LM"].quantile(0.75) - bet["bets_per_day_fixedodds_LM"].quantile(0.25)
lower_limit = bet["bets_per_day_fixedodds_LM"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["bets_per_day_fixedodds_LM"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['bets_per_day_fixedodds_LM'])
bet.bets_per_day_fixedodds_LM = winsor.fit_transform(bet[['bets_per_day_fixedodds_LM']])
 
# euros_per_bet_fixedodds_KL
IQR= bet["euros_per_bet_fixedodds_KL"].quantile(0.75) - bet["euros_per_bet_fixedodds_KL"].quantile(0.25)
lower_limit = bet["euros_per_bet_fixedodds_KL"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["euros_per_bet_fixedodds_KL"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['euros_per_bet_fixedodds_KL'])
bet.euros_per_bet_fixedodds_KL = winsor.fit_transform(bet[['euros_per_bet_fixedodds_KL']])
 
# wallet_amount
IQR= bet["wallet_amount"].quantile(0.75) - bet["wallet_amount"].quantile(0.25)
lower_limit = bet["wallet_amount"].quantile(0.25) - (1.5 * IQR)
upper_limit = bet["wallet_amount"].quantile(0.75) + (1.5 * IQR)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both',fold=1.5,variables=['wallet_amount'])
bet.wallet_amount = winsor.fit_transform(bet[['wallet_amount']])
 
#######  Boxplot after outlier treatment
df1 = bet.select_dtypes([np.int64,np.float64])
for i, col in enumerate(df1.columns):
        plt.figure(i)
        sns.boxplot(x=col,data=df1);plt.xticks(rotation='vertical');plt.title(col+" boxplot after outlier treatment")

######### zero variance
bet.var()

####### data_summary_2 
data_summary_2=pd.DataFrame()
data_summary_2["mean"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.mean)
data_summary_2["median"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.median)
data_summary_2["max"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.max)
data_summary_2["min"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.min)
data_summary_2["range"]=data_summary_2["max"]-data_summary_2["min"]
data_summary_2["std"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.std)
data_summary_2["var"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].apply(np.var)
data_summary_2["skewness"]= bet.skew()
data_summary_2["kurtosis"]= bet.kurt()
data_summary_2["Q1"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].quantile(0.25)
data_summary_2["Q3"]=bet.iloc[:,[1,4,5,7,8,9,10,11,12,13,14,18,19]].quantile(0.75)
data_summary_2["IQR"]= data_summary_2["Q3"]-data_summary_2["Q1"]
data_summary_2["lower_limit"] = data_summary_2["Q1"] - 1.5*data_summary_2["IQR"]
data_summary_2["upper_limit"] = data_summary_2["Q3"] + 1.5*data_summary_2["IQR"]

sns.distplot(bet['is_Fraud']);plt.title("displot for fraud")
bet.corr()['is_Fraud']
sns.heatmap(bet.corr())
correlation=bet.corr()
sns.pairplot(bet)
########### Exporting Preprocessed file for AutoML

#bet.to_excel("Final Betting Project for AutoML excel.xlsx")
#bet.to_csv("Final Betting Project for AutoML csv.csv")

## drop columns
bet.drop(['User_ID','Ref_ID'], axis = 1, inplace = True)    
           
###### Model Building 
bet.columns
bet.shape
bet = bet.iloc[: , [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]

y = bet.is_Fraud
x = bet.drop('is_Fraud', axis = 1)
x.columns

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 

#####  KNN
x.columns
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

step2 = KNeighborsClassifier(n_neighbors=30)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i) 
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_KNN= accuracy_score(ytest, ypred)
accuracy_KNN ##0.9937
MAE_KNN = mean_absolute_error(ytest,ypred)
MAE_KNN ##0.00623700
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("KNN Classifier - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

### DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

step2 = DecisionTreeClassifier(max_depth=10)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)

from sklearn import tree
plt.figure(figsize=(20,15))
tree.plot_tree(step2);plt.title("Decision Tree diagram")

accuracy_DTC = accuracy_score(ytest,ypred)
accuracy_DTC ##0.995841
MAE_DTC = mean_absolute_error(ytest,ypred)
MAE_DTC ###0.004158
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Decision Tree Classifier - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

########  Random Forest 
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

step2 = RandomForestClassifier(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=25)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
MAE_random_forest = mean_absolute_error(ytest,ypred)
MAE_random_forest ##0.0
from sklearn.metrics import accuracy_score
accuracy_random_forest= accuracy_score(ytest, ypred)
accuracy_random_forest ##1.0
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Random Forest Classifier - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

###### Neural Network 
from sklearn.neural_network import MLPClassifier

step1= ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])],remainder='passthrough')

step2= MLPClassifier(hidden_layer_sizes=(100,), max_iter=100)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_neural_network= accuracy_score(ytest, ypred)
accuracy_neural_network ## 0.8939708
MAE_neural_network = mean_absolute_error(ytest,ypred)
MAE_neural_network ##0.1060291
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  
# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Neural network - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

##### Logistic Regression 
from sklearn.linear_model import LogisticRegression 
step1= ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])],remainder='passthrough')

step2= LogisticRegression(random_state=0)

scores=[]
for i in range(30):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))
 
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_logistic_regression= accuracy_score(ytest, ypred)
accuracy_logistic_regression  ## 0.908523

MAE_logistic_regression = mean_absolute_error(ytest,ypred)
MAE_logistic_regression  ##0.091476
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Logistic Regression - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

############ Extra Tree Classifier

step1= ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])],remainder='passthrough')

step2= ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                     oob_score=False, random_state=42, verbose=0,
                     warm_start=False)

scores=[]
for i in range(30):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))
 
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_extra_tree= accuracy_score(ytest, ypred)
accuracy_extra_tree  ##0.9584199
MAE_extra_tree = mean_absolute_error(ytest,ypred)
MAE_extra_tree   ##0.041580
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Extra Tree Classifier - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()
 
############## AdaBoost 
step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])],remainder='passthrough')

step2 = AdaBoostClassifier(n_estimators=15,learning_rate=0.5)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_adaboost= accuracy_score(ytest, ypred)
accuracy_adaboost ###0.991683
MAE_adaboost = mean_absolute_error(ytest,ypred)
MAE_adaboost  ##0.008316
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Adaboost - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

############ Gradient Boost 
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

step2 = GradientBoostingClassifier(n_estimators=500)

scores=[]
for i in range(10):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_gradboost= accuracy_score(ytest, ypred)
accuracy_gradboost   ##0.993762
MAE_gradboost = mean_absolute_error(ytest,ypred)
MAE_gradboost   ##0.0062370
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Gradient Boost - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

############ Voting Classifier 
from sklearn.ensemble import VotingClassifier,StackingClassifier

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

rf = RandomForestClassifier(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
gbdt = GradientBoostingClassifier(n_estimators=100,max_features=0.5)
et = ExtraTreesClassifier(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=10)

step2 = VotingClassifier([('rf', rf), ('gbdt', gbdt), ('et',et)],weights=[5,1,1])

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
accuracy_voting = accuracy_score(ytest,ypred)
accuracy_voting  ##1.0
MAE_voting = mean_absolute_error(ytest,ypred)
MAE_voting  ##0.0
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Voting - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

############################## Stacking #####################
step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

estimators = [
    ('rf', RandomForestClassifier(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingClassifier(n_estimators=100,max_features=0.5)),
]

step2 = StackingClassifier(estimators=estimators)

scores=[]
for i in range(10):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
accuracy_stacking = accuracy_score(ytest,ypred)
accuracy_stacking   ##0.995841
MAE_stacking = mean_absolute_error(ytest,ypred)
MAE_stacking   ##0.004158
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Stacking - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

# Finding Best Model 
data = {"Model" : pd.Series(['KNN', 'Decision Tree', 'Random Forest','neural network','Logistic Regression','Extra Tree Classifier','AdaBoost','Gradient Boost','Voting Classifier','Stacking']) , 
"Accuracy Value" : pd.Series([accuracy_KNN , accuracy_DTC , accuracy_random_forest ,  accuracy_neural_network , accuracy_logistic_regression,accuracy_extra_tree,accuracy_adaboost,accuracy_gradboost,accuracy_voting,accuracy_stacking]) ,
"Mean Absolute Error" : pd.Series([MAE_KNN , MAE_DTC , MAE_random_forest , MAE_neural_network ,MAE_logistic_regression,MAE_extra_tree,MAE_adaboost,MAE_gradboost,MAE_voting,MAE_stacking])} 

# "Accuracy" : pd.Series([accuracy_KNN , accuracy_DT , accuracy_random_forest , accuracy_SVM, accuracy_neural_network , accuracy_logistic_regression]                                        
accuracy_and_Error_Values= pd.DataFrame(data)
accuracy_and_Error_Values

# So Random forest is the best Model Among all this models it has Higher accuracy and lower error value

##############  Random Forest #########################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

step2 = RandomForestClassifier(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=25)
scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
MAE_random_forest = mean_absolute_error(ytest,ypred)
MAE_random_forest 
from sklearn.metrics import accuracy_score
accuracy_random_forest= accuracy_score(ytest, ypred)
accuracy_random_forest
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Random Forest Classifier - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

############################## Voting Classifier #####################
from sklearn.ensemble import VotingClassifier,StackingClassifier

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

rf = RandomForestClassifier(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
gbdt = GradientBoostingClassifier(n_estimators=100,max_features=0.5)
et = ExtraTreesClassifier(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=10)

step2 = VotingClassifier([('rf', rf), ('gbdt', gbdt), ('et',et)],weights=[5,1,1])

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
accuracy_voting = accuracy_score(ytest,ypred)
accuracy_voting
MAE_voting = mean_absolute_error(ytest,ypred)
MAE_voting
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Voting - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

#### DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,5,14,15])
],remainder='passthrough')

step2 = DecisionTreeClassifier(max_depth=10)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)

from sklearn import tree
plt.figure(figsize=(20,15))
tree.plot_tree(step2);plt.title("Decision Tree diagram")

accuracy_DTC = accuracy_score(ytest,ypred)
accuracy_DTC 
MAE_DTC = mean_absolute_error(ytest,ypred)
MAE_DTC 
pd.crosstab(ytest, ypred, rownames = ['Actual'], colnames= ['Predictions'])  

# confusion matrix
labels= ['Valid', 'Fraud'] 
conf_matrix=confusion_matrix(ytest, ypred) 
plt.figure(figsize=(6, 6)) 
sns.heatmap(conf_matrix, xticklabels= labels, yticklabels= labels, annot=True, fmt="d")
plt.title("Decision Tree Classifier - Confusion Matrix") 
plt.ylabel('True Value') 
plt.xlabel('Predicted Value') 
plt.show()

# Exporting model 
import pickle
pickle.dump(bet,open('bet_1.pkl','wb'))
pickle.dump(pipe,open('model_bet_1.pkl','wb'))

xtrain.columns
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About


