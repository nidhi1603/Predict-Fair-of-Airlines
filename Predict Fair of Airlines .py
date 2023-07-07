#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### DATA READING

# In[5]:


df = pd.read_excel(r"C:\Users\Asus\Downloads\Data_Train.xlsx")


# In[6]:


df.head()


# In[7]:


df.tail(4)


# ### DEALING WITH THE MISSING DATA
# 

# In[8]:


df.info()


# In[9]:


df.isnull()


# In[10]:


## EXTRACT ALL THE ROWS THAT HAS MISSING VALUES
df[df['Total_Stops'].isnull()]


# In[11]:


## SINCE THE NO. OF ROWS HAVING MISSING VALUES IS 1 THEREFORE, WE CAN THINK TO DROP THIS ROW. 


# In[12]:


df.dropna(inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df.dtypes


# In[15]:


##OBJECT IS APPROXIMATELY EQUAL TO STRINGS
##INT64 BELONGS TO NUMPY PACKAGE 
## VARIOUS PACKAGES--> INT64, INT32, INT16
##INT16 MEANS IT HAS A LENGTH OF 16 BITS/2 BYTES WITH A RANGE OF (2^N-1)-1 ) TO (2^N-1)


# ### PERFORM PRE-PROCESSING

# In[16]:


## CONVERT INTO SUITABLE DATA TYPES SUCH AS DATE AND TIME


# In[17]:


df.dtypes


# In[18]:


def change_into_Datetime(col):
    df[col] = pd.to_datetime(df[col])


# In[19]:


import warnings
from warnings import filterwarnings
filterwarnings("ignore")


# In[20]:


df.columns


# In[21]:


for feature in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']:
    change_into_Datetime(feature)


# In[22]:


df.dtypes


# In[23]:


df['Journey_Month'] = df['Date_of_Journey'].dt.month


# In[24]:


df['Journey_Day'] = df['Date_of_Journey'].dt.day


# In[25]:


df['Journey_Year'] = df['Date_of_Journey'].dt.year


# In[26]:


df.head(6)


# In[27]:


def extract_hour_min(df,col):
    df[col+"_hour"] = df[col].dt.hour
    df[col+"_minute"] =df[col].dt.minute
    return df.head(3)
    


# In[28]:


df.columns


# In[29]:


extract_hour_min(df, "Arrival_Time")


# In[ ]:





# In[30]:


extract_hour_min(df, "Dep_Time")


# In[31]:


cols_to_drop = ['Arrival_Time', 'Dep_Time']
df.drop(cols_to_drop, axis=1, inplace = True)


# In[32]:


df.head()


# In[33]:


df.shape


#    ### DATA ANALYSIS
#     1. LETS ANALYSE WHEN WILL MOST OF THE FLIGHTS TAKE OFF
#     (WE CAN COME UP WITH A BAR CHART)

# In[34]:


def flight_dep_time(x):
    
    if(x>4) and (x<=8):
        return "Early Morning"
    elif (x>8) and (x<=12):
        return "Morning"
    elif (x>12) and (x<=16):
        return "Noon"
    elif (x>16) and (x<=20):
        return "Evening"
    elif (x>20) and (x<=24):
        return "Night"
    else:
        return "Late Night"


# In[35]:


df['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind = "bar", color= "green")


# In[36]:


## For making interactive plots use plotly


# In[37]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install chart_studio')
get_ipython().system('pip install cufflinks')


# In[38]:


import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot, iplot, init_notebook_mode, download_plotlyjs
init_notebook_mode(connected = True)
cf.go_offline()


# In[39]:


df['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind = "bar", color= "green")


# ### DATA ANALYSIS
# 
#    ##### 2. LETS ANALYSE WHETHER DURATION IMPACTS ON PRICE OR NOT
#     

# In[40]:


def preprocess_duration(x):
    if 'h' not in x:
      x =  '0h' + ' ' + x
    elif 'm' not in x:
        x = x + ' ' + '0m'
        
    return x


# In[41]:


df['Duration'].apply(preprocess_duration)


# In[42]:


##lambda function--> known as anonymous function


# In[43]:


df['Duration_hours'] = df['Duration'].apply(lambda x: int(x.split(' ')[0][0:-1]))


# In[44]:


df['Duration'].apply(lambda x: int(x.split(' ')[1][0:-1]) if len(x.split(' ')) > 1 else 0)


# In[45]:


df.head()


# In[46]:


df['Duration_total_mins'] = df['Duration'].str.replace('h', "*60").str.replace(' ', "+").str.replace('m', "*1").apply(eval)


# In[47]:


#WHEN BOTH FEATURE ARE CONTINUOUS IN NATURE USE SCATTER OR REGRESSION PLOT 


# In[48]:


sns.scatterplot(x= "Duration_total_mins", y="Price", data=df)


# In[49]:


sns.scatterplot(x= "Duration_total_mins", y="Price", hue = 'Total_Stops',data=df)


# In[50]:


sns.lmplot(x= "Duration_total_mins", y="Price",data=df)


# ### DATA ANALYSIS
# 
#    3. ON WHICH ROUTE JET AIRWAYS IS EXTREMELY USED
#    
#    4. AIRLINE VS PRICE ANALYSIS minimum price, 25th, 50th, 75th percentile, median, mean values, maximum values ---> box plot
#    
#     

# In[51]:


jet_airways = df['Airline']=='Jet Airways'


# In[52]:


df[jet_airways].groupby('Route').size().sort_values(ascending=False)


# In[53]:


sns.boxplot(y = 'Price', x='Airline', data = df.sort_values('Price', ascending = False))
plt.xticks(rotation = 'vertical')
plt.show()


# ### APPLY ONE HOT ON DATA---> FEATURE ENCODING 
# CONVERTING YOUR STRING DATA OR CONVERTING YOUR CATEGORICAL FEATURES INTO NUMERICAL FEATURES FOR MACHINE TO UNDERSTAND
# 
# 

# In[56]:


df.columns


# In[59]:


cat_col = [col for col in df.columns if df[col].dtype=='object']


# In[60]:


num_col = [col for col in df.columns if df[col].dtype!='object']


# In[61]:


cat_col


# In[65]:


df['Source'].unique()


# In[64]:


df['Source'].apply(lambda x: 1 if x=='Banglore' else 0)


# In[67]:


for sub_category in df['Source'].unique():
    df['Source_' + sub_category] = df['Source'].apply(lambda x: 1 if x==sub_category else 0)


# In[68]:


df.head(2)


# In[70]:


df['Airline'].unique()


# ### FEATURE ENGINEERING
# 
# 1. PERFORM TARGET GUIDE ENCODING ON DATA
# 2. PERFORM MANUAL ENCODING ON DATA 

# In[78]:


airlines = df.groupby(['Airline'])['Price'].mean().sort_values().index


# In[79]:


airlines


# In[85]:


dict_airlines = {key:index for index,key in enumerate(airlines,0)}
    


# In[86]:


dict_airlines


# In[90]:


df['Airline'] = df['Airline'].map(dict_airlines)


# In[91]:


df.head(3)


# In[92]:


df['Destination'].unique()


# In[93]:


df['Destination'].replace('New Delhi', 'Delhi', inplace = True)


# In[94]:


df['Destination'].unique()


# In[95]:


destination  = df.groupby(['Destination'])['Price'].mean().sort_values().index


# In[96]:


destination


# In[97]:


dict_destination = {key:index for index,key in enumerate(destination,0)}


# In[98]:


dict_destination


# In[100]:


df['Destination'] = df['Destination'].map(dict_destination)


# In[101]:


df.head(3)


# ### FEATURE ENCODING 
# 1. PERFORM MANUAL ENCODING ON DATA
# 2. REMOVE UN-NECESSARY FEATURES

# In[102]:


df['Total_Stops'].unique()


# In[103]:


stop = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[104]:


df['Total_Stops'] = df['Total_Stops'].map(stop) 


# In[105]:


df['Total_Stops']


# In[106]:


df.head(1)


# In[107]:


df.columns


# In[110]:


df['Additional_Info'].value_counts()/len(df)*100


# In[123]:


df.drop(columns= ['Date_of_Journey', 'Additional_Info','Duration_total_mins', 'Source', 'Journey_Year', 'Route', 'Duration'], inplace = True)


# In[ ]:





# In[124]:


df.head(3)


# ### OUTLIER DETECTION AND DEALING WITH THE OUTLIERS

# In[126]:


### FOR HANDELLING OUTLIERS IN A DATA
def plot(df,col):
    fig, (ax1, ax2,ax3) = plt.subplots(3,1)
    
    sns.distplot(df[col], ax=ax1)
    sns.boxplot(df[col], ax=ax2)
    sns.histplot(df[col], ax=ax3, kde=False)


# In[127]:


plot(df, "Price")


# In[131]:


### FROM THIS PLOT YOU CAN DETECT WHERE ARE THE OUTLIERS
#IQR APPROACH- INTERQUARTILE RANGE (75-25)%ILE VALUE
#MINIMUM =  Q1- 1.5* IQR     (50%ILE = IQR)
#MAXIMUM =  Q3 + 1.5* IQR    (Q1 = 25TH %ILE  Q3= 75TH %ILE)
#REPLACE THE OUTLIERS WITH MEDIAN OF THE PRICE
#IT IS ONE OF THE BEST APPROACH


# In[132]:


q1 = df['Price'].quantile(0.25)
q3 = df['Price'].quantile(0.75)

iqr = q3-q1

maximum = q3 + 1.5*iqr
minimum = q1 - 1.5*iqr


# In[133]:


maximum


# In[134]:


minimum


# In[149]:


outlier = [price for price in df['Price']  if price > maximum or price < minimum]


# In[150]:


print(outlier)


# In[151]:


len(outlier)


# In[152]:


df['Price'] = np.where(df['Price']>=35000, df['Price'].median(), df['Price'])


# In[153]:


plot(df, 'Price')


# In[154]:


y = df['Price']
(/, DEPENDENT, DATAFRAME)


# In[155]:


x = df.drop(['Price'], axis = 1) // INDEPENDENT DATA FRAME


# In[156]:


x


# In[158]:


from sklearn.feature_selection import mutual_info_regression


# In[160]:


imp = mutual_info_regression(x,y)


# In[161]:


imp


# In[166]:


imp_df = pd.DataFrame(imp, index = x.columns)


# In[167]:


imp_df.columns = ['Importance']


# In[168]:


imp_df


# In[170]:


imp_df.sort_values(by = 'Importance', ascending = False)


# ### MODEL BUILDING

# In[172]:


# RANDOM FOREST FOLLOWS ENSABLE LEARNING APPROACH I.E. IT LEARNS FROM MULTIPLE MODELS AND AT THE END IT WILL COMBINE ALL THE LEARNINGS
## COLLECTION OF MULTIPLE DECISION TREES

### NOW WHAT IS DECISION TREE HOW YOU CAN BUILD AND WHAT ARE THE PARAMETERS
#### ML MODEL FOR CLASSIFICATION AND REGRESSION USE CASES
##### A BASE ALGORITHM WHICH IS USED IN EVERY ENSEMBLE TECHNIQUE

###### DECISION TREE- ACCORDING TO CONDITION HOW YOU WORK


#RANDOM FOREST IS  A COLLECTION OF MULTIPLW DECISION TREES
#BOOTSTRAP AGGREGATION OR BAGGING- A MULTIPLE BAGS ARE CREATED THAT AGGREGRATES THE PREDICTION
#ROW AND COLUMN SAMPLING- SUPPOSE IN SAMPLE 1 200 COLUMNS AND IT DOES SOME DECISION TREE IS MADE.
#DECSION TREE HAS A PROPERTY THAT IT HAS HIGH VARIENCE
#HIGH VARIANCE IS COVERTED TO LOW VARIANCE --> WORK OF RANDOM FOREST (RF)


# In[174]:


##LETS BUILD AND SAVE THE MODEL
##WE NEED TRAINING AND TEST DATA
#TRAINING DATA IS THAT DATA FROM WHICH MODEL WILL LEARN SOME TYPE OF RELATIONSHIP
#ML-
#REGRESSION- CAN YOU PREDICT PRICE, SALARY, RATING
#CLASSFICATION- CAN YOU PREDICT WHETHER A PARTICULAR PERSON IS DIABITC OR NOT, 
#CLUSTERING- CLUSTER OF FEW OBJECTS THAT HAS SAME TRAITS


# In[222]:


from sklearn.model_selection import train_test_split


# In[223]:


X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)


# In[224]:


from sklearn.ensemble import RandomForestRegressor


# In[225]:


ml_model  = RandomForestRegressor()


# In[226]:


ml_model.fit(X_train, y_train)


# In[227]:


y_pred =  ml_model.predict(X_test)
    


# In[228]:


y_pred


# In[229]:


from sklearn import metrics


# In[191]:


metrics.r2_score(y_test, y_pred)


# In[192]:


get_ipython().system('pip install pickle')


# In[230]:


import pickle


# In[231]:


file = open(r'C:\Users\Asus\OneDrive\Desktop\cpp/rf_random.pkl', 'wb')


# In[232]:


pickle.dump(ml_model,file)


# In[233]:


model = open(r'C:\Users\Asus\OneDrive\Desktop\cpp/rf_random.pkl', 'rb')


# In[234]:


forest = pickle.load(model)


# In[235]:


y_pred2 = forest.predict(X_test)


# In[236]:


metrics.r2_score(y_test, y_pred2)


# In[237]:


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


# In[238]:


mape(y_test, y_pred)


# In[240]:


from sklearn import metrics


# 

# In[242]:


def predict(ml_model):
    model = ml_model.fit(X_train , y_train)
    print('Training score : {}'.format(model.score(X_train , y_train)))
    y_predection = model.predict(X_test)
    print('predictions are : {}'.format(y_predection))
    print('\n')
    r2_score = metrics.r2_score(y_test , y_predection)
    print('r2 score : {}'.format(r2_score))
    print('MAE : {}'.format(metrics.mean_absolute_error(y_test , y_predection)))
    print('MSE : {}'.format(metrics.mean_squared_error(y_test , y_predection)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(y_test , y_predection))))
    print('MAPE : {}'.format(mape(y_test , y_predection)))
    sns.distplot(y_test - y_predection)


# In[243]:


predict(RandomForestRegressor())


# In[244]:


from sklearn.tree import DecisionTreeRegressor


# In[245]:


predict(DecisionTreeRegressor())


# how to hypertune ml model

# In[246]:


## how to select which ML algo we should apply for
## ans is use Multiple Algos,then go for Hyper-parameter Optimization,then for Cross Validation then go for various metrics 
## & based on domain expertise knowledge Then I can say ya this model perfoms best


# ### Hyperparameter Tuning or Hyperparameter Optimization
#     1.Choose following method for hyperparameter tuning
#         a.RandomizedSearchCV --> Fast way to Hypertune model
#         b.GridSearchCV--> Slower way to hypertune my model
#     2.Choose ML algo that u have to hypertune
#     2.Assign hyperparameters in form of dictionary or create hyper-parameter space
#     3.define searching &  apply searching on Training data or  Fit the CV model 
#     4.Check best parameters and best score

# In[247]:


from sklearn.model_selection import RandomizedSearchCV


# In[248]:


### initialise your estimator
reg_rf = RandomForestRegressor()


# In[249]:


np.linspace(start =100 , stop=1200 , num=6)


# In[250]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start =100 , stop=1200 , num=6)]

# Number of features to consider at every split
max_features = ["auto", "sqrt"]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start =5 , stop=30 , num=4)]

# Minimum number of samples required to split a node
min_samples_split = [5,10,15,100]


# In[251]:


# Create the random grid or hyper-parameter space

random_grid = {
    'n_estimators' : n_estimators , 
    'max_features' : max_features , 
    'max_depth' : max_depth , 
    'min_samples_split' : min_samples_split
}


# In[252]:


random_grid


# In[253]:


## Define searching

# Random search of parameters, using 3 fold cross validation
# search across 576 different combinations


rf_random = RandomizedSearchCV(estimator=reg_rf , param_distributions=random_grid , cv=3 , n_jobs=-1 , verbose=2)


# In[254]:


rf_random.fit(X_train , y_train)


# In[255]:


rf_random.best_params_


# In[256]:


rf_random.best_estimator_


# In[257]:


rf_random.best_score_


# In[ ]:




