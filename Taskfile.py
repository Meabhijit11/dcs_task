#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ##### Importing a required libraries for reading a dataset

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[3]:


data = pd.read_csv('Crop_production.csv')


# ##### Read the dataset using the pandas

# In[4]:


data.head()


# In[5]:


data.columns


# ##### Get all the Columns from our Dataset

# In[6]:


data.isnull().sum()


# ##### Finding any null values present in the dataset

# In[7]:


df = pd.DataFrame(data)


# In[8]:


df.info()


# In[9]:


df.shape


# ##### There are 13 Columns & 99849 Rows in our Dataset.

# In[10]:


df.drop('Unnamed: 0',inplace=True,axis=1)


# ##### Drop the column 'Unnamed 0'

# In[11]:


df.head()


# In[12]:


df['State_Name'].unique()


# In[13]:


df['Crop_Type'].unique()


# In[14]:


df['Crop'].unique()


# In[15]:


# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[16]:


# find categorical variables

numerical  = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical  variables\n'.format(len(numerical)))

print('The numerical  variables are :', numerical)


# In[17]:


df.hist(figsize=(20,30)) #summary of distribution for relevant variables


# #### State_Name, Crop, Crop_Type: These are nominal variables
# 
# 
# #Categories do not have a specific order or ranking.

# In[18]:


from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[19]:


categorical_columns = ['State_Name', 'Crop_Type', 'Crop']

# Apply label encoding to each categorical column
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


# ### Note

# ##### The columns = 'State_Name', 'Crop_Type', 'Crop' , are the 'Nominal Variable' so we should have to use "One Hot Encoding", but
# Since these Columns are Carrying mode amount of "Catogies" so for making prediction easy I have converted it into a 'Label Encoding'

# In[20]:


df.head()


# In[21]:


plt.title('Correlation Matrix\n')
sns.heatmap(df.corr(),annot=True)
plt.show()

The given Columns are not having 'Correlation' between them.
# #### Lets find Outliers

# In[22]:


# Calculate IQR
Q1 = df[numerical].quantile(0.25)
Q3 = df[numerical].quantile(0.75)
IQR = Q3 - Q1

# Define thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (df[numerical] < lower_bound) | (df[numerical] > upper_bound)
outliers_data = df[outliers.any(axis=1)]
outliers_data


# In[23]:


# df = df[~outliers.any(axis=1)]


# ##### We can drop the outliers , but since the size of the outliers are too huge so 
# Instead of dropping them , we can use - 'Capping Outliers'

# In[24]:


for column in numerical:
    df[column] = np.where(df[column] > upper_bound[column], upper_bound[column], df[column])
    df[column] = np.where(df[column] < lower_bound[column], lower_bound[column], df[column])


# In[25]:


df.head()


# In[26]:


df.shape


# In[27]:


# Save the preprocessed data

df.to_csv('preprocessed_data.csv', index=False)


# In[28]:


df = pd.read_csv('preprocessed_data.csv')


# ### Training & Testing of the Model

# In[35]:


from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor


# In[36]:


X = df.drop('Yield_ton_per_hec', axis=1)    

y = df['Yield_ton_per_hec']        # Target Variable


# In[37]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state= 100)


# ##### We have Splited dataset into 80:20 format. 80 % for Training & 20 % for Testing.

# In[38]:


scalar = StandardScaler()

x_train = scalar.fit_transform(x_train)

x_test = scalar.transform(x_test)


# In[39]:


ran_forest = RandomForestRegressor(random_state=6)


# In[40]:


ran_forest.fit(x_train,y_train)


# In[41]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(ran_forest.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(ran_forest.score(x_test, y_test)))


# #### Now lets try Random Forest with HyperParameter Tunning

# In[42]:


n_estimators = [5,20,50,100,90,115,130] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = range(2,20,1) # maximum number of levels allowed in each decision tree
min_samples_split = range(2,10,1)  # minimum sample number to split a node
min_samples_leaf = range(1,10,1) # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'bootstrap': bootstrap}


# In[43]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()


# In[44]:


from sklearn.model_selection import RandomizedSearchCV

rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)


# ##### estimator: The machine learning model to be optimized (e.g., RandomForestRegressor).
# ##### param_distributions: A dictionary where keys are hyperparameters and values are lists or distributions to sample from for tuning (e.g., random_grid).
# ##### n_iter: The number of different hyperparameter combinations to try (e.g., 100).
# ##### cv: The number of cross-validation folds to use for evaluation (e.g., 5).
# ##### random_state: A seed for reproducibility of the random sampling process (e.g., 35).
# ##### n_jobs: The number of CPU cores to use for parallel processing (-1 uses all available cores)

# In[47]:


rf_random.fit(x_train, y_train)


# In[48]:


print ('Random grid: ', random_grid, '\n')
# print the best parameters
print ('Best Parameters: ', rf_random.best_params_, ' \n')


# In[49]:


ranf_reg = RandomForestRegressor( max_depth = 19,
 max_features = 'sqrt',
 min_samples_leaf = 6,
 min_samples_split= 2,
 n_estimators = 90,random_state=6)


# In[50]:


ranf_reg.fit(x_train,y_train)


# In[51]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(ranf_reg.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(ranf_reg.score(x_test, y_test)))


# In[53]:


predy_rf = ranf_reg.predict(x_test)


# In[54]:


from sklearn.metrics import r2_score

r_squared_rf = r2_score(y_test, predy_rf)

r_squared_rf


# ##### R2 score of "0.8813" indicate that model performs well in capturing the underlying patterns in the data

# In[55]:


from sklearn.metrics import mean_squared_error

from math import sqrt


# In[63]:


mse_rf = mean_squared_error(y_test,predy_rf)

mse_rf


# In[56]:


rmse_rf = sqrt(mean_squared_error(y_test,predy_rf))

rmse_rf


# ##### RMSE of 0.3875 indicates a moderate level of predictive accuracy

# ### Now Lets Save Our Model for Deployment

# In[58]:


import pickle as pk     # to save the model


# In[60]:


filename = "agri_task.pk"

pk.dump(ranf_reg,open(filename,"wb"))

