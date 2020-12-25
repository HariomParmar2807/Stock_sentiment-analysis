# STOCK SENTIMENT ANALYSIS 


# In[ ]:


## import all required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC


# In[22]:


## include the dataset 
dataset = pd.read_csv("Data.csv")
dataset.shape


# In[23]:


dataset.head()


# In[57]:


#split the train and test data according to date
#from sklearn.model_selection import train_test_split
#train,test = train_test_split(dataset,test_size=0.1,random_state=0)
train = dataset[dataset['Date'] < '20150101']
test = dataset[dataset['Date'] > '20141231']
train.shape


# In[58]:


# cleaning the data
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
# change the column names
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head()
# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head()


# In[59]:


# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head()


# In[60]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[74]:


headlines[1]


# In[68]:


#bag of words model classify into two class by svm classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC


# In[69]:


## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)
# implement SvM Classifier
svclassifier = SVC(kernel='linear')
svclassifier.fit(traindataset,train['Label'])


# In[70]:


# implement SvM Classifier

svclassifier = SVC(kernel='linear')
svclassifier.fit(traindataset,train['Label'])


# In[71]:


##Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = svclassifier.predict(test_dataset)
print(predictions)


# In[72]:


#find accuracy of model
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)


# In[ ]:




