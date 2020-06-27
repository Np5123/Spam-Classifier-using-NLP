import pandas as pd
import numpy as np
import pickle
#importing the dataset
data=pd.read_csv("spam.csv",encoding="latin-1")
data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
data["spam"]=data["v1"].map({"ham":0,"spam":1})

#data cleaning and preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
ps=PorterStemmer()
for i in range(0,len(data)):
    mes=re.sub('[^a-zA-z]',' ',data["v2"][i])
    mes=mes.lower()
    words=mes.split()
    words=[ps.stem(word) for word in words if not word in stopwords.words("english")]
    mes=" ".join(words)
    corpus.append(mes)

#creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)
x=cv.fit_transform(corpus).toarray()
y=data["spam"]

#creating a pickle file
pickle.dump(cv,open("cv-transform.pkl","wb"))

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=13)

#training the model using NaiveBayes
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(x_train,y_train)

#creating a pickle file
pickle.dump(model,open("spam-sms-model.pkl","wb"))