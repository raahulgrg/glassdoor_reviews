# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:35:24 2018

@author: rahul.garg
"""
import pandas as pd
import numpy as np
import pickle
import nltk
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import xgboost, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

#%%Reading the data from pickle file
with open('D:/Python/synaptics/gdr_assignment_labelled.pkl', 'rb') as f:
    label_data = pickle.load(f)


with open('D:/Python/synaptics/gdr_assignment_pros_cons.pkl', 'rb') as f:
    review_data = pickle.load(f)
#%% Filtering data on single labels (removing multi label data)

label_data.head()
label_data['label_count'] = ""
for i in range(0,len(label_data)):
    label_data.iloc[i,2] = len(label_data.label.iloc[i])

label_data.head()

label_data_new = label_data.loc[label_data.label_count == 1,:]
label_data_new.shape

#%% Representing Text as Numerical data
#We will use CountVectorizer to "convert text into a matrix of token counts"
#import
from sklearn.feature_extraction.text import CountVectorizer
#instantiate
vect = CountVectorizer()
#fit
vect.fit(label_data_new.pp_sent)
vect.get_feature_names()
# This step deals with removal of all types of noisy entities present in the text.
# Some Noise Removal happens at the method countvectoriser
# language stopwords (commonly used words of a language – is, am, the, of, in etc),
# URLs or links, social media entities (mentions, hashtags),
# punctuations and industry specific words.
# No punctuations
# all words lowercase
# No duplicatessettings

# Transform training data into a 'document term matrix'
label_data_dtm = vect.transform(label_data_new.pp_sent)
label_data_dtm.shape
#convert the sparse matrix into a dense matrix
label_data_dtm.toarray()
#examine the vocabulary and document-term matrix together
pd.DataFrame(label_data_dtm.toarray(), columns = vect.get_feature_names())

# transform testing data into a document-term matrix 
#In order to make a prediction, 
#the new observation must have the same features as the training observations,
# both in number and meaning
review_data.head()
review_data['full_review'] = review_data.pros+review_data.cons
review_data_dtm = vect.transform(review_data.full_review)
review_data_dtm.toarray()
pd.DataFrame(review_data_dtm.toarray(), columns = vect.get_feature_names())

#Summary:
#vect.fit(train) learns the vocabulary of the training data
#vect.transform(train) uses the fitted vocabulary to build a document-term matrix from the training data
#vect.transform(test) uses the fitted vocabulary to build a document-term matrix from the testing data (and ignores tokens it hasn't seen before)

#%%Building Feature Matrix and response variable

label_data_new.shape
label_data_new.dtypes
label_data_new.head()
label_data_new.pp_sent.value_counts()
label_data_new.label.value_counts()
label_data_new['label'] = label_data_new.label.astype('str')
label_data_new.label.value_counts()

#convert label to numerical variables
label_data_new.dtypes
label_data_new['label_num'] = label_data_new.label.factorize()[0]
label_data_new.head()

X = label_data_new.pp_sent
y = label_data_new.label_num

X.shape
y.shape

#why should we do train test split before vectorisation
#purpose to train test split is model evaluation
#past data should be exchangable for future data
#If you vectorise and then train test split, the dtm will contain every single words in either the train and test
# we would want to simulate the real world

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)   
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#%% Vectorising our dataset
#instantiate
vect = CountVectorizer()
#Fit
vect.fit(X_train)
# learn training data vocabulary, then use it to create a document-term matrix
X_train_dtm = vect.transform(X_train)
# examine the document-term matrix# exami 
X_train_dtm    
# transform testing data (using fitted vocabulary) into a document-term matrix# trans 
X_test_dtm = vect.transform(X_test)
X_test_dtm

#%% Building and Evaluating the Model

# The multinomial Naive Bayes classifier is suitable for classification with discrete features  
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm# make  
y_pred_class = nb.predict(X_test_dtm)
# calculate accuracy of class predictions# 
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
#accuracy = 0.94


#%%We will compare multinomial Naive Bayes with logistic regression:

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# train the model using X_train_dtm# train 
logreg.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)
# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)
metrics.confusion_matrix(y_test, y_pred_class)
#accuracy = 0.96

#%% In practice, fractional counts such as tf-idf works better

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X_train)
X_train_tfidf =  tfidf_vect.transform(X_train)
X_test_tfidf =  tfidf_vect.transform(X_test)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(X_train)
X_train_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(X_train)
X_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 
X_test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_test) 
#%% train model
def train_model(classifier, feature_vector_train, label, feature_vector_valid,valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
#%% apply models
def apply_models(models):
results = pandas.DataFrame(columns = ["WordCount","Word-Level TF" , "N-gram TF","Char Level TF"])
for name,model in models:
    a1 = train_model(model, X_train_dtm, y_train, X_test_dtm,y_test)
    a2 = train_model(model, X_train_tfidf, y_train, X_test_tfidf,y_test)
    a3 = train_model(model, X_train_tfidf_ngram, y_train, X_test_tfidf_ngram,y_test)
    a4 = train_model(model, X_train_tfidf_ngram_chars, y_train, X_test_tfidf_ngram_chars,y_test)
    print ("done" , name)
    results.loc[name] = [a1,a2,a3,a4]
return results
#%%We will implement following different classifiers for this purpose:
#Naive Bayes Classifier
#Logistic Regression
#RandomForestClassifier
#XGBoost

models = []
models.append(('NB', naive_bayes.MultinomialNB()))
models.append(('LR', linear_model.LogisticRegression()))
models.append(('RF', ensemble.RandomForestClassifier()))
models.append(('XGB', xgboost.XGBClassifier()))

result = apply_models(models)
print(result)
#
#     WordCount  Word-Level TF  N-gram TF  Char Level TF
#NB    0.946032       0.949206   0.916931       0.891005
#LR    0.966138       0.956614   0.935979       0.935450
#RF    0.929101       0.928042   0.931746       0.883598
#XGB   0.928042       0.929101   0.861376       0.922222

#%% Other ML and Deep Learning Models that can be used:

#Support Vector Machine
#Shallow Neural Networks
#Deep Neural Networks
#Convolutional Neural Network (CNN)
#Long Short Term Modelr (LSTM)
#Gated Recurrent Unit (GRU)
#Bidirectional RNN
#Recurrent Convolutional Neural Network (RCNN)
#Other Variants of Deep Neural Networks

