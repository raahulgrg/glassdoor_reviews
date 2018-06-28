# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:35:24 2018

@author: rahul.garg
"""

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import pandas as pd
import pickle
import nltk

with open('D:/Python/synaptics/gdr_assignment_labelled.pkl', 'rb') as f:
    label_data = pd.DataFrame(pickle.load(f))


with open('D:/Python/synaptics/gdr_assignment_pros_cons.pkl', 'rb') as f:
    review_data = pickle.load(f)
   
type(label_data)
label_data.label.value_counts()


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(label_data['pp_sent'], label_data['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(label_data['pp_sent'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(label_data['pp_sent'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(label_data['pp_sent'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(label_data['pp_sent'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

#%%Model Building
#We will implement following different classifiers for this purpose:
#
#Naive Bayes Classifier
#Linear Classifier
#Support Vector Machine
#Bagging Models
#Boosting Models
#Shallow Neural Networks
#Deep Neural Networks
#Convolutional Neural Network (CNN)
#Long Short Term Modelr (LSTM)
#Gated Recurrent Unit (GRU)
#Bidirectional RNN
#Recurrent Convolutional Neural Network (RCNN)
#Other Variants of Deep Neural Networks

#def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
#    # fit the training dataset on the classifier
#    classifier.fit(feature_vector_train, label)
#    
#    # predict the labels on validation dataset
#    predictions = classifier.predict(feature_vector_valid)
#    
#    if is_neural_net:
#        predictions = predictions.argmax(axis=-1)
#    
#    return metrics.accuracy_score(predictions, valid_y)