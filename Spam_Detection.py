import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


#messages= [line.rstrip() for line in open(r'F:/SMSSpamCollection')]
#stop_words= nltk.download('stopwords')

messages= pd.read_csv('F:/SMSSpamCollection',sep='\t',names=['label','message'])

#pd.set_option('display.max_columns', None)
#print(messages.groupby('label').describe())


messages['length']= [len(i) for i in messages['message']]

#print(messages[messages['label']=='ham']['length'].mean())

#messages.hist(column='length',by='label',bins=60)
#plt.show()

"""
#Removing Punctuations 

a= "hey jane! how are you?"

punc= string.punctuation

a_new= [i for i in a if i not in punc]

a_new= ''.join(a_new)
"""

#tokens ae cleaned text, cleaned from punctuations and stopwords
	
def clean_text(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]



bow_transformer= CountVectorizer(analyzer=clean_text).fit(messages['message'])

#print(len(bow_transformer.vocabulary_))

mess4= messages['message'][3]

bow4= bow_transformer.transform(mess4)

#print(messages) 

messages_bow= bow_transformer.transform(messages['message'])

#print('shape of matrix:',messages_bow.shape)

"""
To check sparsity of the matrix
messages_bow.nnz
"""

tfidf_transformer= TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)

#print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf= tfidf_transformer.transform(messages_bow)




