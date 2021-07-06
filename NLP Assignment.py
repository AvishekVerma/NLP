# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:16:44 2021

@author: Avishek
"""

################################### NLP- Topic Modelling Assignment #########################


#Name: Avishek kumar verma
#Batch Id: 05012021_10A.M


#-----------------------------------------Problem -1 -------------------------------------------#


#Problem Statement-1 
#1)	Perform NLP – Topic Modelling and Text summarization by following all the steps as mentioned below: -
#2)	Data Cleaning using regular expressions, Count Vectorizer, POS Tagging, NER, Topic Modelling (LDA, LSA)
#   and Text summarization.


#--------------------LDA(Latent Dirichlet Allocation)----------------------#

import pandas as pd
import re

#importing data
tweets = pd.read_csv(r'C:\Users\admin\Desktop\D.S-360\22.NLP\Data.csv', usecols=['text'])
tweets.head(10)

HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#'

#Custom function to clean the data
def clean(text):
    text = re.sub(HANDLE, ' ', text)
    text = re.sub(LINK, ' ', text)
    text = re.sub(SPECIAL_CHARS, ' ', text)
    return text

tweets['text'] = tweets.text.apply(clean)
tweets.head(10)

# Building LDA model
from gensim.parsing.preprocessing import preprocess_string

tweets = tweets.text.apply(preprocess_string).tolist()

from gensim import corpora
from gensim.models.ldamodel import LdaModel

dictionary = corpora.Dictionary(tweets)
corpus = [dictionary.doc2bow(text) for text in tweets]

NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)

ldamodel.print_topics(num_words=5)

from gensim.models.coherencemodel import CoherenceModel

#custom function to calculate Coherance score 
def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

#custom function to calculate Coherance value
def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(tweets, dictionary, ldamodel)
        yield coherence

#taking coherance scores for 6 topics
min_topics, max_topics = 10,16
coherence_scores = list(get_coherence_values(min_topics, max_topics))

#Plot graph using matplotlib 
import matplotlib.pyplot as plt

x = [int(i) for i in range(min_topics, max_topics)]

ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);




#-----------------------------------------Problem -2 -----------------------------------------#

#Problem Statement-2
#Perform topic modelling and text summarization on the given text data hint use NLP-TM text file.

#imorting useful libraries for text summarization
import nltk
nltk.download('stopwords')

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest

#Data pre processing
STOPWORDS = set(stopwords.words('english') + list(punctuation))
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

#Defined custom functio to calculate frequency of words
def compute_word_frequencies(word_sentences):
    words = [word for sentence in word_sentences 
                     for word in sentence 
                         if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    # Drop words if too common or too uncommon
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies
####

#Custom defined function to calculate scores of sentence 
def sentence_score(word_sentence, word_frequencies):
    return sum([ word_frequencies.get(word,0) 
                    for word in word_sentence])
###
    
#Custom defned function to extract summary based on sentence scores
def summarize(text:str, num_sentences=3):
    """
    Summarize the text, by return the most relevant sentences
     :text the text to summarize
     :num_sentences the number of sentences to return
    """
    text = text.lower() # Make the text lowercase
    
    sentences = sent_tokenize(text) # Break text into sentences 
    
    # Break sentences into words
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Compute the word frequencies
    word_frequencies = compute_word_frequencies(word_sentences)
    
    # Calculate the scores for each of the sentences
    scores = [sentence_score(word_sentence, word_frequencies) for word_sentence in word_sentences]
    sentence_scores = list(zip(sentences, scores))
    
    # Rank the sentences
    top_sentence_scores = nlargest(num_sentences, sentence_scores, key=lambda t: t[1])
    
    # Return the top sentences
    return [t[0] for t in top_sentence_scores]
###
    
#loading text dataset to python
with open('C:\\Users\\admin\\Desktop\\D.S-360\\22.NLP\\NLP-TM.txt', 'r',encoding="utf8") as file:
    lor = file.read()

lor

len(sent_tokenize(lor))

summarize(lor)

summarize(lor, num_sentences=1)





