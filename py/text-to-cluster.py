# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:30:33 2019

@author: Jake Robertson
"""

import pandas
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nlp = spacy.load('en_core_web_sm')
k = 5

def print_k_means(raw, k_means, v):
    centroids = k_means.cluster_centers_.argsort()[:,::-1]
    features = v.get_feature_names()
    print("\nTopics:")
    for i in range(k):
        topic = [features[j] for j in centroids[i,:]][0]
        print( topic)
    return

def tokenize(raw):
    doc = nlp(raw) #assign array of words
    tokens = [] #declare array of processed words
    for i in range(len(doc)): #iterate through array of words
        if doc[i].is_stop == False: #if the word is not a stop word
            if doc[i].dep_ == "amod": #if the word is an adjectival modifier
                try:
                    if doc[i - 1].dep_ == "advmod": #if the previous word is an adverbal modifier
                        if doc[i + 1].dep_ == "attr" or doc[i + 1].dep_ == "dobj": #if the next word is an attribute or a direct object
                            print(doc[i - 1].dep_ + " " + doc[i].dep_ + " " + doc[i + 1].dep_) #print dependancies
                            print(doc[i - 1].text + " " + doc[i].text + " " + doc[i + 1].text) #print words
                            tokens.append(doc[i].lemma_) #add word to processed words
                except: #word is at beginning or end of words
                    print("error")
    return tokens

def cluster(raw):
    u = TfidfVectorizer(use_idf = True)
    tokens = tokenize(raw)
    X = u.fit_transform(tokens)
    k_means = KMeans(n_clusters = k, n_init = 1000)
    k_means.fit(X)
    return k_means, u

def read():
    raw = ""
    for i in range(1, 6):
        file = open(str(i) + ".txt", "r")
        lines = file.readlines()
        for line in lines:
            raw += " " + line
    return raw
        
def main():
    raw = read()
    k_means, u = cluster(raw)
    print_k_means(raw, k_means, u)
    return
main()
