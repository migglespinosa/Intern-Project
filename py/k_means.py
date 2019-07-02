# -*- coding: utf-8 -*-
"""
@author: Jake Robertson
"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer
from sklearn.cluster import KMeans as algorithm

preprocessor = spacy.load('en_core_web_sm')

class K_Means:

    clusters = 5

    def __init__(self, dirty):
        self.preprocess(dirty)
        self.cluster()

    def preprocess(self, dirty):
        self.preprocessed = preprocessor(dirty)
        self.clean = []
        for word in self.preprocessed:
            if word.is_stop == False:
                self.clean.append(word.lemma_)

    def cluster(self):
        self.vector = vectorizer(use_idf = True)
        self.matrix = self.vector.fit_transform(self.clean)
        self.model = algorithm(n_clusters = self.clusters, n_init = 1000)
        self.model.fit(self.matrix)
