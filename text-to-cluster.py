# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:30:33 2019

@author: Jake Robertson
"""

import spacy #processing tool
from sklearn.feature_extraction.text import TfidfVectorizer #vectorizing tool
from sklearn.cluster import KMeans #K-Means Text Clustering Algorithm

spacy = spacy.load('en_core_web_sm') #loads processing tool
k = 5 #assigns number of clusters
sample_text = ["we really like the relationship that", "we've had with BNY Mellon it has been a", "great success story for Eagle the", "priorities that be my melon has through", "both acid servicing and asset management", "are extremely aligned with our what our", "clients are asking us to do being able", "to make decisions in terms of features", "and functions and our software", "investments that we make in terms of", "global growth are all aligned with what", "our parent is asking us to do as well", "the combined set of solutions that BNY", "Mellon and Eagle can bring together to", "the table is a real differentiator", "relative to some of our competitors", "niche standalone software vendors that", "can't put that spectrum or services", "together or struggling to build those", "relationships and we look at people that", "are looking to partner with us having", "the BNY Mellon relationship and", "ownership structure behind us gives us", "not only stability but also the ability", "to make investments that we wouldn't be", "able to make on our own", "so when I speak to our clients and", "prospects about what are we investing in", "as an organization it's very important", "for people to understand where we're", "going because a relationship today is", "something that people are looking at for", "the next 10 years they don't want to go", "through a major transformation", "initiative re-platform systems with", "somebody they don't feel comfortable", "doing business with for the next ten", "years so we are continually over", "investing relative to our competitors in", "R&D; over twenty five percent of our", "revenues go back into the product and", "about 50 percent of our resources", "globally are part of our R", "organization we have to continue to be", "an innovative technology company", "providing state-of-the-art solutions to", "our clients and we'll continue to", "reinvest in that in 2015 we have our", "next major release coming out in the", "first quarter filled with lots of new", "features across all of our three product", "suites - the technology investments", "we're making we're also continuing to", "invest in global growth so we're", "expanding our presence in Poland we're", "adding more resources in India we're", "continuing to build out our European", "office our Asia office we're opening a", "new office in Australia the continued", "global investments that we need to make", "for our clients because our clients are", "all growing global to really provide a", "true 24.7 global support model for", "everybody", "so I'm really excited about some of the", "investments that we're gonna see come to", "fruition in 2015", "around our Eagle access business we've", "been investing around people in process", "and technology to provide that platform", "the scalability and growth that we need", "to continue to have that be our primary", "deployment model and it's not just about", "outsourcing the technology around the", "software we're able to put a whole", "portfolio of value-added services on top", "of Eagle access that clients are getting", "a lot of leverage from so things that", "we're doing around the integration with", "data business intelligence solutions", "data replication are just three examples", "of areas where we can build that once", "and have everybody on Eagle access", "leverage that investment that we're", "making so what does that mean it means", "that they get a quicker time to market", "they get a lower cost of ownership and", "they get to leverage the investments", "that we're making for the long run and", "we're going to continue to invest in", "those services on Eagle access", "Eagle pioneered the data centric", "approach to solving problems from our", "inception and we still truly believe", "that that's driving a lot of the", "decisions that people are making and", "it'll be the trends will continue into", "2015", "whether it's an accounting system", "replacement because it's being used not", "fit for purpose and they need to solve a", "data issue around that you know all the", "I board projects out there are data", "centric problems and they end up to", "solve better data for investment", "decision-making or people that want to", "outsource don't want to give up control", "of their data so they need something", "such as the BML an encore solution which", "is where they could outsource but still", "have an eagle data management solution", "coupled with that to provide ownership", "and control over their data and continue", "to leverage that I still see the driving", "force between a lot of the decisions and", "investments that are being made in our", "industry are still going back to getting", "better data for the organization making", "better quality decisions and having", "confidence in the data that you're", "running your business on now we tell us", "that our clients tell us that their data", "that's the asset that's very important", "to them and being able to solve their" , "problems through a data centric model", "still comes back to our core principles" , "as an organization"]

def print_k_means(k_means, vectorizer):
    centroids = k_means.cluster_centers_.argsort()[:,::-1]
    features = vectorizer.get_feature_names()
    for i in range(k):
        cluster = [features[j] for j in centroids[i,:]]
        print(cluster[0])
    return

def cluster(dirty_text):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, use_idf=True) #assigns vectorizer
    matrix = vectorizer.fit_transform(dirty_text) #assigns matrix
    k_means = KMeans(n_clusters=k, n_init=1000) #defines algorithm
    k_means.fit(matrix) #runs algorithm
    return k_means, vectorizer

def tokenize(dirty_text):
    processed_text = spacy(dirty_text) #assigns processed text
    clean_text = [] #defines clean text
    for word in processed_text: #iterates through processed text
        if word.pos_ == "NOUN" and word.lemma_ != "eagle": #if part of speech is noun
            clean_text.append(word.lemma_) #adds word to clean text
    return clean_text
        
def main():
    k_means, vectorizer = cluster(sample_text)
    print_k_means(k_means, vectorizer)
    return
main()
