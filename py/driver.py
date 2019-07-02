# -*- coding: utf-8 -*-
"""
@author: Jake Robertson
"""

import pandas
from k_means import K_Means

def write(model, vector, clusters):
    cluster_centers = model.cluster_centers_.argsort()[:,::-1]
    cluster_words = vector.get_feature_names()
    for i in range(clusters):
        cluster_topic = [cluster_words[j] for j in cluster_centers[i,:]][0]
        print(cluster_topic)
    return

def read():
    dirty = ""
    for i in range(1, 8):
        file = open(str(i) + ".txt", "r")
        lines = file.readlines()
        for line in lines:
            dirty += " " + line
    return dirty
        
def main():
    dirty = read()
    instance = K_Means(dirty)
    write(instance.model, instance.vector, instance.clusters)
main()

