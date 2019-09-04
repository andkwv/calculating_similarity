#%%
from sklearn.feature_extraction.text import CountVectorizer
import numpy
from numpy import pad

#%%
data_location = './data/document/business/'
doc1 = (data_location + '001.txt')
doc2 = (data_location + '002.txt')

def doc_similarity(doc1, doc2):
    #
    vocabulary = open(doc1).read().split(' ')
    vocabulary.append(doc2)

    vectorizer = CountVectorizer()
    res = vectorizer.fit_transform(vocabulary).todense()
    vocabulary = vectorizer.vocabulary_

    #
    doc1 = open(doc1).read().split(' ')
    doc2 = open(doc2).read().split(' ')

    #
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    doc1 = vectorizer.fit_transform(doc1).todense()
    doc2 = vectorizer.fit_transform(doc2).todense()

    #
    doc1 = doc1.sum(axis=0)
    doc2 = doc2.sum(axis=0)

    #%%
    dot_product = float(numpy.inner(doc1, doc2))
    doc1_length = numpy.linalg.norm(doc1)
    doc2_length = numpy.linalg.norm(doc2)

    return dot_product / (doc1_length * doc2_length)

print(doc_similarity(doc1, doc2))