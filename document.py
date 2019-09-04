#%%
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
data_location = './data/document/business/'
doc1 = (data_location + '001.txt')
doc2 = (data_location + '002.txt')

#%%
def doc_similarity(doc1, doc2):
    documents = [open(doc1).read(), open(doc2).read()]
    vectorized = TfidfVectorizer().fit_transform(documents)
    similarity_vector = vectorized * vectorized.T
    similarity_vector = similarity_vector.toarray()

    # return similarity_value
    return similarity_vector[0][1]

#%%
print(doc_similarity(doc1, doc2).toarray())