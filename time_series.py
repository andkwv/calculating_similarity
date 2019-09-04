#%%
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
location = './data/time_series/'
time_series_1 = (location + 'experiment_01.csv')
time_series_2 = (location + 'experiment_02.csv')

#%%
def time_series_similiarity(time_series_1, time_series_2):
    time_series = [open(time_series_1).read(), open(time_series_2).read()]
    vectorized = TfidfVectorizer().fit_transform(time_series)
    similarity_vector = vectorized * vectorized.T
    similarity_vector = similarity_vector.toarray()

    # return similarity_value
    return similarity_vector[0][1]

print(time_series_similiarity(time_series_1, time_series_2))
