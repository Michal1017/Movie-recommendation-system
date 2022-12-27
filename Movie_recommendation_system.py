from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("movies_metadata.csv")

# check how much data is missing
print("Number of missing features in dataset:")
print(df.isnull().sum())

# our recommendation system will be based on categories like: adult,budget,genres,original language,popularity,vote_avarage
df = df[['title', 'adult', 'budget', 'genres',
         'original_language', 'popularity', 'vote_average', 'overview', 'release_date']]


# extract release year from release date
df['release_year'] = df['release_date'].str.extract(
    r'([0-9]{4})', expand=True).astype(float)

# print some information about dataset after changes
print("First 5 movies of our movie dataset:")
print(df.head())
print("Feauters of movies:")
print(df.columns)
print('Size of our dataset:')
print(str(df.shape))


print("Number of missing features in dataset before removing missing data:")
print(df.isnull().sum())

# removing films which have missing data
df.dropna(inplace=True)

print("Number of missing features in dataset after removing missing data:")
print(df.isnull().sum())

print('Size of our dataset after removing missing data:')
print(str(df.shape))

# extrude only genres names from genres column
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in eval(x)])

df['genres'] = df['genres'].astype(str)

print("Example genres of films after cleaning genres column:")
print(df['genres'].head())

# histogram with number of films with specific avarage vote
plt.hist(df['vote_average'], bins=20, range=(0, 10), rwidth=0.5)
plt.ylabel('Number of films')
plt.title('Histogram with avarage film votes')
plt.show()

print('Number of films which are for adult or not:')
print(df['adult'].value_counts())

print('How many films have specific original language:')
print(df['original_language'].value_counts())

# popularity column has two data types string and float
# we need only float, so string values are convert to float
df['popularity'] = df['popularity'].astype(float)

# histogram with number of films with specific popularity
plt.hist(df['popularity'], bins=200, rwidth=0.5)
plt.ylabel('Number of films')
plt.xlabel('Popularity as float number')
plt.title('Histogram with popularity of films')
plt.show()

# create new feature which include genres, adult, original_language and tagline features
df['mixed_data'] = df['genres'] + " " + \
    df['original_language'] + " " + df['adult'] + " "+df['overview']

# get indices of movies
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])

# function which return titles of similar movies


def biggest_similarities(similarities):
    big_similarities = []
    for i in similarities:
        if(i[1] > 0.95):
            big_similarities.append(i)
    return big_similarities

# function which recommend films with the most similiar genres language and adult rate


def get_recommendations(title, cosine_sim):
    # getting index of movie for which we want recommendation
    index_of_movie = indices[title]

    # getting scores of other films similarities to that movie
    similarities = list(enumerate(cosine_sim[index_of_movie]))

    # sorting this similarities
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    #similarities = biggest_similarities(similarities)

    # get 20 the most similar movies
    similarities = similarities[1:11]

    # get indices of the most similar movies
    recommend_movie_indices = [i[0] for i in similarities]

    # return titles of the most similar movies
    return df['title'].iloc[recommend_movie_indices]


# cosine similarity for mixed data feature
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['mixed_data'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

# get recommendation of similar movies
print("Similar movies to Jumanji:")
print(get_recommendations('Jumanji', cosine_sim))

print("Similar movies to Se7en:")
print(get_recommendations('Se7en', cosine_sim))


def get_same_genres_recommendations(title, cosine_sim):
    # getting index of movie for which we want recommendation
    index_of_movie = indices[title]

    # getting scores of other films similarities to that movie
    similarities = list(enumerate(cosine_sim[index_of_movie]))

    # sorting this similarities
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    similarities = biggest_similarities(similarities)

    # get 20 the most similar movies
    similarities = similarities[1:]

    # get indices of the most similar movies
    recommend_movie_indices = [i[0] for i in similarities]

    # return titles of the most similar movies
    return df['title'].iloc[recommend_movie_indices]


# create matrix with similarities which movie genres are similar  using cosine similarity
count2 = CountVectorizer(stop_words='english')
count_matrix2 = count.fit_transform(df['genres'])

cosine_sim2 = cosine_similarity(count_matrix2, count_matrix2)

print("Films with the same genres like Jumanji:")
print(get_same_genres_recommendations('Jumanji', cosine_sim2))

# normalize of budget, popularity and vote_average feauters
df['budget_norm'] = MinMaxScaler().fit_transform(
    np.array(df['budget']).reshape(-1, 1))
df['popularity_norm'] = MinMaxScaler().fit_transform(
    np.array(df['popularity']).reshape(-1, 1))
df['vote_average_norm'] = MinMaxScaler().fit_transform(
    np.array(df['vote_average']).reshape(-1, 1))
df['release_year_norm'] = MinMaxScaler().fit_transform(
    np.array(df['release_year']).reshape(-1, 1))

# function which recommend movies according budget, avarage vote and popularity


def numerical_features_recommendation(title):
    # getting index of movie for which we want recommendation
    index_of_movie = indices[title]
    euclidian_distance = []

    # calculate euclidian distance for every movie
    for i in range(df.shape[0]):
        distance = sqrt((df['budget_norm'][index_of_movie]-df['budget_norm'][i])**2+(df['popularity_norm'][index_of_movie] -
                        df['popularity_norm'][i])**2+(df['vote_average_norm'][index_of_movie]-df['vote_average_norm'][i])**2+(df['release_year_norm'][index_of_movie]-df['release_year_norm'][i])**2)
        euclidian_distance.append((i, distance))

    euclidian_distance = sorted(euclidian_distance, key=lambda x: x[1])

    # get 10 films with lowest euclidian distance
    euclidian_distance = euclidian_distance[1:11]

    recommend_movie_indices = [i[0] for i in euclidian_distance]

    recommend_movie = df.iloc[recommend_movie_indices]

    return recommend_movie['title']


print('Recommendation from numerical values recommendation for Jumanji:')
print(numerical_features_recommendation('Jumanji'))

# function which combinate two before recommendation functions


def improved_recommendation(title, cosine_sim):
    # getting index of movie for which we want recommendation
    index_of_movie = indices[title]

    # getting scores of other films similarities to that movie
    similarities = list(enumerate(cosine_sim[index_of_movie]))

    # sorting this similarities
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    similarities = similarities[1:21]

    # get indices of the most similar movies
    recommend_movie_indices = [i[0] for i in similarities]

    # get euclidian distance for numerical features in dataset
    euclidian_distance = []
    for i in recommend_movie_indices:
        distance = sqrt((df['budget_norm'][index_of_movie]-df['budget_norm'][i])**2+(df['popularity_norm'][index_of_movie] -
                        df['popularity_norm'][i])**2+(df['vote_average_norm'][index_of_movie]-df['vote_average_norm'][i])**2+(df['release_year_norm'][index_of_movie]-df['release_year_norm'][i])**2)
        euclidian_distance.append((i, distance))

    euclidian_distance = sorted(euclidian_distance, key=lambda x: x[1])

    # get top most similiar movies
    euclidian_distance = euclidian_distance[1:11]

    # get indices of most similiar movies
    recommend_movie_indices = [i[0] for i in euclidian_distance]

    recommend_movie = df.iloc[recommend_movie_indices]

    return recommend_movie['title']


print('Improved recommendation for Jumanji:')
print(improved_recommendation('Jumanji', cosine_sim))

print('Improved recommendation for 8 Mile:')
print(improved_recommendation('8 Mile', cosine_sim))

print('Improved recommandation for The Chronicles of Narnia: Prince Caspian:')
print(improved_recommendation(
    'The Chronicles of Narnia: Prince Caspian', cosine_sim))
