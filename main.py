

import os
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

import pandas as pd
import numpy as np
import surprise
# import matplotlib.pyplot as mp
# !pip install surprise
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate 
import seaborn as sns



def get_recommendations_cs(title):
    df2=pd.read_csv('movie.csv')


    C= df2['vote_average'].mean()
    m= df2['vote_count'].quantile(0.9)

    q_movies = df2.copy().loc[df2['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)




    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string
    df2['overview'] = df2['overview'].fillna('')

    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df2['overview'])

    #Output the shape of tfidf_matrix
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


    # Function that takes in movie title as input and outputs most similar movies
    # def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in df2['title'].values:
        return ('This movie is not in our database.\nPlease check if you spelled it correct using camel casing')
    else:
        # Get the index of the movie that matches the title
        # if title not in df2
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return df2['title'].iloc[movie_indices]




# import os
# #Import TfIdfVectorizer from scikit-learn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from flask import Flask, render_template, request
# # libraries for making count matrix and similarity matrix
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import linear_kernel

# import pandas as pd
# import numpy as np
# import surprise
# # import matplotlib.pyplot as mp
# # !pip install surprise
# from surprise import Reader, Dataset, SVD
# from surprise.model_selection import cross_validate 
# import seaborn as sns




def get_recommendations_svd():
    df=pd.read_csv('ratings1.csv')
    #print(df)
    df['rating'] = df['rating'].astype(float)
    p = df.groupby('rating')['rating'].count()
    # print(p)
    #mp.boxplot(p)

    df = df[pd.notnull(df['rating'])]
    f = ['count','mean']
    dfout= df.groupby('movieId')['rating'].agg(f)

    # print(dfout)
    cut = round(dfout['count'].quantile(0.2),0)
    drop_movie_list = dfout[dfout['count'] < cut].index
    # print(cut)
    # print(drop_movie_list)
    df = df[~df['movieId'].isin(drop_movie_list)]
    # print(df)
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']][:], reader)
    svd = SVD()
    cross_validate(svd, data, measures=['RMSE', 'MAE'])

    dfin=pd.read_csv('title.csv')

    dfi=dfin.drop_duplicates(subset='movieId', keep="last")

    #dfi = dfin.drop_duplicates()
    #dfin.rename(columns = {'id':'movieId'}, inplace = True)

    df1= df[(df['userId'] == 1) & (df['rating'] == 5)]
    # print (df1)
    df2=df1.merge(dfi)
    # print (df2)

    dfin1=dfi.copy()
    dfin1 = dfin1[~dfin1['movieId'].isin(drop_movie_list)]
    dfin1 = dfin1.reset_index()

    data1 = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

    trainset = data1.build_full_trainset()
    svd.fit(trainset)

    dfin1['Est_Score'] = dfin1['movieId'].apply(lambda x: svd.predict(7, x).est)

    dfin1 = dfin1.drop('movieId', axis = 1)

    dfin1 = dfin1.sort_values('Est_Score', ascending=False)
    # print(dfin1.head(20)['title'])
    return dfin1.head(10)


app = Flask(__name__)

@app.route("/")
def home():
    r = get_recommendations_svd()
    q=r['title']
    print(q)
    return render_template('home.html',r=q,t='l')
    # return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    print(movie)
    r = get_recommendations_cs(movie)
    print("hello")
    print(r)
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')


if __name__ == '__main__':
    app.run()



# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template('home.html')

# @app.route("/recommend")
# def recommend():
#     movie = request.args.get('movie')
#     print(movie)
#     r = get_recommendations(movie)
#     print("hello")
#     print(r)
#     if type(r)==type('string'):
#         return render_template('recommend.html',movie=movie,r=r,t='s')
#     else:
#         return render_template('recommend.html',movie=movie,r=r,t='l')



# if __name__ == '__main__':
#     app.run()



