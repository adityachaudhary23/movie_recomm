
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
# import matplotlib.pyplot as mp
# !pip install surprise
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate 
import seaborn as sns




def get_recommendations(title, cosine_sim=cosine_sim):
    df=pd.read_csv('ratings1.csv')
    #print(df)
    df['rating'] = df['rating'].astype(float)
    p = df.groupby('rating')['rating'].count()
    print(p)
    #mp.boxplot(p)

    df = df[pd.notnull(df['rating'])]
    f = ['count','mean']
    dfout= df.groupby('movieId')['rating'].agg(f)

    print(dfout)
    cut = round(dfout['count'].quantile(0.2),0)
    drop_movie_list = dfout[dfout['count'] < cut].index
    print(cut)
    print(drop_movie_list)
    df = df[~df['movieId'].isin(drop_movie_list)]
    print(df)
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']][:], reader)
    svd = SVD()
    cross_validate(svd, data, measures=['RMSE', 'MAE'])

    dfin=pd.read_csv('title.csv')

    dfi=dfin.drop_duplicates(subset='movieId', keep="last")

    #dfi = dfin.drop_duplicates()
    #dfin.rename(columns = {'id':'movieId'}, inplace = True)

    df1= df[(df['userId'] == 1) & (df['rating'] == 5)]
    print (df1)
    df2=df1.merge(dfi)
    print (df2)

    dfin1=dfi.copy()
    dfin1 = dfin1[~dfin1['movieId'].isin(drop_movie_list)]
    dfin1 = dfin1.reset_index()

    data1 = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

    trainset = data1.build_full_trainset()
    svd.fit(trainset)

    dfin1['Est_Score'] = dfin1['movieId'].apply(lambda x: svd.predict(7, x).est)

    dfin1 = dfin1.drop('movieId', axis = 1)

    dfin1 = dfin1.sort_values('Est_Score', ascending=False)
    return dfin1.head(10)


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    # movie = request.args.get('movie')
    # print(movie)
    r = get_recommendations()
    print("hello")
    print(r)
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run()
