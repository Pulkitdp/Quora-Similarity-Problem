from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
import pandas as pd
import numpy as np

################ Converting 3D arrray of array into 1D array ###############
def arr_convert_1d(arr):
    arr=np.array(arr)
    arr=np.concatenate( arr, axis=0 )
    arr=np.concatenate( arr, axis=0 )
    return arr

############ Cosine Similarity ###############
cos=[]
def cosine(trans):
    cos.append(cosine_similarity(trans[0],trans[1]))

############ Manhatten Distance ###############
manhatten=[]
def manhatten_distance(trans):
    manhatten.append(pairwise_distances(trans[0],trans[1],metric='manhattan'))
    # print(arr_convert_1d(manhatten))

############ euclidean Distance ###############
euclidean=[]
def euclidean_function(vectors):
    euc=euclidean_distances(vectors[0],vectors[1])
    euclidean.append(euc)

############ TF - IDF ###############
def tfidf(str1,str2):
    ques=[]
    dataset=pd.read_csv('quora_duplicate_questions.tsv',delimiter='\t',encoding='utf-8')
    x=dataset.iloc[:,1:5]
    x=x.dropna(how='any')
    for k in range(len(x)):
        for j in [2,3]:
            ques.append(x.iloc[k,j])
    vect=TfidfVectorizer()
    vect.fit(ques)

    corpus=[str1,str2]
    trans=vect.transform(corpus)
    # trans=trans.toarray()
    # tf_idf.append(trans)

    euclidean_function(trans)
    cosine(trans)
    manhatten_distance(trans)
    # minkowski_function(trans)
    # canberra(trans)
    return convert()

############ Minkowski Distance ###############
minkow=[]
def minkowski_function(vectors):
    minkow.append(distance.minkowski(vectors[0],vectors[1], 1).toarray())

############ Canberra Distance ###############
canbe=[]
def canberra(vectors):
    canbe.append(distance.canberra(vectors[0],vectors[1]))

def convert():
    dataf=pd.DataFrame()
    lis2=arr_convert_1d(manhatten)
    dataf['manhatten']=lis2
    lis2=arr_convert_1d(cos)
    dataf['cos_sim']=lis2
    lis2=arr_convert_1d(euclidean)
    dataf['euclidean']=lis2
    return dataf