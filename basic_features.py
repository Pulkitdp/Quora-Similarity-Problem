import pandas as pd
from nltk.tokenize import word_tokenize

dif,diwords,start_word,common_word=[],[],[],[]

def all_data():
    return dif,diwords,start_word,common_word

def diff_len(str1,str2):
    # dif=[]
    q1=len(str(str1))
    q2=len(str(str2))
    dif.append(q1-q2)
    return dif

def diff_words(str1,str2):
    # diwords=[]
    d1=len(str(str1).split())
    d2=len(str(str2).split())
    diwords.append(d1-d2)
    return diwords

def start_words(str1,str2):
    # start_word=[]
    d1=str1.split()
    d2=str2.split()
    if(d1[0]==d2[0]):
        start_word.append(1)
    else:
        start_word.append(0)
    return start_word

def common_words(str1,str2):
    # common_word=[]
    count=0
    d1=str1.split()
    d2=str2.split()
    for i in d1:
        for j in d2:
            if i==j:
                count+=1
    common_word.append(count)
    return common_word
