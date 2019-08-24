import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer,word_tokenize
import basic_features as bf
import advance_features as af
import tf_idf

def data_x():
    # data=pd.read_csv('quora_duplicate_questions.tsv',delimiter='\t',encoding='utf-8')
    # data=data.iloc[2000:2500,:]
    str1=input('Enter your ques 1 - ')
    str2=input('enter your ques 2 - ')
    daa=[[0,1,2,str1,str2]]
    data=pd.DataFrame(daa)
    x=data.iloc[:,1:5]
    # y=data.iloc[:,5].values
    x=rmpun(x)
    x=lower(x)

    dataf=pd.DataFrame()    
    temp=pd.DataFrame()

    dif,diff_words,start_word,common_word,ratio,sort_ratio,jcc_sim,wo_sh=[],[],[],[],[],[],[],[]

    for i in range(len(x)):
        bf.diff_len(str(x.iloc[i,2]),str(x.iloc[i,3]))
        bf.diff_words(str(x.iloc[i,2]),str(x.iloc[i,3]))
        bf.start_words(str(x.iloc[i,2]),str(x.iloc[i,3]))
        bf.common_words(str(x.iloc[i,2]),str(x.iloc[i,3]))
        af.fuzz_ratio(str(x.iloc[i,2]),str(x.iloc[i,3]))
        af.fuzz_sort(str(x.iloc[i,2]),str(x.iloc[i,3]))
        af.jaccard(str(x.iloc[i,2]),str(x.iloc[i,3]))
        af.word_share(str(x.iloc[i,2]),str(x.iloc[i,3]))
        pos_data=af.parts_of_speech(str(x.iloc[i,2]),str(x.iloc[i,3]))
        temp=temp.append(pos_data,ignore_index=True, sort=False)
        dataframe=tf_idf.tfidf(str(x.iloc[i,2]),str(x.iloc[i,3]))
        print(i)

    dif,diwords,start_word,common_word = bf.all_data() 
    ratio,sort_ratio,jcc_sim,wo_sh=af.all_data()

    dataf['deff_len']=dif
    dataf['diff_words']=diwords
    dataf['start_words']=start_word
    dataf['common']=common_word
    dataf['fuzz_ratio']=ratio
    dataf['fuzz_sort']=sort_ratio
    dataf['juccard']=jcc_sim
    dataf['word_share']=wo_sh
    dataf=pd.concat([dataf,temp], axis = 1)
    dataf=pd.concat([dataf,dataframe], axis = 1)
    # y=np.array(y)
    # dataf['target']=y
    dataf.to_csv('aaa.csv',index=False)

    return dataf

# def data_y(data):
    # data=pd.read_csv('quora_duplicate_questions.tsv',delimiter='\t',encoding='utf-8')
    # y=data.iloc[:,5]
    # return y

def lower(x):
    for i in range(len(x)):
        for j in [2,3]:
            x.iloc[i,j]=x.iloc[i,j].lower()
    return x

def rmpun(x):
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(x)):
        for j in [2,3]:
            temp=''
            token=tokenizer.tokenize(x.iloc[i,j])
            for k in range(len(token)):
                temp+=token[k]+" "
            x.iloc[i,j]=temp
    return x
