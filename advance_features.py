from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.stem import WordNetLemmatizer
import sys

ratio,sort_ratio,jcc_sim,wo_sh=[],[],[],[]

def all_data():
    return ratio,sort_ratio,jcc_sim,wo_sh

############ Fuzz Ratio ##############
def fuzz_ratio(str1,str2):
    # ratio=[]
    rat=fuzz.ratio(str1,str2)
    ratio.append(rat)
    return ratio

############ Fuzz Sort Ratio ##############
def fuzz_sort(str1,str2):
    # sort_ratio=[]
    rat=int(fuzz.token_sort_ratio(str1,str2))
    sort_ratio.append(rat)
    return sort_ratio

############ Parts of Speech ##############
def parts_of_speech(str1,str2):
    noun,verb,adj,adverb,conj,inter,pron,prep=[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]
    # try:
    for i in [str1,str2]:
        word,tags=[],[]
        nou,ver,adjec,adver,con,inte,pro,pre=0,0,0,0,0,0,0,0
        word=word_tokenize(i)
        tags=nltk.pos_tag(word)
        word1=pd.DataFrame(tags)
        for k in range(len(word1)):
            if word1.iloc[k,1]=='RB':
                adver+=1
            elif word1.iloc[k,1]=='CC':
                con+=1
            elif word1.iloc[k,1]=='NN':
                nou+=1
            elif word1.iloc[k,1]=='PRP':
                pro+=1
            elif word1.iloc[k,1]=='VB':
                ver+=1
            elif word1.iloc[k,1]=='JJ':
                adjec+=1
            elif word1.iloc[k,1]=='IN':
                pre+=1    
        if i==str1:
            noun[0].append(nou)
            verb[0].append(ver)
            adj[0].append(adjec)
            adverb[0].append(adver)
            conj[0].append(con)
            pron[0].append(pro)
            prep[0].append(pre)
        elif i==str2:
            noun[1].append(nou)
            verb[1].append(ver)
            adj[1].append(adjec)
            adverb[1].append(adver)
            conj[1].append(con)
            pron[1].append(pro)
            prep[1].append(pre)
    if str1==str2:
        temp=noun[0][1]
        noun[0].pop(1)
        noun[1].append(temp)

        temp=verb[0][1]
        verb[0].pop(1)
        verb[1].append(temp)

        temp=adj[0][1]
        adj[0].pop(1)
        adj[1].append(temp)
        
        temp=adverb[0][1]
        adverb[0].pop(1)
        adverb[1].append(temp)

        temp=conj[0][1]
        conj[0].pop(1)
        conj[1].append(temp)

        temp=pron[0][1]
        pron[0].pop(1)
        pron[1].append(temp)

        temp=prep[0][1]
        prep[0].pop(1)
        prep[1].append(temp)

    dataframe=pd.DataFrame({
        'q1_noun':noun[0],
        'q2_noun':noun[1],
        'q1_verb':verb[0],
        'q2_verb':verb[1],
        'q1_adj':adj[0],
        'q2_adj':adj[1],
        'q1_adverb':adverb[0],
        'q2_adverb':adverb[1],
        'q1_conj':conj[0],
        'q2_conj':conj[1],
        'q1_pronoun':pron[0],
        'q2_pronoun':pron[1],
        'q1_prepo':prep[0],
        'q2_prepo':prep[1],
    })
    return dataframe

############## Jaccard Similarity ##############
def jaccard(str1,str2):
    # jcc_sim=[]
    lem=WordNetLemmatizer()
    d1=str1.split()
    d2=str2.split()
    list1,list2=[],[]
    for j in d1:    
        list1=lem.lemmatize(j)
    for k in d2:
        list2=lem.lemmatize(k)
    jcc_sim.append(jaccard_similarity(list1,list2))
    return jcc_sim

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    # print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)

################### Word Share ##################
def word_share(str1,str2):
    # wo_sh=[]
    d1=str1.split()
    d2=str2.split()
    intersection = len(list(set(d1).intersection(d2)))
    union = (len(d1) + len(d2)) - intersection
    wo_sh.append(float(intersection / union))
    return wo_sh