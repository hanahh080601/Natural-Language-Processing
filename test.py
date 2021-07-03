import pandas as pd
from typing import Text
import numpy as np
import json

special_chars = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}',
                '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                '9', '']

raw_data = pd.read_csv('./CSV/kinh_doanh.csv')
raw_data = raw_data["Text"]
# print(raw_data)

# Bộ từ điển
VOCAB = {}

# List các từ có trong văn bản
SENTENCE_LIST = []

# Số câu trong đoạn
NUM_SENTENCES = 0

# Xử lí từ câu thành bộ từ điển {key (từ): value (số lần từ đó xuất hiện)}
def prepro_sen(sen):
    global VOCAB
    global SENTENCE_LIST
    dict_local = {}
    sen = str(sen).strip().lower()
    for spechar in special_chars:
        if spechar in sen:
            sen = sen.replace(spechar, '')
    if sen != '':
        SENTENCE_LIST.append(sen)
        list_words = sen.split(" ")
        for word in list_words:
            if word !='':
                if word in VOCAB:
                    VOCAB[word] += 1
                else:
                    VOCAB[word] = 1

# Xử lí từ một văn bản => các câu
def prepro_row(text):
    global NUM_SENTENCES
    data = []
    text = str(text).split("\n")
    data += [para.replace('\r', '').split(' ') for para in text]
    NUM_SENTENCES += len(data)
    for sen in data:
        prepro_sen(sen)

# Xử lí 1 dataframe => 1 bộ từ điển
def prepro_df(text_arr):
    processed_data = []
    for row in text_arr:
        processed_data.append(prepro_row(row))


prepro_df(raw_data)

def termfreq(document, word):
    N = len(document)
    occurance = 0
    for word_doc in document:
        if word_doc == word:
            occurance += 1
    return occurance / N


def inverse_doc_freq(word):
    if word not in VOCAB.keys():
        word_occurance = 1
    else:
        if VOCAB[word] == 0:
            word_occurance = VOCAB[word] + 1
        else:
            word_occurance = VOCAB[word]
    return np.log(NUM_SENTENCES / word_occurance)


def tf_idf(sentence):
    tf_idf_vec = []
    for word in sentence:
        tf = termfreq(sentence, word)
        idf = inverse_doc_freq(word)
        value = tf * idf
        tf_idf_vec.append(value)
    return tf_idf_vec


vector = []
for sentence in SENTENCE_LIST:
    vector.append(tf_idf(sentence=sentence))
with open('data.json', 'w') as fp:
    json.dump(VOCAB, fp, sort_keys=True, indent=4)
df = pd.DataFrame(vector)
df.to_csv("vector.csv", index=True)


'''
#Tính TF với wordDict là bộ từ điển có key là từ, value là số lần xuất hiện, bow là câu được xử lí
# 1 từ , 1 câu ? 
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

#Tính IDF với docList là 1 đoạn văn bản gồm nhiều câu: SENTENCE_LIST
def computeIDF(docList):
    import math
    idfDict = {}
    N = NUM_SENTENCES
    idfDict = dict.fromkeys(docList[0].keys(), 0) # Cái này đ hiểu lắm :)) cười ẻ :))
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))     
    return idfDict

#Tính TF-IDF với tfBow là giá trị trả về của computeTF, idfs là giá trị trả về của computeIDF
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

def final_result():
    vector = {}
    docList = []
    for sentence in SENTENCE_LIST:
        set_word = prepro_sen(sen=sentence)
        tf = computeTF(set_word, set(list(sentence.split(" "))))
        docList.append(set_word)
    idf = computeIDF(docList)
    for sentence in SENTENCE_LIST:
        set_word = prepro_sen(sen=sentence)
        tf_idf = computeTFIDF(set_word, idf)
        vector.update(tf_idf)
    return list(vector.values)
    #print(vector.values)


df = final_result()
final_result = pd.DataFrame(final_result)
with open('data.json', 'w') as fp:
    json.dump(VOCAB, fp, sort_keys=True, indent=4)
final_result.to_csv("Vector_way1.csv",index=True)
'''