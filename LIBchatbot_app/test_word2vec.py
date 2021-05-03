import gensim
from gensim.models.word2vec import Word2Vec

import pandas
from pymongo import MongoClient
from collections import Counter, defaultdict

# 請先確認 MongoDB 是否開啟
#sudo service mongod restart

def get_mongodb_row(DatabaseName, CollectionName):
    client = MongoClient("localhost", 27017)
    db = client[DatabaseName]
    collection = db[CollectionName]

    cursor = collection.find({}, {"_id":1, "LibraryName":1, "ID":1, 
         "Question":1, "Answer":1, "Category":1, "Keyword":1, "RelatedQ":1, 
         "Q_WS":1, "A_WS":1, "QA_WS":1, 
         "Q_POS":1, "A_POS":1, "QA_POS":1, 
         "Q_NER":1, "A_NER":1, "QA_NER":1, 
         "Q_WS | POS":1, "A_WS | POS":1, "QA_WS | POS":1, 
         "adjusted_Category":1, "new_Cluster":1})

    Question_list = []
    Q_WS_list = []
    A_WS_list = []
    Answer_list = []
    new_Cluster_list = []
    AllField_list = []
    
    for row in cursor:
        Question_list.append(row['Question'])
        Q_WS_list.append(row['Q_WS'])
        A_WS_list.append(row['A_WS'])
        Answer_list.append(row['Answer'])
        new_Cluster_list.append(row['new_Cluster'])
        AllField_list.append((row['Question'], row['Q_WS'], row['A_WS'], row['Answer'], row['_id']))

    return Question_list, Q_WS_list, A_WS_list, Answer_list, new_Cluster_list, AllField_list

Question_list, Q_WS_list, A_WS_list, Answer_list, new_Cluster_list, AllField_list = get_mongodb_row('Library', 'FAQ')



all_dict = defaultdict(int)
char_dict = defaultdict(int)
word_dict = defaultdict(int)

for row in Q_WS_list:
    for ws in row:
        all_dict[ws] += 1
        if len(ws) == 1:
            char_dict[ws] += 1
        else:
            word_dict[ws] += 1
# print(all_dict.keys())
# print(char_dict.keys())
print(word_dict['食物'])

"""
############## 字元 ##############
#下載 http://nlp.tmu.edu.tw/word2vec/index.html
char_model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/bessyhuang/Downloads/y_360W_cbow_1D_300dim_2020v1.bin', unicode_errors='ignore', binary=True)
char_vocab = list(char_model.wv.vocab) # model_vocab
# print(char_vocab)


for c in char_dict.keys():
    char = c
    try:
        char_similar = char_model.most_similar(char)
        print('- {}\t{}'.format(char, [c for c, score in char_similar]))
    except:
        print('- {}\t該字彙沒有在字典裡'.format(char))
"""
print('----------------------')
############## 單字 ##############
#下載 http://nlp.tmu.edu.tw/word2vec/index.html

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors as KV

word_model = KV.load_word2vec_format('./tmunlp_1.6B_WB_300dim_2020v1.bin', \
 unicode_errors='ignore', binary=True)

word_vocab = list(word_model.vocab) # model_vocab
# print(word_vocab)

"""
while True:
    query = input('Input:')
    res = word_model.most_similar(query, topn = 10)
    for item in res:
        print(item, item[0], '\n')
"""


import pickle

with open('./food_dict.pkl', 'rb') as fp:
    wiki_food_dict = pickle.load(fp)
fp.close()
#print(wiki_food_dict)

stop_expandword_list = ['圖書館', '論文', '格式', '書箱', '個人', '團體', '館內', '刷卡', '館藏']
freq_21_w_list = []
for faq_word, freq in word_dict.items():
    if freq >= 21 and faq_word not in stop_expandword_list:
        freq_21_w_list.append(faq_word)

# 字典擴展
expand_dict = defaultdict(list)

for faq_word in freq_21_w_list: #FAQ
    #try:
    if faq_word in word_vocab: #model
        word_similar = word_model.most_similar(faq_word)
        w_sim_list = [w for w, score in word_similar]
        #print('- {}\t{}'.format(faq_word, w_sim_list))
        for w_sim in w_sim_list:
            if w_sim in word_dict.keys():
                print('- {}\t{}'.format(faq_word, w_sim_list))
                expand_dict[faq_word] = w_sim_list
                break
    else:
        pass
        #print('- {}\t該字彙沒有在字典裡'.format(faq_word))
        
wiki_food_dict.update(expand_dict)
with open('./food_expand_dict.pkl', 'wb') as fp:
    pickle.dump(wiki_food_dict, fp)
fp.close()

