from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
 
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage, TemplateSendMessage, CarouselTemplate, CarouselColumn, URITemplateAction

import requests
from bs4 import BeautifulSoup
from googlesearch import search

from collections import defaultdict, Counter
from math import log, sqrt
from opencc import OpenCC
import pandas as pd
import wptools
import pickle
import sys
sys.path.insert(0, '/home/bessyhuang/AI-Chatbot/LIBchatbot_app')
import pymongo_custom_module as PymongoCM
import text_preprocess as tp
import FAQ_custom_judgement as Judge

from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)

### pip install gdown ###
# data_utils.download_data_gdown("./")
    
# ------------------------------------------------------------------------------------------
Question_list, Q_WS_list, A_WS_list, Answer_list, new_Cluster_list, AllField_list = PymongoCM.get_mongodb_row("Library", "FAQ")
FAQ_df = pd.DataFrame({"content": Q_WS_list, "new_Cluster": new_Cluster_list, "answer": Answer_list})
FAQ_df["clean_msg"] = FAQ_df.content.apply(tp.text_process)

# ------------------------------------------------------------------------------------------
zipped = zip(Q_WS_list, new_Cluster_list)
Cluster_set = set(new_Cluster_list)

def SegList_ClusterGroup(zipped):
    Cluster_dict = defaultdict(list)
	
    for row_Q, row_c in zipped:
        if row_c in Cluster_dict:
            Cluster_dict[row_c] += row_Q
        else:
            Cluster_dict[row_c] = row_Q
    # print(Cluster_dict['Cluster 1'])
    return Cluster_dict

SegString_Cluster_dict = SegList_ClusterGroup(zipped)
Q = SegString_Cluster_dict.values()
cluster = SegString_Cluster_dict.keys()
FAQ_cluster_df = pd.DataFrame({'content': Q, 'category': cluster})

### Input Data
FAQ_cluster_df["clean_msg"] = FAQ_cluster_df.content.apply(tp.text_process)

Q_clean_list = FAQ_cluster_df.clean_msg.tolist()
processed_docs = Q_clean_list

# --------------------------------------------------------------------------------------------
# ====== total_tokens 尚未正規化字彙的出現頻率 ======
NO_norm_vocab_freqs = []
no_norm_doc_list = []
for no_norm_doc in Q_WS_list:
    no_norm_doc_list += no_norm_doc
NO_norm_vocab_freqs.append(Counter(no_norm_doc_list))
total_tokens = len(NO_norm_vocab_freqs[0])


# ====== 建立 vocab_freqs (正規化字彙的出現頻率) ======
vocab_freqs = []
norm_doc_str = ""
for norm_doc in processed_docs:
    norm_doc_str += norm_doc + " "
vocab_freqs.append(Counter(norm_doc_str.split()))
total_vocabs = len(vocab_freqs[0])


# ====== 建立 vocab (正規化字彙的索引) ======
vocab = defaultdict(int)
for term in norm_doc_str.split():
    if term not in vocab:
        vocab[term] = len(vocab)


# ====== doc_TF 儲存每個文件分別的字彙及詞頻Counter ======
doc_TF = []
for norm_doc_STRING in processed_docs:
    norm_doc = list(norm_doc_STRING.split())
    TF = Counter(norm_doc) # 計算詞頻
    #TF_Output: Counter({'館際合作': 2, '回覆': 2, '時間': 2, '多久': 2}) -> One document
    doc_TF.append(TF)


class InvertedIndex:
    def __init__(self, vocab, doc_TF):
        self.vocab = vocab
        self.vocab_len = len(vocab)
        self.doc_len = [0] * len(doc_TF)
        self.doc_TF = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        self.max_doc_len = 0   #Longest Document Length
        
        for docid, term_freqs in enumerate(doc_TF):            
            doc_len = sum(term_freqs.values())            
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len            
            self.total_num_docs += 1
            for term, freq in term_freqs.items():
                term_ID = vocab[term]
                self.doc_ids[term_ID].append(docid)
                self.doc_TF[term_ID].append(freq)                
                self.doc_freqs[term_ID] += 1
                
    def num_docs(self):
        return self.total_num_docs

    def term_show_in_docids(self, term):
        term_ID = self.vocab[term]
        return self.doc_ids[term_ID]

    def term_in_each_doc_TermFreqs(self, term):
        term_id = self.vocab[term]
        return self.doc_TF[term_id]

    def term_show_in_N_docs(self, term): 
        term_ID = self.vocab[term]
        return self.doc_freqs[term_ID]

INV_INDEX = InvertedIndex(vocab, doc_TF)

# ======== TF-IDF ========
def query_tfidf(query, invindex, k=5):
    scores = Counter()   # scores 儲存了 docID 和他們的 TF-IDF分數
    N = invindex.num_docs() # Cluster -1 ~ Cluster 530 => 532 個
    query_vector = []
    query_term = []
    
    for term in query:
        if term in vocab:
            term_show_in_N_docs = invindex.term_show_in_N_docs(term)
            query_idf = log(N / term_show_in_N_docs)
            query_vector.append(query_idf)
            query_term.append(term)
        else:
            query_vector.append(0)
            query_term.append(term)

    for term in query:
        i = 0
        term_show_in_N_docs = invindex.term_show_in_N_docs(term)
        for docid in invindex.term_show_in_docids(term):
            term_in_each_doc_TermFreqs = invindex.term_in_each_doc_TermFreqs(term)[i]     
            term_Maxfreqs_in_doc = max(invindex.term_in_each_doc_TermFreqs(term))

            doc_len = invindex.doc_len[docid] #每個 doc 的長度
            tfidf_cal = log(1 + term_in_each_doc_TermFreqs) * log(N / term_show_in_N_docs) / sqrt(doc_len)

            scores[docid] += tfidf_cal
            i += 1
    return scores.most_common(k)
    
    
# 問題
Q_list = []
for doc in FAQ_cluster_df.content:
    Q_list.append(''.join(doc))

# 回答
A_list = []
sectors = FAQ_df.groupby('new_Cluster')
sectors_len = len(sectors)
for ClusterN in range(-1, sectors_len -1, 1):
    ClusterN_index = list(sectors.get_group('Cluster {}'.format(ClusterN)).index)[0]
    A_list.append(FAQ_df.loc[ClusterN_index].answer)


# ----- wikipedia 擴展關鍵詞 ---------------------------------
with open('./LIBchatbot_app/food_expand_dict.pkl', 'rb') as fp:
    wiki_food_dict = pickle.load(fp)
fp.close()


subCategory_dict = defaultdict(str)
for key, subcat_items in wiki_food_dict.items():
    for item in subcat_items:
        subCategory_dict[item] = key
#print('+++', wiki_food_dict['食物'], subCategory_dict['蔬菜'])


wiki_GroupCategory_list = []
for key in wiki_food_dict.keys():
    wiki_GroupCategory_list.append(wiki_food_dict[key] + [key])
#print('wiki_category_list = ', wiki_GroupCategory_list)


total_wiki = []
for i in wiki_GroupCategory_list:
    total_wiki += i
#print('total_wiki = ', total_wiki)


custom_match_dict = {
    '零食':'食物', '飲料':'飲料', 
    '系統':'電腦', '借閱證':'閱覽證', 
    '團討室':'團體 討論室', '互借':'館際 互借', 
    '智慧財產權':'智財權', '誰': 'wiki', '書籍':'書',
    '那些':'哪些', '哪些':'那些', '開':'開館', 
    '吃':'食物', '別的': '其他', '連不上':'無法 連線'
    } # wiki_category : FAQ_vocab

# ------------------------------------------------------------
# ----- 查詢館藏的停用詞擷取 -----------------------------------
cluster529_stopwords = set()
Cluster529 = list(sectors.get_group('Cluster 529').content)[0]
Cluster194 = list(sectors.get_group('Cluster 194').content)[0]
for ws in Cluster529:
    cluster529_stopwords.add(ws)
for ws in Cluster194:
    if ws == '空中英語教室':
        pass
    else:
        cluster529_stopwords.add(ws)
cluster529_stopwords.add('可以')
cluster529_stopwords.add('幫')
cluster529_stopwords.add('查')
cluster529_stopwords.add('什麼')
cluster529_stopwords.add('推薦')
cluster529_stopwords.add('最近')
cluster529_stopwords.add('好看')

#print('***', cluster529_stopwords)
# ------------------------------------------------------------
# ----- 查詢Wikipedia的停用詞擷取 ------------------------------
cluster530_stopwords = set()
Cluster530 = list(sectors.get_group('Cluster 530').content)[0]
for ws in Cluster530:
    cluster530_stopwords.add(ws)
# print('***', cluster530_stopwords)
# ------------------------------------------------------------

def query_WS(query):
    ws = WS("./data", disable_cuda=False)
    pos = POS("./data", disable_cuda=False)
    #ner = NER("./data", disable_cuda=False)
    
    with open('./LibraryCommonWords/WikiDict_plus_QAkeywordsDict.pkl', 'rb') as fp:
        WikiDict_plus_QAkeywordsDict = pickle.load(fp)
    fp.close()
    dictionary1 = construct_dictionary(WikiDict_plus_QAkeywordsDict)
    
    word_sentence_list = ws([query], 
            segment_delimiter_set = {":" ,"：" ,":" ,".." ,"，" ,"," ,"-" ,"─" ,"－" ,"──" ,"." ,"……" ,"…" ,"..." ,"!" ,"！" ,"〕" ,"」" ,"】" ,"》" ,"【" ,"）" ,"｛" ,"｝" ,"“" ,"(" ,"「" ,"]" ,")" ,"（" ,"《" ,"[" ,"『" ,"』" ,"〔" ,"、" ,"．" ,"。" ,"." ,"‧" ,"﹖" ,"？" ,"?" ,"?" ,"；" ," 　" ,"" ,"　" ,"" ,"ㄟ" ," :" ,"？" ,"〞" ,"]" ,"／" ,"=" ,"？" ," -" ,"@" ,"." ,"～" ," ：" ,"：" ,"<", ">" ," - " ,"──" ,"~~" ,"`" ,": " ,"#" ,"/" ,"〝" ,"：" ,"'" ,"$C" ,"?" ,"?" ,"*" ,"／" ,"[" ,"." ,"?" ,"-" ,"～～" ,"\""},
            recommend_dictionary = dictionary1, # 效果最好！
            coerce_dictionary = construct_dictionary({'OPAC':2, 'OK':2, '滯還金':2, '浮水印':2, '索書號':2, '圖書館':2, '館藏':2, '館內':2, '連不上':2}), # 強制字典 
    )
    print(word_sentence_list[0])
    
    pos_sentence_list = pos(word_sentence_list)
    print(pos_sentence_list[0])

    #entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    #print(entity_sentence_list)
    
    return word_sentence_list[0], pos_sentence_list[0]

    
@csrf_exempt
def callback(request):
    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')
 
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            return HttpResponseForbidden()
        except LineBotApiError:
            return HttpResponseBadRequest()
 
        for event in events:
            if isinstance(event, MessageEvent):  # 如果有訊息事件
                # query預處理，為了讓查詢跟索引內容相同
                query, query_pos = query_WS(event.message.text)
                print('Raw Query:', query)
                clean_query = list(''.join(tp.text_process(query)).split())
                print(clean_query)
                print('---------------------------\n')
                
                final_query = []
                # ----- wikipedia 擴展關鍵詞 -------------------------------------
                for w in clean_query:
                    if (w in total_wiki) and (w not in vocab.keys()):
                        wikiCategory_term = subCategory_dict[w]
                        try:
                            # e.g. 零食
                            query_term = custom_match_dict[wikiCategory_term]
                            final_query.append(query_term)
                        except:
                            # e.g. 食物
                            final_query.append(wikiCategory_term)

                    else:
                        # 沒有在 FAQ ，也沒有在 wiki
                        try:
                            query_term = custom_match_dict[w]
                            if ' ' in query_term:
                                for i in query_term.split():
                                    final_query.append(i)
                            else:
                                final_query.append(w)
                                final_query.append(query_term)
                        except:
                            final_query.append(w)

                print('Final_query =', final_query)
                print('--------------------------------------------------------')
                
                results = query_tfidf(final_query, INV_INDEX)
                msg_list = []
                
                total_score = 0
                for rank, res in enumerate(results):
                    total_score += res[1]
                avg_score = total_score / len(results)
                print('avg_score {} = total_score {} / n_results {}'.format(avg_score, total_score, len(results)))
                
                
                for rank, res in enumerate(results):
                    print("排名 {:2d}\tClusterN {:8d}\tSCORE {:.3f} \n- 訓練語料 {:}".format(rank+1, res[0], res[1], Q_list[res[0]][:]))
                    if res[1] > 0.673:
                        if res[1] > avg_score:
                            # ----- 查詢館藏的關鍵字擷取 -----------------------------------
                            if res[0] - 1 == 529 or res[0] - 1 == 194 or res[0] - 1 == 129:
                                search_FJULIB_KEYWORD = ""
                                for w in final_query: 
                                    if w not in cluster529_stopwords:
                                        search_FJULIB_KEYWORD += w

                                print('\n查詢館藏的關鍵字擷取：', search_FJULIB_KEYWORD)
                                raw_res = 'https://library.lib.fju.edu.tw:444/search~S0*cht/?searchtype=Y&searcharg=' #A_list[res[0]]
                                final_res = raw_res + search_FJULIB_KEYWORD
                                msg = [Q_list[res[0]], final_res]
                                msg_list.append(msg)
                                print("- 系統回覆 {:}\n".format(msg[1]))

                            # ----- 查詢Wikipedia的停用詞擷取 ------------------------------
                            elif res[0] - 1 == 530 or res[0] - 1 == 356:
                                search_WIKI_KEYWORD = ""
                                for w in final_query: 
                                    if w not in cluster530_stopwords:
                                        search_WIKI_KEYWORD += w
                    
                                print('\n查詢 Wikipedia 的關鍵字擷取：', search_WIKI_KEYWORD)
                                raw_res = A_list[res[0]]
                
                                try:
                                    # 摘要
                                    page1 = wptools.page(search_WIKI_KEYWORD, lang='zh').get_restbase('/page/summary/')
                                    print(page1)
                                    summary = page1.data['exrest']
                                    wikiURL = page1.data['url']

                                    cc = OpenCC('s2twp')
                                    final_res = cc.convert(summary) + '\n' + wikiURL
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                                except:
                                    #final_res = '很抱歉，Wikipedia 上找不到 {} 此一條目！'.format(search_WIKI_KEYWORD)
                                    final_res = 'https://zh.wikipedia.org/wiki/{}'.format(search_WIKI_KEYWORD)
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                                break
                                
                            elif res[0] - 1 == 139 or res[0] - 1 == 158 or res[0] - 1 == 65 or res[0] - 1 == 85 or res[0] - 1 == 118:
                                final_res = Judge.OpeningHours_parser(query, query_pos)
                                msg = [Q_list[res[0]], final_res]
                                msg_list.append(msg)
                                print("- 系統回覆 {:}\n".format(msg[1]))
                                break
                            
                            elif res[0] - 1 == 427 or res[0] - 1 == 426:
                                final_res = A_list[res[0]]
                                msg = [Q_list[res[0]], final_res]
                                msg_list.append(msg)
                                print("- 系統回覆 {:}\n".format(msg[1]))
                                break
                                
                            elif res[0] - 1 == 386:
                                """
                                if "疫情" in query:
                                    final_res = Judge.OpeningHours_parser(query, query_pos)
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                                    ###break
                                else:
                                    final_res = A_list[res[0]]
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)         
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                                """
                                final_res = A_list[res[0]]
                                msg = [Q_list[res[0]], final_res]
                                msg_list.append(msg)         
                                print("- 系統回覆 {:}\n".format(msg[1]))
                                
                            elif res[0] - 1 == 318 or res[0] - 1 == 26 or res[0] - 1 == 18:
                                if '大學生' in final_query:
                                    final_res = '大學部學生借閱總數以三十冊為限，借期為二十八日；無人預約時得續借一次。'
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                                
                                elif '研究生' in final_query:
                                    final_res = '研究生借閱總數以四十冊為限，借期為四十二日；無人預約時得續借一次。研究生自入學第二年起，因撰寫學位論文之需而提出申請者，得辦理延長借書。延長借書之借期為六十日；無人預約時得續借一次。'
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                            
                                elif '老師' in final_query or '教師' in final_query:
                                    final_res = '本校教師借閱總數以七十五冊為限，借期為一百二十日；無人預約時得續借一次。教師以其研究計畫專案經費購買之圖書，得辦理「專案借書」，借期至該計畫結束為止，不受第四款冊數及借期之限制。'
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                            
                                elif '校友' in final_query:
                                    final_res = '校友及退休人員借書以五冊為限，借期為二十八日，不得續借。'
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                                
                                else:
                                    final_res = A_list[res[0]]
                                    msg = [Q_list[res[0]], final_res]
                                    msg_list.append(msg)
                                    print("- 系統回覆 {:}\n".format(msg[1]))
                               
                            #elif :
                                #pass
                            
                            else:
                                final_res = A_list[res[0]]
                                msg = [Q_list[res[0]], final_res]
                                msg_list.append(msg)
                                print("- 系統回覆 {:}\n".format(msg[1]))
                                
                    else:
                        final_res = event.message.text
                        msg = [Q_list[res[0]], final_res]
                        msg_list.append(msg)
                        print("- 系統回覆 {:}\n".format(msg[1]))
                        ###break
                                
                #print("排名 {:2d} DOCID {:8d} ClusterN {:8d} SCORE {:.3f} \n內容 {:}\n回覆 {:}\n".format(rank+1, res[0], res[0]-1, res[1], Q_list[res[0]][:50], final_res))
                
                new_msg_list = []
                for q, a in msg_list:
                    if a not in new_msg_list:
                        new_msg_list.append(a)
                line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=a) for a in new_msg_list])
        return HttpResponse()
    else:
        return HttpResponseBadRequest()
