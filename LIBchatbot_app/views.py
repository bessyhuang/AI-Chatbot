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

from collections import defaultdict
from collections import Counter
from math import log, sqrt
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/bessyhuang/Desktop/django_linebot_files/LIBchatbot_app')
import pymongo_custom_module as PymongoCM
import text_preprocess as tp


line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)

# ------------------------------------------------------------------------------------------
Question_list, Q_WS_list, Answer_list, new_Cluster_list, AllField_list = PymongoCM.get_mongodb_row("Library", "FAQ")
FAQ_df = pd.DataFrame({"content": Q_WS_list, "new_Cluster": new_Cluster_list, "answer": Answer_list})
FAQ_df["clean_msg"] = FAQ_df.content.apply(tp.text_process)

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
### total_tokens 尚未正規化字彙的出現頻率
NO_norm_vocab_freqs = []
no_norm_doc_list = []
for no_norm_doc in Q_WS_list:
	no_norm_doc_list += no_norm_doc
NO_norm_vocab_freqs.append(Counter(no_norm_doc_list))
# print(NO_norm_vocab_freqs)

total_tokens = len(NO_norm_vocab_freqs[0])
# print(total_tokens)


### 建立 vocab_freqs (正規化字彙的出現頻率)
vocab_freqs = []
norm_doc_str = ""
for norm_doc in processed_docs:
	norm_doc_str += norm_doc + " "
vocab_freqs.append(Counter(norm_doc_str.split()))
#print(vocab_freqs)


### 建立 vocab (正規化字彙的索引)
vocab = defaultdict(int)
for term in norm_doc_str.split():
	if term not in vocab:
		vocab[term] = len(vocab)
# print('~~~', vocab["借書"], vocab)
# print(len(vocab_freqs[0]))


### doc_term_freqs 儲存每個文件分別的字彙及詞頻Counter
doc_term_freqs = []
for norm_doc in processed_docs:
	# print(list(norm_doc.split()), type(norm_doc))
	norm_doc = list(norm_doc.split())

	# 計算詞頻
	tfs = Counter(norm_doc)
	doc_term_freqs.append(tfs)
# print(len(doc_term_freqs))
# print('+++---+++', doc_term_freqs[0])


class InvertedIndex:
	def __init__(self, vocab, doc_term_freqs):
		# vocab：正規化字彙的索引
		self.vocab = vocab
		# doc_len：每個文件的長度
		self.doc_len = [0] * len(doc_term_freqs)
		# doc_term_freqs：儲存這個 doc 中相對應 doc_ids (doc_vocab) 的詞頻。各個詞出現在每個文件的詞頻
		self.doc_term_freqs = [[] for i in range(len(vocab))]
		# doc_ids：儲存這個 doc 所包含的 所有字彙(doc_vocab) 的 ID
		self.doc_ids = [[] for i in range(len(vocab))]
		# doc_freqs：各個詞總共出現在多少個文件中
		self.doc_freqs = [0] * len(vocab)
		# total_num_docs：總共處理的文件數量
		self.total_num_docs = 0
		# max_doc_len：單一文件最長的長度
		self.max_doc_len = 0

		for docid, term_freqs in enumerate(doc_term_freqs):
			doc_len = sum(term_freqs.values())
			self.max_doc_len = max(doc_len, self.max_doc_len) ###
			self.doc_len[docid] = doc_len
			self.total_num_docs += 1
			for term, freq in term_freqs.items():
				term_id = vocab[term]
				self.doc_ids[term_id].append(docid)
				self.doc_term_freqs[term_id].append(freq)
				self.doc_freqs[term_id] += 1

	def num_terms(self):
		return len(self.doc_ids)

	def num_docs(self):
		return self.total_num_docs

	def docids(self, term):
		term_id = self.vocab[term]
		return self.doc_ids[term_id]

	def freqs(self, term):
		term_id = self.vocab[term]
		return self.doc_term_freqs[term_id] # doc_term_freqs：儲存這個 doc 中相對應 doc_ids (doc_vocab) 的詞頻。各個詞出現在每個文件的詞頻

	def f_t(self, term): 
		term_id = self.vocab[term]
		return self.doc_freqs[term_id] # doc_freqs：各個詞總共出現在多少個文件中

	def space_in_bytes(self):
		# 我們假設每個integer使用8 bytes
		space_usage = 0
		for doc_list in self.doc_ids:
			space_usage += len(doc_list) * 8
		for freq_list in self.doc_term_freqs:
			space_usage += len(freq_list) * 8
		return space_usage

# print("Number of documents = {}".format(len(processed_docs)))
# print("Number of unique terms = {}".format(len(vocab)))
# print("Number of tokens = {}".format(total_tokens))

# print("--- --- ---")
invindex = InvertedIndex(vocab, doc_term_freqs)

# print inverted index stats
# print("documents = {}".format(invindex.num_docs()))
# print("number of terms = {}".format(invindex.num_terms()))
# print("longest document length = {}".format(invindex.max_doc_len))
# print("uncompressed space usage MiB = {:.3f}".format(invindex.space_in_bytes() / (1024.0 * 1024.0)))

# 給定一個查詢(String)和一個索引(Class)，回傳k個文件
def query_tfidf(query, index, k=10):
	# scores 儲存了docID和他們的TF-IDF分數
	scores = Counter()
	N = index.num_docs()
	
	for term in query:
		i = 0
		f_t = index.f_t(term)
		for docid in index.docids(term):
			# f_(d,t)
			f_d_t = index.freqs(term)[i]
			d = index.doc_len[docid]
			tfidf_cal = log(1+f_d_t) * log(N/f_t) / sqrt(d)
			scores[docid] += tfidf_cal
			i += 1
	
	return scores.most_common(k)

Q_list = []
for doc in FAQ_cluster_df.content:
	Q_list.append(''.join(doc))

A_list = []
sectors = FAQ_df.groupby('new_Cluster')
sectors_len = len(sectors)
for ClusterN in range(-1, sectors_len -1, 1):
    ClusterN_index = list(sectors.get_group('Cluster {}'.format(ClusterN)).index)[0]
    # print(FAQ_df.loc[ClusterN_index].answer)
    A_list.append(FAQ_df.loc[ClusterN_index].answer)


def Keyword_filter(stemmed_query):
	query_keyword = []
	for q in stemmed_query:
		print(q)
		if q not in vocab.keys():
			query_keyword.append(q)
		else:
			pass
	return query_keyword

@csrf_exempt
def callback(request):
 
	if request.method == 'POST':
		signature = request.META['HTTP_X_LINE_SIGNATURE']
		body = request.body.decode('utf-8')
 
		try:
			events = parser.parse(body, signature)  # 傳入的事件
		except InvalidSignatureError:
			return HttpResponseForbidden()
		except LineBotApiError:
			return HttpResponseBadRequest()
 
		for event in events:
			if isinstance(event, MessageEvent):  # 如果有訊息事件
				
				# 查詢語句
				stemmed_query = event.message.text.split()

				query_keyword = Keyword_filter(stemmed_query)
				if query_keyword == []:
					msg_list = []
					results = query_tfidf(stemmed_query, invindex)

					count = 0
					for rank, res in enumerate(results):
						if count < 5:
							print("排名 {:2d} DOCID {:8d} ClusterN {:8d} SCORE {:.3f} \n內容 {:}".format(rank+1, res[0], res[0]-1, res[1], Q_list[res[0]][:50]))
							msg = [Q_list[res[0]], A_list[res[0]]]
							msg_list.append(msg)
							count += 1
						else:
							pass

					print(msg_list)
					for i, j in msg_list:
						print(i, j)

					line_bot_api.reply_message(
						event.reply_token,[
							TextSendMessage(text=content) for score, content in msg_list
						])

				else:
					# Wiki
					msg_list = []
					Keyword = ' '.join(query_keyword)
					print(Keyword)
					url = 'https://zh.wikipedia.org/wiki/' + Keyword #'https://www.google.com/search?q={}'.format(Keyword)
					response = requests.get(url)

					wiki_msg = [Keyword, response.url]
					msg_list.append(wiki_msg)
					# line_bot_api.reply_message(  # 回復傳入的訊息文字
					#	 event.reply_token,
					#	 TextSendMessage(text=wiki_msg)
					# )

					# Google
					for url in search(Keyword, stop=2):
						response = requests.get(url)
						soup = BeautifulSoup(response.text, 'html.parser')
						title = soup.find('title').text
						msg = [title, url]
						msg_list.append(msg)

					line_bot_api.reply_message(
						event.reply_token, [
							TextSendMessage(text=url) for title, url in msg_list
						])
		return HttpResponse()
	else:
		return HttpResponseBadRequest()