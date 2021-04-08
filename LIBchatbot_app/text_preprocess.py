from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import TfidfVectorizer



def text_process(mess):
    with open("/home/bessyhuang/AI-Chatbot/LIBchatbot_app/StopWords/stop_words.txt", encoding="utf-8") as f:
        stop_words = f.read().splitlines()

    CustomStopwords = ['「', '」', '，', '。', '？', '、', '【', '】', '（', '）', '.', '﹖']
    CustomStopwords += stop_words

    STOPWORDS = stopwords.words('english') + CustomStopwords

    # 非標點符號的 characters，就存進 list
    nopunc = ''
    for char in mess:
        if 'Turnitin' in char:
            char = 'Turnitin'
            nopunc = nopunc + char + ' '
        elif 'Ezproxy' in char:
            char = 'Ezproxy'
            nopunc = nopunc + char + ' '
        elif 'TEJ' in char:
            char = 'TEJ'
            nopunc = nopunc + char + ' '
        elif 'MOD' in char:
            char = 'MOD'
            nopunc = nopunc + char + ' '
        elif 'Proxy' in char:
            char = 'Proxy'
            nopunc = nopunc + char + ' '
        elif 'APA' in char:
            char = 'APA'
            nopunc = nopunc + char + ' '
        elif 'EndNote' in char:
            char = 'EndNote'
            nopunc = nopunc + char + ' '
        elif 'NDDS' in char:
            char = 'NDDS'
            nopunc = nopunc + char + ' '
        elif 'JCR' in char:
            char = 'JCR'
            nopunc = nopunc + char + ' '
        elif 'The Economist' in char:
            char = 'The Economist'
            nopunc = nopunc + char + ' '
        elif 'Science Citation Index' in char:
            char = 'SCI'
            nopunc = nopunc + char + ' '

        elif '"' in char:
            char = char.replace('"', '').replace('＞', '').replace('-', '').replace('>', '')
            nopunc = nopunc + char + ' '

        elif '?' in char:
            pass
        elif '//' in char:
            pass
        elif '*' in char:
            pass
        elif '>' in char:
            pass
        elif '<' in char:
            pass
        elif ':' in char:
            pass
        elif '@' in char:
            pass
        elif '：' in char:
            pass
        elif '.' in char:
            pass
        elif char not in string.punctuation:
            nopunc = nopunc + char + ' '
    # 非停用字的 word，轉成小寫字母後就存進 list
    return ' '.join([word.lower() for word in nopunc.split() if word.lower() not in STOPWORDS])

def tfidf(data_list, max_df=0.8, min_df=2):
    tfidf_vectorizer = TfidfVectorizer(norm="l2", max_df=max_df, min_df=min_df)
    # max_df=0.7 表示若單詞在70%的文件裡都出現過，則視為高頻詞，對文件分類無幫助，會剔除這個詞。
    # min_df=2 表示若單詞出現次數過低，只出現2次以下(含)，對文件分類無幫助，會剔除這個詞。
    # max_features=500 進一步過濾辭典大小，會根據TF-IDF權重由高到低排序，取前20000個權重高的單詞構成辭典。

    X_tfidf = tfidf_vectorizer.fit_transform(data_list)
    print(X_tfidf.shape)

    tfidf_array = X_tfidf.toarray()
    # print(tfidf_array.shape)

    tfidf_T_array = X_tfidf.T.toarray()
    # print(tfidf_T_array.shape)

    terms = tfidf_vectorizer.get_feature_names()
    # 得到詞典單詞 (words)，根據索引即可得到每個類別裡權重最高的那些單詞了。
    # print(len(words))
    # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
    # https://stackoverflow.com/questions/54745482/what-is-the-difference-between-tfidf-vectorizer-and-tfidf-transformer

    return tfidf_vectorizer, X_tfidf, tfidf_array, tfidf_T_array, terms
