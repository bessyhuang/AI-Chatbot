from datetime import datetime, timezone, timedelta

def judge_day(day):
    if day == 'Sunday':
        return 'Sunday 各館的開放時間如下：\n濟時樓：9:00 ~ 18:00\n公博樓：不開放\n國璽樓：8:00 ~ 23:00'
               
    elif day == 'Saturday':
        return 'Saturday 各館的開放時間如下：\n濟時樓：9:00 ~ 18:00\n公博樓：9:00 ~ 18:00\n國璽樓：8:00 ~ 23:00'
               
    else:
        return '平日 各館的開放時間如下：\n濟時樓：8:00 ~ 22:00\n公博樓：8:00 ~ 21:30\n國璽樓：8:00 ~ 23:00'
               
def OpeningHours_parser(query_ws, query_pos):
    Nd_words = []
    for ws, pos in zip(query_ws, query_pos):
        if pos == 'Nd':
            Nd_words.append(ws)
        elif ws == '疫情':
            return '因短期疫情持續升高，雙北已提升至三級警戒，依防疫規定，圖書館屬應關閉場所，本館緊急於 5/15 中午 12:30 起閉館。若開放時間及借還書相關服務有變動，請隨時留意圖書館網站公告，不便之處，敬請見諒。\nhttp://web.lib.fju.edu.tw/chi/news/20210517'
            
    #print('---', Nd_words)
    
    if Nd_words != [] :
        if '今天' in Nd_words:              
            mytime_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
            mytime_tw = mytime_utc.astimezone(timezone(timedelta(hours=8)))
            print(mytime_tw)    
            return judge_day(mytime_tw.strftime("%A"))
            
        elif '週一' in Nd_words or '平日' in Nd_words or '週二' in Nd_words or '週三' in Nd_words or '週四' in Nd_words or '週五' in Nd_words:
            return judge_day('Weekdays')
                                
        elif '週六' in Nd_words:
            return judge_day('Saturday')
    
        elif '週日' in Nd_words:
            return judge_day('Sunday')
    
        elif '假日' in Nd_words:
            final_res1 = judge_day('Saturday')
            final_res2 = judge_day('Sunday')
            return final_res1 + '\n------\n' + final_res2

        else:
            return '輔大圖書館各館的開放時間，詳情請見: http://web.lib.fju.edu.tw/chi/intro/opentime\n國定及校定假日特殊開放時間: http://web.lib.fju.edu.tw/chi/news/20200915'
    
    else:
        return '輔大圖書館各館的開放時間，詳情請見: http://web.lib.fju.edu.tw/chi/intro/opentime\n國定及校定假日特殊開放時間: http://web.lib.fju.edu.tw/chi/news/20200915'

