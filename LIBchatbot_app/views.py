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

line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)
 
 
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
                msg_list = []

                # Wiki
                Keyword = event.message.text
                url = 'https://zh.wikipedia.org/wiki/' + Keyword #'https://www.google.com/search?q={}'.format(Keyword)
                response = requests.get(url)

                wiki_msg = [Keyword, response.url]
                msg_list.append(wiki_msg)
                # line_bot_api.reply_message(  # 回復傳入的訊息文字
                #     event.reply_token,
                #     TextSendMessage(text=wiki_msg)
                # )

                # Google
                for url in search(Keyword, stop=3):
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.find('title').text
                    msg = [title, url]
                    msg_list.append(msg)
                print(msg_list)
                line_bot_api.reply_message(
                	event.reply_token, [
                		TextSendMessage(text=url) for title, url in msg_list
                	])
        return HttpResponse()
    else:
        return HttpResponseBadRequest()