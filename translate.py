import sys
import http.client
import hashlib
import json
import urllib
import random
from openpyxl import load_workbook
import time
from tqdm import tqdm


def read_file(l, f):
    appid = '20190913000334193'
    secretKey = 'ahJKXNOLr0653W5mF5V5'
    httpClient = None
    myurl = '/api/trans/vip/translate'
    while f:
        try:
            line = f.pop(-1)
            fromLang = 'en'  # 源语言
            toLang = 'zh'  # 翻译后的语言
            # if i % 30 == 0:
            #     time.sleep(10)
            index = line.split("\t")[0]
            text = line.split("\t")[1].replace(" [E21] ", " ").replace(
                " [E12] ", " ").replace(" [E22]", " ").replace("[E11] ", " ").replace("  ", " ")
            #print(index + "\t"+line, end="\t")
            salt = random.randint(32768, 65536)
            sign = appid + text + str(salt) + secretKey
            sign = hashlib.md5(sign.encode()).hexdigest()
            myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
                text) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
                salt) + '&sign=' + sign

            httpClient = http.client.HTTPConnection(
                'api.fanyi.baidu.com')
            httpClient.request('GET', myurl)  # response是HTTPResponse对象
            response = httpClient.getresponse()
            jsonResponse = response.read().decode("utf-8")  # 获得返回的结果，结果为json格式
            js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
            # print(js)
            dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
            print(index + "\t" + text + "\t"+dst)   # 打印结果
            l.append(index + "\t" + text + "\t"+dst)
        except:
            break


if __name__ == "__main__":
    f = open("dataset/train.tsv").readlines()[::-1]
    l = []
    while f:
        read_file(l, f)
    # except:
    #     salt = random.randint(32768, 65536)
    #     httpClient.close()
    #     f.append(line)
    #     time.sleep(5)
