from urllib import request, parse
import urllib
import ssl
import sys
import base64
import json
import os

# host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=F2FUmuxf4MZe4lyADZiF9hTQ&client_secret=GtsCc6ucf0NaayH9Yu12xkms8ZdeArg0'
# req = request.Request(host)
# req.add_header('Content-Type', 'application/json; charset=UTF-8')
# response = request.urlopen(req)
# content = response.read()
# print(content)
# if content:
# print(content)
image_file_path = 'source_two/image12'
text_file_path = 'source_two/text12'
if not os.path.exists(text_file_path):
    os.makedirs(text_file_path)
images = os.listdir(image_file_path)
url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/webimage?access_token=24.af51a2ea2271ee69b4df58858224e3ba.2592000.1537683914.282335-11722929'
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
for image in images:
    image_path = os.path.join(image_file_path, image)
    f = open(image_path, 'rb')
    lsf = base64.b64encode(f.read())
    f.close()
    data = {'image': lsf}
    data = parse.urlencode(data).encode('utf-8')
    requ = request.Request(url, headers=headers, data=data)
    response = request.urlopen(requ)
    c = response.read()
    print(c.decode('utf-8'))
    words_results = json.loads(c.decode('utf-8')).get('words_result')

    if words_results:
        name = image.split('.')[0]
        with open(f'{text_file_path}/{name}.txt', 'w', encoding='utf-8') as w:
            for words_result in words_results:
                w.write(words_result.get('words') + '\n')
