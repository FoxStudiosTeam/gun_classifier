import requests
'''
headers = {
    'Content-Type': 'multipart/form-data',
    #'x-api-key': 'xxxxxx-xxxxx-xxxx-xxxx-xxxxxxxx',
}

files = {
    'file': open('./e.jpg', 'rb'),
}
r = requests.post('http://46.242.121.246:25601', headers=headers, files={'file': open('./iz2.png', 'rb')})
print(r.json())
'''

r = requests.get('http://127.0.0.1:5000/video_feed')
print(r)



