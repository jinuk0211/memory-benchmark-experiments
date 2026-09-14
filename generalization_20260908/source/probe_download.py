import html, json, re, time
from urllib.parse import urljoin, urlparse
import requests

s = requests.Session()
files = s.get('https://pypi.org/pypi/torch/2.8.0/json',timeout=20).json()['urls']
file = next(f for f in files if 'cp312-cp312-manylinux_2_28_x86_64' in f['filename'])
page = s.get('https://pypi.tuna.tsinghua.edu.cn/simple/torch/',timeout=20)
links = re.findall('href="([^"]+)"',page.text)
mirror = next(urljoin(page.url,html.unescape(x)) for x in links if file['filename'] in x)
for url in [file['url'],mirror]:
    start=time.monotonic(); count=0
    with s.get(url,headers={'Range':'bytes=0-2097151'},stream=True,timeout=30) as r:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            count += len(chunk)
            if count >= 2097152: break
    print(json.dumps({'host':urlparse(url).hostname,'MB_per_sec':count/(time.monotonic()-start)/1e6,'seconds':time.monotonic()-start}),flush=True)
