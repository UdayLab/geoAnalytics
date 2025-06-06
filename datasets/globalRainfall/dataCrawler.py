import requests
from bs4 import BeautifulSoup
import os

base_url = 'https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/2021/'

os.makedirs('files', exist_ok=True)

response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

for link in soup.find_all('a', href=True):
    href = link['href']
    if href.endswith('.nc'):
        file_url = base_url + href
        print(f'Downloading {file_url}')
        r = requests.get(file_url)
        with open(os.path.join('files', href), 'wb') as f:
            f.write(r.content)
