import os
import requests
import zipfile
import io

os.makedirs("data", exist_ok=True)
url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
zip_path = "data/online_retail.zip"

print("Downloading dataset from UCI Machine Learning Repository...")
r = requests.get(url)
r.raise_for_status()

with open(zip_path, 'wb') as f:
    f.write(r.content)

print("Extracting dataset...")
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    z.extractall("data/")

print("Download complete! Files saved in ./data/")
