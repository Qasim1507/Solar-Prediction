import requests
from PIL import Image
from io import BytesIO

session = requests.Session()

# First visit the main page to get cookies
session.get("https://himawari8.nict.go.jp/", verify=False)

# Now try the image
url = "https://himawari8-dl.nict.go.jp/himawari8/img/D531106/4d/550/2026/03/31/055000_1_1.png"
r = session.get(url, verify=False, timeout=30, headers={
    "Referer": "https://himawari8.nict.go.jp/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    img = Image.open(BytesIO(r.content))
    print(f"Image size: {img.size}, mode: {img.mode}")