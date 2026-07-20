from __future__ import annotations

import json
import re
import sys
import urllib.request


url = sys.argv[1]
html = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
if not match:
    raise SystemExit("No __NEXT_DATA__ found")
data = json.loads(match.group(1))
doc = data["props"]["pageProps"]["document"]
print("keys", sorted(doc.keys()))
for key, value in doc.items():
    if isinstance(value, str) and ("http" in value or "pdf" in value.lower()):
        print("string", key, value[:500])
    elif isinstance(value, (dict, list)):
        print("complex", key, type(value).__name__)
print(json.dumps(doc, ensure_ascii=False, indent=2)[:5000])
