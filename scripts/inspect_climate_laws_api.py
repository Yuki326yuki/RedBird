from __future__ import annotations

import re
import urllib.request


html = urllib.request.urlopen("https://climate-laws.org/geographies/india", timeout=30).read().decode(
    "utf-8", "ignore"
)
paths = re.findall(r'src="([^"]+\.js)"', html)
print("script_count", len(paths))
for path in paths:
    url = "https://climate-laws.org" + path if path.startswith("/") else path
    js = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
    if any(term in js for term in ["vespaSearchResults", "BACKEND_API_URL", "/search", "continuation_token"]):
        print("MATCH", path, len(js))
        for term in ["vespaSearchResults", "BACKEND_API_URL", "continuation_token", "/search"]:
            idx = js.find(term)
            if idx >= 0:
                print("TERM", term, js[max(0, idx - 500) : idx + 1000])
