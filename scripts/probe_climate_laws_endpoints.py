from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


def load_env():
    html = urllib.request.urlopen("https://climate-laws.org/geographies/india", timeout=30).read().decode(
        "utf-8", "ignore"
    )
    data = json.loads(
        re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html).group(1)
    )
    return data["props"]["pageProps"]["envConfig"]


env = load_env()
headers = {"app-token": env["BACKEND_API_TOKEN"], "Accept": "application/json"}
paths = [
    "/families/india-biennial-transparency-report-btr1_9756",
    "/documents/india-biennial-transparency-report-btr1_9756",
    "/document/india-biennial-transparency-report-btr1_9756",
    "/families/UNFCCC.family.i00007690.n0000",
]
for path in paths:
    url = env["BACKEND_API_URL"] + path
    req = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=20)
        print("OK", path, response.status, response.read(1000).decode("utf-8", "ignore"))
    except urllib.error.HTTPError as exc:
        print("HTTP", path, exc.code, exc.read(500).decode("utf-8", "ignore"))
    except Exception as exc:
        print("ERR", path, type(exc).__name__, exc)

payload = {
    "query_string": "",
    "exact_match": False,
    "keyword_filters": {
        "countries": ["india"],
        "regions": [],
        "subdivisions": [],
        "category": [],
    },
    "sort_field": "date",
    "sort_order": "desc",
    "page_size": 5,
    "offset": 0,
    "max_passages_per_doc": 0,
    "continuation_tokens": [],
    "corpus_import_ids": [],
    "metadata": [],
}
req = urllib.request.Request(
    env["BACKEND_API_URL"] + "/searches",
    data=json.dumps(payload).encode("utf-8"),
    headers={**headers, "Content-Type": "application/json"},
    method="POST",
)
try:
    response = urllib.request.urlopen(req, timeout=30)
    print("SEARCH", response.status, response.read(2000).decode("utf-8", "ignore"))
except urllib.error.HTTPError as exc:
    print("SEARCH_HTTP", exc.code, exc.read(2000).decode("utf-8", "ignore"))
