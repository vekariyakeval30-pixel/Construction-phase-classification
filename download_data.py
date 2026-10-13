"""Download the public CC BY 4.0 HZNU-BuildStage-4 v1 dataset with checksums."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = 'https://data.mendeley.com/public-api/datasets/wy8jnypms4'
HEADERS = {'Accept': 'application/vnd.mendeley-public-dataset.1+json', 'User-Agent': 'ConstructionPhaseStudy/1.0'}

def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def main():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    folders = json.loads(fetch(API + '/folders/1'))
    by_id = {f['id']: f for f in folders}
    records = []
    for folder in folders:
        if not folder['name'].startswith('stage_'):
            continue
        split = by_id[folder['parent_id']]['name']
        start = 0
        while True:
            files = json.loads(fetch(API + f"/files?folder_id={folder['id']}&version=1&$start={start}&$limit=1000"))
            for item in files:
                details = item['content_details']
                records.append(dict(path=f"{split}/{folder['name']}/{item['filename']}",
                    source_split=split, label=folder['name'], file_id=item['id'],
                    sha256=details['sha256_hash'], size=details['size'], url=details['download_url']))
            if len(files) < 1000:
                break
            start += 1000
    (data / 'source_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'{len(records)} images, {sum(r["size"] for r in records)/1e6:.1f} MB', flush=True)
    def download(record):
        path = data / record['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']:
            return
        blob = fetch(record['url'])
        if hashlib.sha256(blob).hexdigest() != record['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        path.write_bytes(blob)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(download, records), 1):
            if i % 100 == 0:
                print(f'Downloaded and verified {i}/{len(records)}', flush=True)
    print('Download complete. Source: https://doi.org/10.17632/wy8jnypms4.1', flush=True)

if __name__ == '__main__':
    main()
