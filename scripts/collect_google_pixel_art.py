import json, os, re, time, hashlib, mimetypes
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

API = 'https://www.googleapis.com/customsearch/v1'
OUT = Path('nano arcked dat/google_images')
IMG = OUT / 'images'
IMG.mkdir(parents=True, exist_ok=True)
API_KEY = os.environ['GOOGLE_API_KEY']
CSE_ID = os.environ['GOOGLE_CSE_ID']
PER_QUERY = max(1, min(10, int(os.getenv('RESULTS_PER_QUERY', '10'))))
MAX_DOWNLOADS = max(1, int(os.getenv('MAX_DOWNLOADS', '100')))

# Broad, varied pixel-art search terms. This collects search results, not only assets
# advertised as free. License status is explicitly marked for later review.
QUERIES = [
 '32x32 pixel art warrior sprite full body', '64x64 pixel art knight character sprite',
 '64x64 pixel art mage wizard sprite full body', 'pixel art archer rogue assassin sprite sheet',
 '16 bit pixel art fantasy hero standing sprite', 'pixel art paladin berserker samurai ninja sprite',
 'pixel art pirate viking gladiator soldier sprite', 'pixel art fantasy merchant blacksmith civilian NPC',
 'pixel art dragon monster demon undead skeleton zombie sprite', 'pixel art goblin orc troll golem elemental creature sprite',
 'pixel art elf dwarf fairy angel vampire ghost sprite', 'pixel art animal beast wolf bear cat sprite',
 'pixel art robot mecha cyborg android alien sprite', 'pixel art cyberpunk soldier sci fi character sprite',
 'pixel art spaceship vehicle arcade sprite', 'pixel art sword weapon shield armor helmet item',
 'pixel art potion chest key door portal game asset', 'pixel art castle building tree plant rock crystal',
 'retro 8 bit arcade character sprite transparent background', 'retro 16 bit RPG full body character sprite',
 '128x128 detailed pixel art character sprite', 'pixel art idle walking running attacking animation frame',
 'pixel art casting magic fire ice lightning character sprite', 'pixel art front view side view back view character sprite',
 'pixel art horror monster creature sprite', 'pixel art steampunk character robot sprite',
 'pixel art Egyptian mythology inspired fantasy game character sprite', 'pixel art medieval fantasy NPC sprite'
]
# Skip results whose titles/snippets explicitly indicate these subjects. This text filter
# is a precaution, not a guarantee of visual-content screening.
EXCLUDE = re.compile(r'\b(allah|prophet|muhammad|mohammed|jesus|moses|abraham|noah|abu bakr|umar ibn|uthman|uthmaan|ali ibn abi|quran as character|living quran)\b', re.I)
IMAGE_EXTS = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp'}
manifest = OUT / 'manifest.jsonl'
seen_urls, seen_hashes = set(), set()
if manifest.exists():
    for line in manifest.read_text('utf-8', errors='ignore').splitlines():
        try:
            item = json.loads(line)
            if item.get('image_url'): seen_urls.add(item['image_url'])
            if item.get('sha256'): seen_hashes.add(item['sha256'])
        except Exception: pass


def api_search(query):
    from urllib.parse import urlencode
    params = urlencode({'key': API_KEY, 'cx': CSE_ID, 'q': query, 'searchType': 'image', 'num': PER_QUERY, 'safe': 'active'})
    req = Request(API + '?' + params, headers={'User-Agent': 'pixel-art-dataset-collector/1.0'})
    with urlopen(req, timeout=25) as res:
        return json.loads(res.read().decode('utf-8'))


def ext_for(url, mime):
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in IMAGE_EXTS: return ext
    return { 'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp', 'image/bmp': '.bmp' }.get(mime, '')

count = 0
with manifest.open('a', encoding='utf-8') as mf:
    for query in QUERIES:
        if count >= MAX_DOWNLOADS: break
        print(f'Query: {query}', flush=True)
        try:
            payload = api_search(query)
        except Exception as e:
            print(f'API search failed for query: {e}', flush=True)
            continue
        for result in payload.get('items', []):
            if count >= MAX_DOWNLOADS: break
            image = result.get('image') or {}
            url = result.get('link', '')
            title = result.get('title', '')
            context = image.get('contextLink', '')
            if not url or url in seen_urls or EXCLUDE.search(title + ' ' + context): continue
            mime = (result.get('mime') or '').split(';')[0].lower()
            ext = ext_for(url, mime)
            if not ext: continue
            try:
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; PixelArtDatasetCollector/1.0)'})
                with urlopen(req, timeout=20) as res:
                    content_type = (res.headers.get('Content-Type') or '').split(';')[0].lower()
                    data = res.read(12 * 1024 * 1024 + 1)
                if len(data) > 12 * 1024 * 1024 or len(data) < 16: continue
                if content_type not in IMAGE_EXTS.values() and not data.startswith((b'\x89PNG', b'\xff\xd8\xff', b'GIF8', b'RIFF')): continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen_hashes: continue
                # Basic signature checks prevent HTML error pages saved as images.
                if not (data.startswith((b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff', b'GIF87a', b'GIF89a', b'RIFF'))): continue
                filename = f'google_{digest[:16]}{ext}'
                (IMG / filename).write_bytes(data)
                row = {
                    'file': f'images/{filename}', 'sha256': digest, 'query': query,
                    'title': title, 'image_url': url, 'source_page': context,
                    'thumbnail_url': image.get('thumbnailLink'),
                    'width_reported': image.get('width'), 'height_reported': image.get('height'),
                    'mime_reported': mime, 'download_content_type': content_type,
                    'license_status': 'unknown_pending_review',
                    'source': 'Google Programmable Search image results',
                    'note': 'Search result does not establish reuse or AI-training rights; review source and license.'
                }
                mf.write(json.dumps(row, ensure_ascii=False) + '\n'); mf.flush()
                seen_urls.add(url); seen_hashes.add(digest); count += 1
                print(f'Downloaded {count}/{MAX_DOWNLOADS}: {filename}', flush=True)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as e:
                print(f'Skipped inaccessible image: {e}', flush=True)
        time.sleep(0.2)

summary = {'new_downloads_this_run': count, 'total_manifest_records': sum(1 for _ in manifest.open(encoding='utf-8')), 'query_count': len(QUERIES), 'license_status': 'unknown_pending_review'}
(OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)
