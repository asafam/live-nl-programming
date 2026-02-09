"""
Zapier Template URL Extractor

Fetches Zapier's template sitemap XML and extracts all template detail URLs.
Uses a worker-based queue architecture for concurrent processing.

Usage:
    python extract_zapier_templates.py          # Resume from existing file
    python extract_zapier_templates.py --force  # Start fresh
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import Queue

import requests
from tqdm import tqdm

SITEMAP_URL = "https://zapier.com/sitemaps/templates/Xf3Lk9YbRzQnmgP2eCJtA5Wxv0NdMu6T"
TEMPLATE_URL_PREFIX = "https://zapier.com/templates/details/"
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "zapier_templates_raw.jsonl"
USER_AGENT = "ZapierTemplateExtractor/1.0 (Educational/Research purposes)"
SITEMAP_NAMESPACE = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
NUM_WORKERS = 5


def fetch_url(url: str) -> str:
    """Fetch URL content with proper headers."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def parse_loc_tags(xml_content: str) -> list[str]:
    """Parse XML and extract all <loc> tag contents."""
    root = ET.fromstring(xml_content)
    urls = []
    for loc in root.iter(f"{SITEMAP_NAMESPACE}loc"):
        if loc.text:
            urls.append(loc.text.strip())
    return urls


def extract_title(html: str) -> str | None:
    """Extract title from HTML page."""
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_description(html: str) -> str | None:
    """Extract meta description from HTML page."""
    match = re.search(
        r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']description["\']',
            html,
            re.IGNORECASE,
        )
    if match:
        return match.group(1).strip()
    return None


def extract_how_it_works(html: str) -> str | None:
    """Extract 'How ... works' h2 heading from HTML page."""
    match = re.search(
        r'<h2[^>]*>\s*(how\s+.+?\s+works)\s*</h2>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


class TemplateExtractor:
    """Manages worker-based extraction of Zapier templates."""

    def __init__(self, output_path: Path, num_workers: int = NUM_WORKERS, force: bool = False):
        self.output_path = output_path
        self.num_workers = num_workers
        self.force = force
        self.page_queue: Queue[tuple[int, str] | None] = Queue()
        self.seen_urls: set[str] = set()
        self.seen_titles: set[str] = set()
        self.seen_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.total_templates = 0
        self.pages_processed = 0
        self.total_pages = 0
        self.output_file = None
        self.pbar: tqdm | None = None

    def load_existing(self) -> int:
        """Load existing URLs and titles from output file for resume support."""
        if not self.output_path.exists() or self.force:
            return 0

        count = 0
        with open(self.output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if url := record.get("url"):
                        self.seen_urls.add(url)
                    if title := record.get("title"):
                        self.seen_titles.add(title)
                    count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def worker(self, worker_id: int) -> None:
        """Worker that processes sitemap pages from the queue."""
        while True:
            item = self.page_queue.get()
            if item is None:
                self.page_queue.task_done()
                break

            page_num, page_url = item
            try:
                page_xml = fetch_url(page_url)
                urls = parse_loc_tags(page_xml)
                template_urls = [u for u in urls if u.startswith(TEMPLATE_URL_PREFIX)]

                new_templates = []
                with self.seen_lock:
                    for url in template_urls:
                        if url not in self.seen_urls:
                            self.seen_urls.add(url)
                            new_templates.append(url)

                for url in new_templates:
                    try:
                        html = fetch_url(url)
                        title = extract_title(html) or ""
                        description = extract_description(html) or ""
                        how_it_works = extract_how_it_works(html) or ""

                        with self.seen_lock:
                            if title and title in self.seen_titles:
                                continue
                            if title:
                                self.seen_titles.add(title)

                        record = {
                            "url": url,
                            "title": title,
                            "description": description,
                            "how_it_works": how_it_works,
                        }
                        with self.file_lock:
                            self.output_file.write(json.dumps(record) + "\n")
                            self.output_file.flush()
                            self.total_templates += 1
                    except requests.RequestException:
                        with self.file_lock:
                            record = {
                                "url": url,
                                "title": None,
                                "description": None,
                                "how_it_works": None,
                                "error": "fetch_failed",
                            }
                            self.output_file.write(json.dumps(record) + "\n")
                            self.output_file.flush()

                with self.file_lock:
                    self.pages_processed += 1
                    if self.pbar:
                        self.pbar.update(1)
                        self.pbar.set_postfix(templates=self.total_templates)

            except (requests.RequestException, ET.ParseError):
                with self.file_lock:
                    self.pages_processed += 1
                    if self.pbar:
                        self.pbar.update(1)

            self.page_queue.task_done()

    def run(self) -> int:
        """Run the extraction process."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_count = self.load_existing()
        if existing_count > 0:
            print(f"Resuming: loaded {existing_count} existing templates ({len(self.seen_urls)} URLs, {len(self.seen_titles)} titles)")
            self.total_templates = existing_count
            file_mode = "a"
        else:
            file_mode = "w"

        print(f"Fetching sitemap index from {SITEMAP_URL}...")
        index_xml = fetch_url(SITEMAP_URL)
        page_urls = parse_loc_tags(index_xml)
        self.total_pages = len(page_urls)
        print(f"Found {self.total_pages} sitemap pages to process")
        print(f"Starting {self.num_workers} workers...")
        print(f"Writing to {self.output_path}...\n")

        with open(self.output_path, file_mode, encoding="utf-8") as f:
            self.output_file = f

            self.pbar = tqdm(total=self.total_pages, desc="Processing pages", unit="page")

            try:
                workers = []
                for i in range(self.num_workers):
                    t = threading.Thread(target=self.worker, args=(i,))
                    t.start()
                    workers.append(t)

                for i, page_url in enumerate(page_urls, 1):
                    self.page_queue.put((i, page_url))

                self.page_queue.join()

                for _ in range(self.num_workers):
                    self.page_queue.put(None)

                for t in workers:
                    t.join()
            finally:
                self.pbar.close()

        return self.total_templates


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract Zapier template URLs and metadata")
    parser.add_argument("--force", action="store_true", help="Start fresh, ignoring existing output file")
    args = parser.parse_args()

    try:
        extractor = TemplateExtractor(OUTPUT_FILE, force=args.force)
        total = extractor.run()
        print(f"\nTotal templates saved: {total}")

    except requests.RequestException as e:
        print(f"HTTP error while fetching sitemap index: {e}")
        raise SystemExit(1)
    except ET.ParseError as e:
        print(f"XML parsing error in sitemap index: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
