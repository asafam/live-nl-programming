"""
Enhanced re-scraper for Zapier template pages.

For each candidate URL, fetches the full page and extracts the actual
"How it works" section content (paragraphs and list items below the H2),
not just the heading text.

Only keeps records where ≥ 3 steps were found and total content > 150 chars.

Usage:
    python -m src.data.rescrape_zapier_templates \\
        --input outputs/zapier_filtered_candidates.jsonl \\
        --output outputs/zapier_enriched_candidates.jsonl \\
        --workers 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path
from queue import Queue

import requests
from tqdm import tqdm

USER_AGENT = "ZapierTemplateExtractor/1.0 (Educational/Research purposes)"
DEFAULT_WORKERS = 5
MIN_STEPS = 3
MIN_CONTENT_CHARS = 150


def fetch_url(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def _strip_tags(html: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", html).strip()


def _decode_entities(text: str) -> str:
    """Decode common HTML entities."""
    replacements = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
        "&#x27;": "'", "&#x2F;": "/",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    # Numeric entities
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    return text


def _extract_section_content(html: str, start: int) -> list[str]:
    """Extract paragraphs and list items from HTML starting at `start` until the next heading/section."""
    section_html = html[start:]
    boundary = re.search(
        r'(?:<h[23][^>]*>|</section>|<footer)',
        section_html,
        re.IGNORECASE,
    )
    section_html = section_html[:boundary.start()] if boundary else section_html[:5000]

    steps: list[str] = []

    # Extract list items first (most structured)
    li_matches = re.findall(r'<li[^>]*>(.*?)</li>', section_html, re.IGNORECASE | re.DOTALL)
    if li_matches:
        for li in li_matches:
            text = _decode_entities(_strip_tags(li))
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 15:
                steps.append(text)

    # Fall back to paragraphs
    if not steps:
        p_matches = re.findall(r'<p[^>]*>(.*?)</p>', section_html, re.IGNORECASE | re.DOTALL)
        for p in p_matches:
            text = _decode_entities(_strip_tags(p))
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 15:
                steps.append(text)

    return steps


def extract_how_it_works_steps(html: str) -> list[str]:
    """Extract step content from workflow-description sections.

    Tries multiple strategies in priority order:
    1. H2/H3 heading matching 'how.*works'
    2. H2/H3 heading matching 'overview' or 'how to'
    3. Any substantial paragraph content between headings
    """
    # Strategy 1: 'How ... works' heading (any level h2-h4)
    h_match = re.search(
        r'<h[2-4][^>]*>\s*how\s+.+?\s+works\s*</h[2-4]>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if h_match:
        steps = _extract_section_content(html, h_match.end())
        if steps:
            return steps

    # Strategy 2: 'How it works' or 'How to' heading
    h_match = re.search(
        r'<h[2-4][^>]*>\s*(?:how\s+it\s+works|how\s+to\s+(?:get\s+started|use|set\s+up))\s*</h[2-4]>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if h_match:
        steps = _extract_section_content(html, h_match.end())
        if steps:
            return steps

    # Strategy 3: 'Overview' section
    h_match = re.search(
        r'<h[2-4][^>]*>\s*overview\s*</h[2-4]>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if h_match:
        steps = _extract_section_content(html, h_match.end())
        if steps:
            return steps

    return []


def _extract_steps_legacy(html: str) -> list[str]:
    """Legacy extraction — H2-only 'how.*works' (kept for compatibility)."""
    h2_match = re.search(
        r'<h2[^>]*>\s*how\s+.+?\s+works\s*</h2>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if not h2_match:
        return []

    # Slice HTML from after the H2 to find content
    after_h2 = html[h2_match.end():]

    # Find the next structural boundary (next H2, H3, or closing section)
    boundary = re.search(
        r'(?:<h[23][^>]*>|</section>|<footer)',
        after_h2,
        re.IGNORECASE,
    )
    section_html = after_h2[:boundary.start()] if boundary else after_h2[:5000]

    steps: list[str] = []

    # Extract ordered/unordered list items first (most structured)
    li_matches = re.findall(r'<li[^>]*>(.*?)</li>', section_html, re.IGNORECASE | re.DOTALL)
    if li_matches:
        for li in li_matches:
            text = _decode_entities(_strip_tags(li))
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 15:  # Skip very short/empty items
                steps.append(text)

    # If no list items, fall back to paragraphs
    if not steps:
        p_matches = re.findall(r'<p[^>]*>(.*?)</p>', section_html, re.IGNORECASE | re.DOTALL)
        for p in p_matches:
            text = _decode_entities(_strip_tags(p))
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 15:
                steps.append(text)

    return steps


def is_sufficient(steps: list[str]) -> bool:
    """Return True if the extracted steps meet minimum quality thresholds."""
    if len(steps) < MIN_STEPS:
        return False
    total_chars = sum(len(s) for s in steps)
    return total_chars >= MIN_CONTENT_CHARS


class ReScraper:
    def __init__(self, input_path: Path, output_path: Path, num_workers: int, force: bool):
        self.input_path = input_path
        self.output_path = output_path
        self.num_workers = num_workers
        self.force = force

        self.queue: Queue[dict | None] = Queue(maxsize=num_workers * 4)
        self.file_lock = threading.Lock()
        self.seen_urls: set[str] = set()

        self.total = 0
        self.kept = 0
        self.pbar: tqdm | None = None
        self.output_file = None

    def _load_existing(self) -> None:
        """Load already-scraped URLs to support resuming."""
        if self.output_path.exists() and not self.force:
            with open(self.output_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if url := rec.get("url"):
                            self.seen_urls.add(url)
                            self.kept += 1
                    except json.JSONDecodeError:
                        pass

    def worker(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                self.queue.task_done()
                break

            url = record["url"]
            try:
                html = fetch_url(url)
                steps = extract_how_it_works_steps(html)

                if is_sufficient(steps):
                    enriched = {
                        "url": url,
                        "title": record.get("title", ""),
                        "description": record.get("description", ""),
                        "raw_steps": steps,
                    }
                    with self.file_lock:
                        self.output_file.write(json.dumps(enriched) + "\n")
                        self.output_file.flush()
                        self.kept += 1

            except requests.RequestException:
                pass  # Silently skip failed fetches
            finally:
                with self.file_lock:
                    self.total += 1
                    if self.pbar:
                        self.pbar.update(1)
                        self.pbar.set_postfix(kept=self.kept)
                self.queue.task_done()

    def run(self) -> None:
        self._load_existing()

        # Count input records for progress bar
        with open(self.input_path, encoding="utf-8") as f:
            num_input = sum(1 for _ in f)

        skipped_existing = len(self.seen_urls)
        if skipped_existing:
            print(f"Resuming: {skipped_existing} URLs already processed")

        file_mode = "w" if self.force else "a"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, file_mode, encoding="utf-8") as fout:
            self.output_file = fout
            self.pbar = tqdm(total=num_input - skipped_existing, desc="Re-scraping", unit="page")

            workers = []
            for _ in range(self.num_workers):
                t = threading.Thread(target=self.worker, daemon=True)
                t.start()
                workers.append(t)

            with open(self.input_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("url") in self.seen_urls:
                        continue
                    self.queue.put(record)

            # Signal workers to stop
            for _ in range(self.num_workers):
                self.queue.put(None)

            for t in workers:
                t.join()

            self.pbar.close()

        print(f"\nFetched {self.total:,} pages → {self.kept:,} kept ({100 * self.kept / max(self.total, 1):.1f}%)")
        print(f"Output: {self.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Zapier template re-scraper")
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force", action="store_true", help="Re-fetch all, ignoring existing output")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    scraper = ReScraper(args.input, args.output, args.workers, args.force)
    scraper.run()


if __name__ == "__main__":
    main()
