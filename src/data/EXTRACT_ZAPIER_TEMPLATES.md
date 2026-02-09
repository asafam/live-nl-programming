# Zapier Template URL Extractor

Extracts all Zapier workflow template URLs and titles from their XML sitemap.

## Usage

```bash
python src/data/extract_zapier_templates.py
```

## Architecture

Uses a worker-based queue system for concurrent processing:

1. Main thread fetches the sitemap index and enqueues all page URLs
2. Worker threads (default: 5) pull pages from the queue
3. Each worker fetches the sitemap page, extracts template URLs
4. For each template URL, the worker fetches the page and extracts the title
5. Results are written incrementally to JSONL as they complete

## Output

Creates `outputs/zapier_templates_raw.jsonl` with one JSON object per line:

```json
{"url": "https://zapier.com/templates/details/slack-meeting-reminders", "title": "Get Slack reminders for meetings - Zapier"}
{"url": "https://zapier.com/templates/details/trello-to-sheets", "title": "Add Trello cards to Google Sheets - Zapier"}
```

## Dependencies

- `requests` - for HTTP requests

## Notes

- Uses 5 concurrent workers by default (configurable via `NUM_WORKERS`)
- Uses a polite User-Agent header
- Handles HTTP errors gracefully, continues processing
- Deduplicates URLs across pages
- Thread-safe file writes
