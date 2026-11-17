"""Download a small, license-documented classical MIDI subset from Mutopia."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_SOURCE = "https://www.mutopiaproject.org/cgibin/make-table.cgi?Instrument=Piano"


def download_subset(output_dir: str | Path, limit: int = 8, source_url: str = DEFAULT_SOURCE) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(source_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.lower().endswith((".mid", ".midi")):
            url = urljoin(response.url, href)
            if url not in {item["url"] for item in links}:
                links.append({"url": url, "title": anchor.get_text(" ", strip=True) or Path(href).stem})
    if not links:
        raise RuntimeError("No MIDI download links were discovered at the Mutopia source page")
    downloaded = []
    for index, item in enumerate(links[:limit], start=1):
        filename = Path(item["url"].split("?")[0]).name or f"mutopia_{index}.mid"
        if not filename.lower().endswith((".mid", ".midi")):
            filename = f"mutopia_{index}.mid"
        destination = output_dir / filename
        data = requests.get(item["url"], timeout=30)
        data.raise_for_status()
        destination.write_bytes(data.content)
        downloaded.append({"filename": filename, "title": item["title"], "source_url": item["url"], "license": "Mutopia piece-specific public-domain or Creative Commons license; verify the individual source page before redistribution."})
    (output_dir / "DATASET_SOURCE.json").write_text(json.dumps({"source": source_url, "license_summary": "Mutopia provides music in the public domain or under Creative Commons licenses; each piece lists its specific license.", "files": downloaded}, indent=2), encoding="utf-8")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/midi")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    for item in download_subset(**vars(args)):
        print(item["filename"], item["source_url"])


if __name__ == "__main__":
    main()
