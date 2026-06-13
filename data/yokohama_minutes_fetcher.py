#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


UA = "YokohamaMinutesFetcher/2.0-playwright-rendered-download"


@dataclass
class MinuteRecord:
    source_url: str
    council_id: Optional[str]
    schedule_id: Optional[str]
    minute_id: Optional[str]
    status: str
    page_title: Optional[str] = None
    council_title: Optional[str] = None
    session_title: Optional[str] = None
    meeting_date_text: Optional[str] = None
    html_chars: int = 0
    text_chars: int = 0
    local_html_path: Optional[str] = None
    local_text_path: Optional[str] = None
    notes: list[str] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def safe_name(s: str) -> str:
    s = norm(s)
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", s)
    return s.strip("_") or "file"


def parse_ids(url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    qs = parse_qs(urlparse(url).query)
    return (
        qs.get("council_id", [None])[0],
        qs.get("schedule_id", [None])[0],
        qs.get("minute_id", [None])[0],
    )


def extract_session_title(council_title: str, text: str) -> Optional[str]:
    src = f"{council_title}\n{text[:5000]}"

    m = re.search(r"(令和\s*[元０-９0-9]+\s*年第\s*[０-９0-9]+\s*回\s*(?:定例会|臨時会))", src)
    if m:
        return re.sub(r"\s+", "", m.group(1))

    m = re.search(r"(平成\s*[元０-９0-9]+\s*年第\s*[０-９0-9]+\s*回\s*(?:定例会|臨時会))", src)
    if m:
        return re.sub(r"\s+", "", m.group(1))

    return None


def extract_meeting_date(council_title: str, text: str) -> Optional[str]:
    src = f"{council_title}\n{text[:5000]}"

    m = re.search(r"([０-９0-9]{1,2}月[０-９0-9]{1,2}日)[－ー―\-]?[０-９0-9]*号?", src)
    if m:
        return m.group(1)

    m = re.search(r"(令和\s*[元０-９0-9]+年[０-９0-9]{1,2}月[０-９0-9]{1,2}日)", src)
    if m:
        return norm(m.group(1))

    m = re.search(r"(平成\s*[元０-９0-9]+年[０-９0-9]{1,2}月[０-９0-9]{1,2}日)", src)
    if m:
        return norm(m.group(1))

    return None


def load_urls(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    urls = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            line = line.split(",")[-1].strip()
        urls.append(line)

    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def rendered_text_from_html(html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")

    page_title = norm(soup.title.get_text(" ")) if soup.title else ""

    council_el = soup.find(id="council-title")
    council_title = norm(council_el.get_text(" ")) if council_el else ""

    plain = soup.find(id="plain-minute")
    if plain:
        text = plain.get_text("\n")
    else:
        text = soup.get_text("\n")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return page_title, council_title, text


def fetch_one(page, url: str, out_dir: Path, wait_ms: int) -> MinuteRecord:
    council_id, schedule_id, minute_id = parse_ids(url)

    rec = MinuteRecord(
        source_url=url,
        council_id=council_id,
        schedule_id=schedule_id,
        minute_id=minute_id,
        status="fetch_failed",
    )

    try:
        page.goto(url, wait_until="networkidle", timeout=45000)

        # Wait for rendered transcript area if it appears.
        try:
            page.wait_for_selector("#plain-minute pre, #plain-minute, #council-title", timeout=15000)
        except Exception:
            pass

        page.wait_for_timeout(wait_ms)

        html = page.content()
        page_title, council_title, text = rendered_text_from_html(html)

        rec.page_title = page_title
        rec.council_title = council_title
        rec.session_title = extract_session_title(council_title, text)
        rec.meeting_date_text = extract_meeting_date(council_title, text)
        rec.html_chars = len(html)
        rec.text_chars = len(text)

        minutes_dir = out_dir / "minutes"
        minutes_dir.mkdir(parents=True, exist_ok=True)

        name = safe_name(
            f"{rec.session_title or 'unknown'}_{rec.meeting_date_text or 'dateNA'}_"
            f"c{council_id or 'NA'}_s{schedule_id or 'NA'}_m{minute_id or 'NA'}"
        )

        html_path = minutes_dir / f"{name}.html"
        text_path = minutes_dir / f"{name}.txt"

        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(text, encoding="utf-8")

        rec.local_html_path = str(html_path.relative_to(out_dir))
        rec.local_text_path = str(text_path.relative_to(out_dir))

        if "会議録表示" in page_title or "会議録" in council_title or "会議録" in text[:2000]:
            rec.status = "downloaded"
        else:
            rec.status = "downloaded_unverified"
            rec.notes.append("Downloaded rendered page, but did not clearly detect 会議録.")

        return rec

    except Exception as e:
        rec.notes.append(str(e))
        return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("yokohama_gikai_db"))
    ap.add_argument("--urls", type=Path, default=Path("minute_urls.txt"))
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--wait-ms", type=int, default=2000)
    args = ap.parse_args()

    args.db.mkdir(parents=True, exist_ok=True)

    urls = load_urls(args.urls)
    if not urls:
        raise RuntimeError(f"No URLs found in {args.urls}")

    records = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(user_agent=UA)
        page = context.new_page()

        for i, url in enumerate(urls, 1):
            rec = fetch_one(page, url, args.db, args.wait_ms)
            records.append(rec)

            print(
                f"[{i}/{len(urls)}] {rec.status} "
                f"council_id={rec.council_id} schedule_id={rec.schedule_id} "
                f"{rec.session_title or ''} {rec.meeting_date_text or ''} "
                f"html_chars={rec.html_chars} text_chars={rec.text_chars}"
            )

        browser.close()

    out_json = args.db / "yokohama_gikai_minutes.json"
    out_json.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "ran_at": now_iso(),
        "command": " ".join([sys.executable] + sys.argv),
        "script": "yokohama_minutes_fetcher.py",
        "script_version": UA,
        "urls_total": len(urls),
        "downloaded_total": sum(1 for r in records if r.status.startswith("downloaded")),
        "output": str(out_json),
    }

    with (args.db / "run_metadata.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"\nWrote: {out_json}")
    print(f"Downloaded: {meta['downloaded_total']} / {meta['urls_total']}")


if __name__ == "__main__":
    main()