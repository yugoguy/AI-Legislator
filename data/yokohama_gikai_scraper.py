#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

INDEX_URL = "https://www.city.yokohama.lg.jp/shikai/kiroku/kekka/gian.html"
UA = "YokohamaGikaiScraper/0.6-stable-bills-only"

ERA_BASE = {"令和": 2018, "平成": 1988}

PDF_META_RE = re.compile(
    r"[（(]\s*PDF\s*[：:]\s*[0-9０-９,，.]+(?:KB|MB|ＫＢ|ＭＢ|kb|mb)\s*[）)]"
)
DATE_RE = re.compile(r"(\d{1,2})月(\d{1,2})日提出")
SESSION_RE = re.compile(r"((令和|平成)(元|\d+|[０-９]+)年.*?(?:定例会|臨時会))")

SUBMITTER_GROUPS = {"市長提出議案", "議員提出議案", "請願", "陳情", "諮問", "報告"}
SKIP_HEADINGS = {"議決結果", "目次", "本文ここまで", "現在位置"}


@dataclass
class FileRef:
    url: str
    local_path: Optional[str] = None
    text: Optional[str] = None
    kind: str = "pdf"


@dataclass
class BillRecord:
    record_id: str
    source: str
    session_title: str
    session_year: int
    session_url: str
    submitted_date_text: Optional[str] = None
    submitter_group: Optional[str] = None
    item_type: Optional[str] = None
    number_label: Optional[str] = None
    number: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    result: Optional[str] = None
    files: list[FileRef] = field(default_factory=list)

    meeting_records_status: str = "not_collected"
    meeting_records: list[dict] = field(default_factory=list)
    meeting_record_match_status: str = "not_collected"
    meeting_record_matches: list[dict] = field(default_factory=list)

    extraction_notes: list[str] = field(default_factory=list)


def normalize_text(s: str) -> str:
    s = s or ""
    s = PDF_META_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="replace")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def jpnum_to_int(s: str) -> int:
    s = s.strip()
    if s == "元":
        return 1
    return int(s.translate(str.maketrans("０１２３４５６７８９", "0123456789")))


def session_year_from_title(title: str) -> Optional[int]:
    m = re.search(r"(令和|平成)(元|\d+|[０-９]+)年", title or "")
    if not m:
        return None
    return ERA_BASE[m.group(1)] + jpnum_to_int(m.group(2))


def year_from_session_href(href: str) -> Optional[int]:
    m = re.search(r"gian([RH])(\d{1,2})", href or "")
    if not m:
        return None
    era_code, n = m.group(1), int(m.group(2))
    return 2018 + n if era_code == "R" else 1988 + n


def get(url: str, session: requests.Session, sleep: float = 0.4) -> requests.Response:
    time.sleep(sleep)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r


def sanitize_filename(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s)
    return s.strip("_") or "file"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_session_links(index_html: str, years_back: int) -> list[dict]:
    soup = BeautifulSoup(index_html, "html.parser")
    links: list[dict] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_url = urljoin(INDEX_URL, href)
        text = normalize_text(a.get_text(" "))
        path = urlparse(abs_url).path

        if "gian" not in path or not path.endswith(".html") or path.endswith("/gian.html"):
            continue

        year = session_year_from_title(text) or year_from_session_href(path)
        if year is None:
            continue

        title_match = SESSION_RE.search(text)
        title = title_match.group(1) if title_match else text or Path(path).stem

        links.append({"title": title, "url": abs_url, "year": year})

    seen = set()
    out = []
    for x in links:
        if x["url"] not in seen:
            seen.add(x["url"])
            out.append(x)

    if not out:
        sample = normalize_text(soup.get_text(" "))[:800]
        raise RuntimeError(f"No session links found. Text sample: {sample!r}")

    latest = max(x["year"] for x in out)
    min_year = latest - years_back + 1
    return [x for x in out if x["year"] >= min_year]


def is_heading(tag: Tag) -> bool:
    return tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}


def heading_level(tag: Tag) -> int:
    return int(tag.name[1]) if is_heading(tag) else 0


def direct_pdf_links(tag: Tag, base_url: str) -> list[FileRef]:
    out = []
    for a in tag.find_all("a", href=True):
        url = urljoin(base_url, a["href"])
        txt = normalize_text(a.get_text(" "))
        path = urlparse(url).path.lower()
        if path.endswith(".pdf") or "pdf" in txt.lower() or "PDF" in txt:
            out.append(FileRef(url=url, text=txt, kind="pdf"))
    return out


def make_record_id(r: BillRecord, idx: int) -> str:
    num = r.number or r.title or f"item_{idx:04d}"
    safe = re.sub(r"[^0-9A-Za-z一-龥ぁ-んァ-ンー第号市報請願諮問年度]+", "_", num)
    safe = safe.strip("_") or f"item_{idx:04d}"
    return f"{sanitize_filename(r.session_title)}_{safe}"


def table_rows(table: Tag) -> list[tuple[list[str], Tag]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [normalize_text(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append((cells, tr))
    return rows


def parse_table_records(
    table: Tag,
    session_title: str,
    session_year: int,
    session_url: str,
    submitted_date: Optional[str],
    submitter_group: Optional[str],
    item_type: Optional[str],
    start_idx: int,
) -> list[BillRecord]:
    rows = table_rows(table)
    if len(rows) < 2:
        return []

    header = rows[0][0]

    def find_col(names: tuple[str, ...]) -> Optional[int]:
        for i, h in enumerate(header):
            for n in names:
                if n in h:
                    return i
        return None

    idx_number = find_col(("議案番号", "請願番号", "番号"))
    idx_title = find_col(("議案名", "件名", "請願名", "名称"))
    idx_summary = find_col(("内容", "概要", "要旨"))
    idx_result = find_col(("結果", "議決結果", "審査結果"))

    if idx_number is None and idx_title is None:
        return []

    records: list[BillRecord] = []

    for row, tr in rows[1:]:
        if len(row) < 2:
            continue

        number = row[idx_number] if idx_number is not None and idx_number < len(row) else None
        title = row[idx_title] if idx_title is not None and idx_title < len(row) else None
        summary = row[idx_summary] if idx_summary is not None and idx_summary < len(row) else None
        result = row[idx_result] if idx_result is not None and idx_result < len(row) else None

        if not title and not number:
            continue

        rec = BillRecord(
            record_id="",
            source="横浜市会 議案及び審議結果",
            session_title=session_title,
            session_year=session_year,
            session_url=session_url,
            submitted_date_text=submitted_date,
            submitter_group=submitter_group,
            item_type=item_type,
            number_label="番号" if number else None,
            number=number,
            title=title,
            summary=summary,
            result=result,
            files=direct_pdf_links(tr, session_url),
            extraction_notes=["parsed_from_table"],
        )
        rec.record_id = make_record_id(rec, start_idx + len(records) + 1)
        records.append(rec)

    return records


def parse_session_page(html: str, session_title: str, session_year: int, session_url: str) -> list[BillRecord]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find(id="tmp_contents") or soup.body or soup

    records: list[BillRecord] = []
    submitted_date: Optional[str] = None
    submitter_group: Optional[str] = None
    item_type: Optional[str] = None
    current: Optional[BillRecord] = None

    def flush_current() -> None:
        nonlocal current
        if current and (current.title or current.number or current.summary):
            if not current.record_id:
                current.record_id = make_record_id(current, len(records) + 1)
            records.append(current)
        current = None

    elements = main.find_all(["h2", "h3", "h4", "h5", "h6", "p", "li", "div", "table"])

    for el in elements:
        if el.name == "table":
            parsed_table = parse_table_records(
                el,
                session_title,
                session_year,
                session_url,
                submitted_date,
                submitter_group,
                item_type,
                len(records),
            )
            if parsed_table:
                flush_current()
                records.extend(parsed_table)
            continue

        txt = normalize_text(el.get_text(" "))
        if not txt:
            continue

        if DATE_RE.search(txt) and len(txt) <= 30:
            flush_current()
            submitted_date = txt
            continue

        if txt in SUBMITTER_GROUPS:
            flush_current()
            submitter_group = txt
            if txt in {"請願", "陳情", "諮問", "報告"}:
                item_type = txt
            continue

        if is_heading(el):
            level = heading_level(el)
            if level <= 3 and txt not in SKIP_HEADINGS and not DATE_RE.search(txt):
                if txt not in SUBMITTER_GROUPS and not txt.startswith(("議案名", "議案番号", "請願番号", "内容", "結果")):
                    flush_current()
                    item_type = txt
                    continue

        if txt.startswith(("議案名：", "請願名：", "件名：")):
            flush_current()
            _, title = txt.split("：", 1)
            current = BillRecord(
                record_id="",
                source="横浜市会 議案及び審議結果",
                session_title=session_title,
                session_year=session_year,
                session_url=session_url,
                submitted_date_text=submitted_date,
                submitter_group=submitter_group,
                item_type=item_type,
                title=title.strip(),
                files=direct_pdf_links(el, session_url),
            )
            continue

        if current is None:
            continue

        if txt.startswith(("議案番号：", "請願番号：")):
            label, value = txt.split("：", 1)
            current.number_label = label
            current.number = value.strip()
            current.files.extend(direct_pdf_links(el, session_url))
            continue

        if txt.startswith("内容："):
            current.summary = txt.split("：", 1)[1].strip()
            current.files.extend(direct_pdf_links(el, session_url))
            continue

        if txt.startswith(("結果：", "議決結果：")):
            current.result = txt.split("：", 1)[1].strip()
            current.files.extend(direct_pdf_links(el, session_url))
            continue

        current.files.extend(direct_pdf_links(el, session_url))

    flush_current()

    for i, r in enumerate(records, 1):
        if not r.record_id:
            r.record_id = make_record_id(r, i)

        seen = set()
        uniq = []
        for f in r.files:
            if f.url and f.url not in seen:
                seen.add(f.url)
                f.text = normalize_text(f.text or "")
                uniq.append(f)
        r.files = uniq

    return records


def download_pdfs(records: list[BillRecord], out_dir: Path, session: requests.Session) -> None:
    pdf_root = out_dir / "pdfs"
    pdf_root.mkdir(parents=True, exist_ok=True)

    for r in records:
        session_dir = pdf_root / sanitize_filename(r.session_title)
        session_dir.mkdir(parents=True, exist_ok=True)

        for f in r.files:
            if not urlparse(f.url).path.lower().endswith(".pdf"):
                continue

            base = r.number or r.title or Path(urlparse(f.url).path).name or "file"
            name = sanitize_filename(base[:100])
            if not name.lower().endswith(".pdf"):
                name += ".pdf"

            path = session_dir / name

            if path.exists() and path.stat().st_size > 0:
                f.local_path = str(path.relative_to(out_dir))
                continue

            try:
                resp = get(f.url, session, sleep=0.2)
                path.write_bytes(resp.content)
                f.local_path = str(path.relative_to(out_dir))
            except Exception as e:
                f.local_path = None
                f.text = ((f.text or "") + f" DOWNLOAD_ERROR={e}").strip()


def bill_from_dict(d: dict) -> BillRecord:
    return BillRecord(
        record_id=d.get("record_id", ""),
        source=d.get("source", "横浜市会 議案及び審議結果"),
        session_title=d.get("session_title", ""),
        session_year=int(d.get("session_year", 0) or 0),
        session_url=d.get("session_url", ""),
        submitted_date_text=d.get("submitted_date_text"),
        submitter_group=d.get("submitter_group"),
        item_type=d.get("item_type"),
        number_label=d.get("number_label"),
        number=d.get("number"),
        title=d.get("title"),
        summary=d.get("summary"),
        result=d.get("result"),
        files=[FileRef(**x) for x in d.get("files", [])],
        meeting_records_status=d.get("meeting_records_status", "not_collected"),
        meeting_records=d.get("meeting_records", []),
        meeting_record_match_status=d.get("meeting_record_match_status", "not_collected"),
        meeting_record_matches=d.get("meeting_record_matches", []),
        extraction_notes=d.get("extraction_notes", []),
    )


def load_existing_records(out_dir: Path) -> list[BillRecord]:
    return [bill_from_dict(x) for x in load_json(out_dir / "yokohama_gikai_bills.json", [])]


def latest_source_by_url(sources: list[dict]) -> dict[str, dict]:
    return {s["url"]: s for s in sources if s.get("kind") == "session" and s.get("url")}


def group_records_by_session_url(records: list[BillRecord]) -> dict[str, list[BillRecord]]:
    out: dict[str, list[BillRecord]] = {}
    for r in records:
        out.setdefault(r.session_url, []).append(r)
    return out


def write_outputs(records: list[BillRecord], sources: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "yokohama_gikai_bills.json", [asdict(r) for r in records])
    write_json(out_dir / "sources.json", sources)

    cols = [
        "record_id",
        "session_title",
        "session_year",
        "submitted_date_text",
        "submitter_group",
        "item_type",
        "number_label",
        "number",
        "title",
        "summary",
        "result",
        "session_url",
        "meeting_records_status",
        "meeting_record_match_status",
        "file_urls",
        "local_pdf_paths",
        "extraction_notes",
    ]

    with (out_dir / "yokohama_gikai_bills.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()

        for r in records:
            row = asdict(r)
            row["file_urls"] = ";".join(f.url for f in r.files)
            row["local_pdf_paths"] = ";".join(f.local_path or "" for f in r.files)
            row["extraction_notes"] = ";".join(r.extraction_notes)
            row = {k: row.get(k, "") for k in cols}
            w.writerow(row)


def append_run_metadata(out_dir: Path, metadata: dict) -> None:
    with (out_dir / "run_metadata.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years-back", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("yokohama_gikai_db"))
    ap.add_argument("--download-pdfs", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--reparse-all", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})

    existing_records = load_existing_records(args.out)
    existing_by_session = group_records_by_session_url(existing_records)
    existing_sources = load_json(args.out / "sources.json", [])
    old_source_by_url = latest_source_by_url(existing_sources)

    index_resp = get(INDEX_URL, sess, sleep=0)
    index_hash = sha256_text(index_resp.text)
    session_links = collect_session_links(index_resp.text, args.years_back)

    print(f"Found {len(session_links)} session pages for latest {args.years_back} year(s).")

    new_records_by_session: dict[str, list[BillRecord]] = {}
    new_sources = [{"kind": "index", "url": INDEX_URL, "status": index_resp.status_code, "content_hash": index_hash}]

    stats = {
        "sessions_targeted": len(session_links),
        "sessions_skipped_same": 0,
        "sessions_parsed_new": 0,
        "sessions_parsed_changed": 0,
        "sessions_failed": 0,
    }

    for s in session_links:
        try:
            resp = get(s["url"], sess, sleep=args.sleep)
            content_hash = sha256_text(resp.text)
            old = old_source_by_url.get(s["url"])
            old_hash = old.get("content_hash") if old else None

            if (not args.reparse_all) and old_hash == content_hash and s["url"] in existing_by_session:
                kept = existing_by_session[s["url"]]
                new_records_by_session[s["url"]] = kept
                stats["sessions_skipped_same"] += 1
                new_sources.append({
                    "kind": "session",
                    "title": s["title"],
                    "year": s["year"],
                    "url": s["url"],
                    "status": resp.status_code,
                    "content_hash": content_hash,
                    "records": len(kept),
                    "parse_status": "skipped_same_hash",
                })
                print(f"SKIP {s['title']}: same hash, kept {len(kept)} records")
                continue

            parsed = parse_session_page(resp.text, s["title"], s["year"], s["url"])
            new_records_by_session[s["url"]] = parsed

            if old_hash is None:
                stats["sessions_parsed_new"] += 1
                parse_status = "parsed_new"
            else:
                stats["sessions_parsed_changed"] += 1
                parse_status = "parsed_changed_or_forced"

            new_sources.append({
                "kind": "session",
                "title": s["title"],
                "year": s["year"],
                "url": s["url"],
                "status": resp.status_code,
                "content_hash": content_hash,
                "old_content_hash": old_hash,
                "records": len(parsed),
                "parse_status": parse_status,
            })
            print(f"OK {s['title']}: {len(parsed)} records ({parse_status})")

        except Exception as e:
            stats["sessions_failed"] += 1
            new_sources.append({"kind": "session", **s, "error": str(e), "parse_status": "failed"})
            if s["url"] in existing_by_session:
                new_records_by_session[s["url"]] = existing_by_session[s["url"]]
                print(f"ERROR {s['title']}: {e}; kept old records")
            else:
                print(f"ERROR {s['title']}: {e}")

    records: list[BillRecord] = []
    for s in session_links:
        records.extend(new_records_by_session.get(s["url"], []))

    if args.download_pdfs:
        download_pdfs(records, args.out, sess)

    write_outputs(records, new_sources, args.out)

    run_meta = {
        "ran_at": now_iso(),
        "command": " ".join([sys.executable] + sys.argv),
        "argv": sys.argv,
        "script": "yokohama_gikai_scraper.py",
        "script_version": UA,
        "index_url": INDEX_URL,
        "years_back": args.years_back,
        "out_dir": str(args.out),
        "download_pdfs": args.download_pdfs,
        "reparse_all": args.reparse_all,
        "index_status": index_resp.status_code,
        "index_content_hash": index_hash,
        **stats,
        "records_total": len(records),
    }
    append_run_metadata(args.out, run_meta)

    print(f"\nWrote {len(records)} records from {len(session_links)} session pages to {args.out}")
    print(f"Metadata appended to {args.out / 'run_metadata.jsonl'}")


if __name__ == "__main__":
    main()