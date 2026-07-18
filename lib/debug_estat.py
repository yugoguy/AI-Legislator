"""Proof that the e-Stat fetch downloads real data and saves it to disk.

Run:
    export ESTAT_APP_ID=<your app id>
    python debug_estat.py                       # searches 保育所 in 横浜市
    python debug_estat.py --query 人口
    python debug_estat.py --id 0003411172       # skip search, fetch this table

It runs the real API, with no stubs:
  1. getStatsList  -> finds candidate tables for the keyword
  2. getStatsData  -> downloads the FIRST candidate in full, following NEXT_KEY
  3. writes estat_<id>.json and estat_<id>.csv into ./estat_debug/
  4. reloads the CSV from disk and prints its shape and head

If step 4 prints rows, the data reached the working directory, which is the whole
point: it is what the AnalyzeData agent reads.
"""

import argparse
import os
import sys
from pathlib import Path

from estat import EstatTool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="保育所 横浜市",
                    help="keyword to search for a table")
    ap.add_argument("--id", default="",
                    help="statsDataId to fetch directly, skipping the search")
    ap.add_argument("--out", default="./estat_debug",
                    help="working directory to save into")
    args = ap.parse_args()

    if not os.getenv("ESTAT_APP_ID"):
        print("ESTAT_APP_ID is not set. Register free at https://www.e-stat.go.jp/api/")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tool = EstatTool(max_results=5)
    tool.set_data_dir(out)          # this is what research.py must now do

    stats_data_id = args.id

    # 1. search
    if not stats_data_id:
        print(f"[1] getStatsList  query={args.query!r}")
        res = tool.use_tool(query=args.query)
        print(f"    ok={res.ok}")
        print("    " + res.text.replace("\n", "\n    "))
        if not res.ok:
            return 1
        # Take the first "[id] ..." line as the table to download.
        for line in res.text.splitlines():
            if line.startswith("["):
                stats_data_id = line[1:].split("]", 1)[0]
                break
        if not stats_data_id:
            print("No table found for that keyword. Try --query with a broader term.")
            return 1
        print(f"\n    -> selected table {stats_data_id}")

    # 2 + 3. download in full and save
    print(f"\n[2] getStatsData  statsDataId={stats_data_id}  (paginating NEXT_KEY)")
    res = tool.use_tool(stats_data_id=stats_data_id)
    print(f"    ok={res.ok}")
    print("    " + res.text.replace("\n", "\n    "))
    if not res.ok:
        return 1

    # 4. prove the file is on disk and is readable as data
    csv_path = out / f"estat_{stats_data_id}.csv"
    json_path = out / f"estat_{stats_data_id}.json"
    print(f"\n[3] files now in {out.resolve()}")
    for p in sorted(out.iterdir()):
        print(f"    {p.name:40s} {p.stat().st_size:>10,d} bytes")

    if not csv_path.exists():
        print("\nFAILED: no CSV was written.")
        return 1

    print(f"\n[4] reloading {csv_path.name} from disk")
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"    shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
        print(f"    columns: {list(df.columns)}")
        print(df.head(10).to_string(index=False))
    except ImportError:
        import csv as _csv
        with open(csv_path, encoding="utf-8") as fh:
            rows = list(_csv.reader(fh))
        print(f"    {len(rows) - 1:,} data rows, {len(rows[0])} cols")
        print(f"    header: {rows[0]}")
        for r in rows[1:6]:
            print(f"    {r}")

    print("\nPASS: real e-Stat data is on disk in the working directory the "
          "AnalyzeData agent runs in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
