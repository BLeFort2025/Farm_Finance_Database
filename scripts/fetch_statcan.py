import io, json, zipfile, requests, pandas as pd
from datetime import date
from pathlib import Path
import yaml
import hashlib

DATA_LATEST = Path("data/latest")
DATA_ARCH   = Path("data/archive")
MANIFEST_FP = Path("data/manifest.json")
TABLES_YML  = Path("config/tables.yml")

def csv_zip_url(table_id: str) -> str:
    # Example: "32-10-0045-01" -> "32100045" -> "32100045-eng.zip"
    pid = table_id.replace("-", "")      # "3210004501"
    file_id = pid[:8]                    # "32100045"
    return f"https://www150.statcan.gc.ca/n1/tbl/csv/{file_id}-eng.zip"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_manifest():
    if MANIFEST_FP.exists():
        return json.loads(MANIFEST_FP.read_text())
    return {"tables": {}}

def save_manifest(m):
    MANIFEST_FP.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FP.write_text(json.dumps(m, indent=2))

def fetch_table(table_id: str) -> tuple[pd.DataFrame, str]:
    url = csv_zip_url(table_id)
    print(f"[INFO] Downloading {table_id} from {url}")
    r = requests.get(url, timeout=180, verify=False)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f, low_memory=False)
    return df, sha256_bytes(r.content)

def persist_if_changed(table_id: str, df: pd.DataFrame, content_hash: str) -> bool:
    latest_fp = DATA_LATEST / f"{table_id}.csv"
    DATA_LATEST.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    prev_hash = manifest["tables"].get(table_id, {}).get("content_hash")

    if prev_hash == content_hash and latest_fp.exists():
        return False

    df.to_csv(latest_fp, index=False)

    stamp = date.today().isoformat()
    arch_dir = DATA_ARCH / stamp
    arch_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(arch_dir / f"{table_id}.csv", index=False)

    manifest["tables"][table_id] = {
        "content_hash": content_hash,
        "rows": int(len(df)),
        "last_updated_local": stamp,
    }
    save_manifest(manifest)
    return True

def run():
    cfg = yaml.safe_load(TABLES_YML.read_text())
    changed = []
    failed = []

    for t in cfg["tables"]:
        tid = t["id"]
        if not t.get("active", True):
            continue

        try:
            df, h = fetch_table(tid)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {tid}: {e}")
            failed.append(tid)
            continue

        if persist_if_changed(tid, df, h):
            changed.append(tid)

    print(f"Updated: {changed}" if changed else "No changes.")
    if failed:
        print(f"Failed to update: {failed}")

if __name__ == "__main__":
    run()
