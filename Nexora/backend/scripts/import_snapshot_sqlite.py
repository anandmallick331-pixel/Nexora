"""
Import EcoTrace production dataset snapshot into SQLite database (s21_db.sqlite3).
Preserves exact IDs, timestamps, foreign keys, and JSON fields.
"""
import json
import os
import sqlite3
import sys

APPLICATION_TABLES = [
    "destinations",
    "sources",
    "locations",
    "datasets",
    "metric_definitions",
    "observations",
    "evidence",
    "business_registrations",
    "community_evidence_submissions",
    "source_conflicts",
    "observation_reconciliations",
    "observation_reconciliation_members",
]

def import_to_sqlite(snapshot_file: str, db_file: str):
    if not os.path.exists(snapshot_file):
        raise FileNotFoundError(f"Snapshot not found: {snapshot_file}")
    
    print(f"Loading snapshot: {snapshot_file}...")
    with open(snapshot_file, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    data = snapshot.get("data", {})
    
    print(f"Connecting to SQLite: {db_file}...")
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    
    # Disable foreign keys temporarily during batch restore
    cur.execute("PRAGMA foreign_keys = OFF;")
    
    try:
        for table_name in APPLICATION_TABLES:
            rows = data.get(table_name, [])
            if not rows:
                print(f"  {table_name:35} : 0 rows (skipped)")
                continue

            # Clear existing data in table
            cur.execute(f'DELETE FROM "{table_name}";')

            columns = list(rows[0].keys())
            col_names = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join(["?"] * len(columns))
            sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

            batch_data = []
            for row in rows:
                row_vals = []
                for col in columns:
                    val = row.get(col)
                    if isinstance(val, (dict, list)):
                        row_vals.append(json.dumps(val))
                    elif isinstance(val, bool):
                        row_vals.append(1 if val else 0)
                    else:
                        row_vals.append(val)
                batch_data.append(row_vals)

            cur.executemany(sql, batch_data)
            print(f"  [OK] {table_name:35} : {len(rows):5} rows imported")

        conn.commit()
        print("\n--- Verifying Counts in SQLite ---")
        for table_name in APPLICATION_TABLES:
            cur.execute(f'SELECT count(*) FROM "{table_name}";')
            cnt = cur.fetchone()[0]
            print(f"  {table_name:35} : {cnt:5} rows in DB")

        print("\n[SUCCESS] SQLite database successfully seeded from production snapshot!")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Import failed: {e}")
        raise
    finally:
        cur.execute("PRAGMA foreign_keys = ON;")
        conn.close()

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snapshot_path = os.path.join(base_dir, "data", "production_snapshot.json")
    db_path = os.path.join(base_dir, "s21_db.sqlite3")
    import_to_sqlite(snapshot_path, db_path)
