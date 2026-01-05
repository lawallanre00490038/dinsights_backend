# db.py
import duckdb

# Persistent DB file (recommended)
DUCKDB_PATH = "analytics.duckdb"

conn = duckdb.connect(DUCKDB_PATH)
conn.execute("PRAGMA threads=4")  # parallel CSV scan
