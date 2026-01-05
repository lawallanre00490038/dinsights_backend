# registry.py
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
from .db import conn

# In-memory cache for fast access
DATASET_REGISTRY: Dict[str, dict] = {}

# Configuration
DATASET_TTL_DAYS = 30  # Datasets expire after 30 days of inactivity
REGISTRY_TABLE = "dataset_registry"

def init_registry_table():
    """Initialize the registry table in DuckDB if it doesn't exist."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
            dataset_id VARCHAR PRIMARY KEY,
            table_name VARCHAR NOT NULL,
            rows INTEGER NOT NULL,
            columns TEXT NOT NULL,  -- JSON array
            source VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

def load_registry_from_db():
    """Load registry from DuckDB into memory."""
    init_registry_table()
    
    try:
        results = conn.execute(f"""
            SELECT dataset_id, table_name, rows, columns, source, created_at, last_accessed
            FROM {REGISTRY_TABLE}
        """).fetchall()
        
        DATASET_REGISTRY.clear()
        for row in results:
            dataset_id, table_name, rows, columns_json, source, created_at, last_accessed = row
            try:
                columns = json.loads(columns_json) if columns_json else []
            except:
                columns = []
            
            DATASET_REGISTRY[dataset_id] = {
                "table": table_name,
                "rows": rows,
                "columns": columns,
                "source": source or "unknown",
                "created_at": created_at.isoformat() if created_at else None,
                "last_accessed": last_accessed.isoformat() if last_accessed else None
            }
    except Exception as e:
        print(f"Error loading registry from DB: {e}")
        DATASET_REGISTRY.clear()

def save_dataset_to_registry(dataset_id: str, table_name: str, rows: int, columns: list, source: str):
    """Save a dataset entry to the persistent registry."""
    init_registry_table()
    
    columns_json = json.dumps(columns)
    now = datetime.now()
    
    # Insert or update
    conn.execute(f"""
        INSERT INTO {REGISTRY_TABLE} (dataset_id, table_name, rows, columns, source, created_at, last_accessed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (dataset_id) DO UPDATE SET
            table_name = EXCLUDED.table_name,
            rows = EXCLUDED.rows,
            columns = EXCLUDED.columns,
            source = EXCLUDED.source,
            last_accessed = EXCLUDED.last_accessed
    """, [dataset_id, table_name, rows, columns_json, source, now, now])
    
    # Update in-memory cache
    DATASET_REGISTRY[dataset_id] = {
        "table": table_name,
        "rows": rows,
        "columns": columns,
        "source": source,
        "created_at": now.isoformat(),
        "last_accessed": now.isoformat()
    }

def update_dataset_access(dataset_id: str):
    """Update the last_accessed timestamp for a dataset."""
    if dataset_id not in DATASET_REGISTRY:
        return
    
    now = datetime.now()
    conn.execute(f"""
        UPDATE {REGISTRY_TABLE}
        SET last_accessed = ?
        WHERE dataset_id = ?
    """, [now, dataset_id])
    
    if dataset_id in DATASET_REGISTRY:
        DATASET_REGISTRY[dataset_id]["last_accessed"] = now.isoformat()

def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset from registry and drop its table."""
    if dataset_id not in DATASET_REGISTRY:
        return False
    
    table_name = DATASET_REGISTRY[dataset_id]["table"]
    
    try:
        # Drop the data table
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        
        # Remove from registry
        conn.execute(f"DELETE FROM {REGISTRY_TABLE} WHERE dataset_id = ?", [dataset_id])
        
        # Remove from memory
        if dataset_id in DATASET_REGISTRY:
            del DATASET_REGISTRY[dataset_id]
        
        return True
    except Exception as e:
        print(f"Error deleting dataset {dataset_id}: {e}")
        return False

def cleanup_expired_datasets() -> int:
    """Delete datasets that haven't been accessed in TTL_DAYS days."""
    init_registry_table()
    
    cutoff_date = datetime.now() - timedelta(days=DATASET_TTL_DAYS)
    
    try:
        # Find expired datasets
        expired = conn.execute(f"""
            SELECT dataset_id, table_name
            FROM {REGISTRY_TABLE}
            WHERE last_accessed < ?
        """, [cutoff_date]).fetchall()
        
        deleted_count = 0
        for dataset_id, table_name in expired:
            try:
                # Drop the data table
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                
                # Remove from registry
                conn.execute(f"DELETE FROM {REGISTRY_TABLE} WHERE dataset_id = ?", [dataset_id])
                
                # Remove from memory
                if dataset_id in DATASET_REGISTRY:
                    del DATASET_REGISTRY[dataset_id]
                
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting expired dataset {dataset_id}: {e}")
        
        return deleted_count
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return 0

def get_registry_stats() -> dict:
    """Get statistics about the registry."""
    init_registry_table()
    
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM {REGISTRY_TABLE}").fetchone()[0]
        total_rows = conn.execute(f"SELECT SUM(rows) FROM {REGISTRY_TABLE}").fetchone()[0] or 0
        
        expired_count = conn.execute(f"""
            SELECT COUNT(*) FROM {REGISTRY_TABLE}
            WHERE last_accessed < ?
        """, [datetime.now() - timedelta(days=DATASET_TTL_DAYS)]).fetchone()[0]
        
        return {
            "total_datasets": total,
            "total_rows": total_rows,
            "expired_datasets": expired_count,
            "ttl_days": DATASET_TTL_DAYS
        }
    except Exception as e:
        print(f"Error getting registry stats: {e}")
        return {"total_datasets": 0, "total_rows": 0, "expired_datasets": 0, "ttl_days": DATASET_TTL_DAYS}

# Load registry on module import
load_registry_from_db()
