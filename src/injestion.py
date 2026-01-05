# ingestion.py
import uuid
import tempfile
import os
import pandas as pd
import re
from fastapi import HTTPException
from .registry import save_dataset_to_registry, update_dataset_access
from .db import conn

# Production configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
MAX_ROWS = 10_000_000  # 10 million rows limit (adjust based on your needs)

def _validate_table_name(table_name: str) -> bool:
    """Validate table name to prevent SQL injection."""
    # Table names should match pattern: dataset_<32 hex chars>
    return bool(re.match(r'^dataset_[a-f0-9]{32}$', table_name))

def store_file_and_register_dataset(file) -> str:
    # Validate file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    dataset_id = str(uuid.uuid4()).replace("-", "")
    table_name = f"dataset_{dataset_id}"
    
    # Validate table name format (security check)
    if not _validate_table_name(table_name):
        raise ValueError(f"Invalid table name format: {table_name}")

    filename = file.filename.lower()

    try:
        # -------------------------
        # Excel files
        # -------------------------
        if filename.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(file.file)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading Excel file: {str(e)}"
                )
            
            # Validate row count
            if len(df) > MAX_ROWS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dataset too large. Maximum {MAX_ROWS:,} rows allowed. Found {len(df):,} rows."
                )

            conn.register("temp_df", df)
            # Use parameterized approach - table name is validated, but still be careful
            conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM temp_df')
            conn.unregister("temp_df")

        # -------------------------
        # CSV / TSV / TXT
        # -------------------------
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(file.file.read())
                tmp_path = tmp.name

            try:
                # First, check row count before creating table
                row_check = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM read_csv_auto(
                        '{tmp_path}',
                        HEADER=true,
                        strict_mode=false,
                        ignore_errors=true,
                        null_padding=true
                    )
                    """
                ).fetchone()[0]
                
                if row_check > MAX_ROWS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Dataset too large. Maximum {MAX_ROWS:,} rows allowed. Found {row_check:,} rows."
                    )
                
                # Create table with validated table name
                conn.execute(
                    f"""
                    CREATE TABLE "{table_name}" AS
                    SELECT * FROM read_csv_auto(
                        '{tmp_path}',
                        HEADER=true,
                        strict_mode=false,
                        ignore_errors=true,
                        null_padding=true
                    )
                    """
                )
            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading CSV file: {str(e)}"
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # -------------------------
        # Metadata
        # -------------------------
        try:
            row_count = conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            columns = [
                col[0]
                for col in conn.execute(f'DESCRIBE "{table_name}"').fetchall()
            ]

            # Save to persistent registry
            save_dataset_to_registry(
                dataset_id=dataset_id,
                table_name=table_name,
                rows=row_count,
                columns=columns,
                source=file.filename
            )

            return dataset_id
        except Exception as e:
            # Cleanup: drop table if metadata extraction fails
            try:
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            except:
                pass
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting dataset metadata: {str(e)}"
            )
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # Cleanup on any other error
        try:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        except:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
