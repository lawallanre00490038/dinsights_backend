import pandas as pd
from pathlib import Path
from typing import Union
from io import StringIO, BytesIO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent 


def load_csv_with_fallback(path_or_content: Union[str, bytes]) -> pd.DataFrame:
    """
    Accepts either a filesystem path (relative or absolute) or raw file content (str or bytes).
    If given content, it will attempt to decode using multiple encodings and load via pandas.
    Supports CSV and Excel (.xlsx, .xls) files.
    If given a path, behavior is unchanged from before.
    """

    # If the caller passed bytes, check if it's an Excel file first
    if isinstance(path_or_content, bytes):
        # Check for Excel file signatures
        # XLSX files start with PK (ZIP signature: 0x504B)
        # XLS files start with different signature (0xD0CF for old format)
        if path_or_content[:2] == b'PK' or path_or_content[:2] == b'\xD0\xCF':
            try:
                return pd.read_excel(BytesIO(path_or_content))
            except Exception as e:
                raise ValueError(f"Failed to parse Excel file content: {e}")
        
        # Otherwise try as CSV content
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                text = path_or_content.decode(enc)
                return pd.read_csv(StringIO(text))
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # Last resort: decode with latin1
        try:
            text = path_or_content.decode("latin1", errors="replace")
            return pd.read_csv(StringIO(text))
        except Exception:
            raise ValueError("Unable to parse CSV from provided bytes content.")
    
    # If string content with CSV-like patterns, try to parse as CSV
    if isinstance(path_or_content, str) and ("\n" in path_or_content or "," in path_or_content):
        try:
            return pd.read_csv(StringIO(path_or_content))
        except Exception as e:
            # maybe it's a path string after all; fallthrough to path logic
            pass

    # Otherwise treat as a path (string path)
    candidate_path = Path(str(path_or_content))
    if candidate_path.is_absolute():
        absolute_path = candidate_path
    else:
        absolute_path = (PROJECT_ROOT / candidate_path).resolve()
    
    print(f"Attempting to load file from: {absolute_path}") # Optional: for debugging

    if not absolute_path.exists():
         raise FileNotFoundError(
             f"File not found at resolved path: {absolute_path}"
         )

    # Check if it's an Excel file by extension
    if absolute_path.suffix.lower() in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
        try:
            return pd.read_excel(absolute_path)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file {absolute_path}: {e}")
    
    # Try multiple encodings for CSV
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(absolute_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {absolute_path} with {enc}: {e}")
            continue

    raise ValueError(f"Unable to decode file {path_or_content!r} with tried encodings.")
