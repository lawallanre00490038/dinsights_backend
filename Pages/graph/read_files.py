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
    If given a path, behavior is unchanged from before.
    """

    # If the caller passed bytes or text content, try to parse directly
    if isinstance(path_or_content, (bytes, str)) and ("\n" in str(path_or_content) or "," in str(path_or_content)):
        # Heuristic: looks like CSV content rather than a path
        content = path_or_content
        # If bytes, try decoding with common encodings
        if isinstance(content, bytes):
            for enc in ("utf-8", "cp1252", "latin1"):
                try:
                    text = content.decode(enc)
                    return pd.read_csv(StringIO(text))
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # If pandas fails for other reasons, raise a helpful error
                    raise
            # Last resort: decode with latin1 which won't fail but may mangle characters
            try:
                text = content.decode("latin1", errors="replace")
                return pd.read_csv(StringIO(text))
            except Exception:
                raise ValueError("Unable to parse CSV from provided bytes content.")
        else:
            # content is str
            try:
                return pd.read_csv(StringIO(content))
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

    # Try multiple encodings
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(absolute_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {absolute_path} with {enc}: {e}")
            continue

    raise ValueError(f"Unable to decode file {path_or_content!r} with tried encodings.")
