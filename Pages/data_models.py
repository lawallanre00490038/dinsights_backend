import re
from typing import Optional, Union
from pydantic import BaseModel, Field, validator, field_validator


def _sanitize_variable_name(raw_name: str) -> str:
    candidate = re.sub(r"\s+", "_", raw_name.strip())
    candidate = re.sub(r"[^0-9a-zA-Z_]", "_", candidate)
    candidate = candidate.strip("_")
    if not candidate:
        raise ValueError("variable_name must contain at least one alphanumeric character.")
    if candidate[0].isdigit():
        candidate = f"data_{candidate}"
    return candidate.lower()


class InputData(BaseModel):
    variable_name: str = Field(..., min_length=1, description="Symbol used to reference the dataset in code")
    # Either provide a path to the dataset (data_path) or the raw file contents (data_content).
    data_path: Optional[str] = Field(None, description="Relative or absolute path to the dataset")
    data_content: Optional[Union[str, bytes]] = Field(
        None, description="Optional raw file content (str or bytes). If provided, this will be used instead of data_path."
    )
    data_description: str = Field("", description="Optional human-readable description of the dataset")

    @validator("variable_name", pre=True)
    def sanitize_variable_name(cls, value: str) -> str:
        if value is None or not str(value).strip():
            raise ValueError("variable_name must not be empty or whitespace.")
        return _sanitize_variable_name(str(value))

    @validator("data_path")
    def validate_data_path(cls, value: Optional[str]) -> Optional[str]:
        # data_path may be None when data_content is provided. Validation of the pair is handled elsewhere
        if value is None:
            return None
        if not str(value).strip():
            raise ValueError("data_path must not be empty or whitespace.")
        return str(value).strip()

    @validator("data_content")
    def validate_data_content(cls, value: Optional[Union[str, bytes]]) -> Optional[Union[str, bytes]]:
        # Accept None, str or bytes. Leave further decoding to the loader.
        if value is None:
            return None
        if not isinstance(value, (str, bytes)):
            raise ValueError("data_content must be a string or bytes containing the file contents.")
        return value
