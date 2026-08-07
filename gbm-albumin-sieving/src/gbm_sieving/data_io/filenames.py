"""Canonical artifact filenames for generated project results."""

from __future__ import annotations

import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_token(value: object) -> str:
    """Return a filesystem-safe, stable token without silently changing case."""
    token = _TOKEN_RE.sub("-", str(value).strip()).strip("-_.")
    if not token:
        raise ValueError("filename token cannot be empty")
    return token


def decimal_token(value: float, *, digits: int = 6) -> str:
    """Encode a decimal for filenames, e.g. 4.25 -> 4p25 and -0.7 -> m0p7."""
    number = float(value)
    sign = "m" if number < 0 else ""
    text = f"{abs(number):.{digits}f}".rstrip("0").rstrip(".")
    return sign + text.replace(".", "p")


def artifact_filename(
    sample_id: str,
    artifact: str,
    *,
    variant: str | None = None,
    extension: str = ".xlsx",
) -> str:
    """Build `{sample}__{artifact}[__{variant}].ext`."""
    ext = extension if extension.startswith(".") else f".{extension}"
    parts = [safe_token(sample_id), safe_token(artifact)]
    if variant:
        parts.append(safe_token(variant))
    return "__".join(parts) + ext.lower()


def radius_variant(radius_nm: float) -> str:
    """Return a canonical effective-radius variant token."""
    return f"radius-{decimal_token(radius_nm)}nm"


def find_artifact(
    directory: str | Path,
    sample_id: str,
    artifact: str,
    *,
    variant: str | None = None,
) -> Path | None:
    """Return the canonical artifact path when it exists."""
    directory = Path(directory)
    canonical = directory / artifact_filename(sample_id, artifact, variant=variant)
    return canonical if canonical.is_file() else None
