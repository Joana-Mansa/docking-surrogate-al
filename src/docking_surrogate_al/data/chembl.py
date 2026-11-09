"""ChEMBL molecule download.

Design note
-----------
Molecules are fetched from the ChEMBL REST API with drug-likeness filters
applied *server side* (molecular weight window, AlogP ceiling, zero Lipinski
violations), so the download is small and the filtering is reproducible from
the recorded query rather than from local post-processing.

Results are sorted by ChEMBL ID before truncation to ``n_molecules``. The API
does not guarantee a stable page order, so sorting is what makes "the first
5000" mean the same thing on a rerun. The raw download is written to disk
before standardisation, so the expensive network stage is never repeated when
only the standardisation parameters change.

The ChEMBL database is versioned and grows over time; the release used is
recorded in the sidecar metadata file next to the raw CSV, because a rerun
against a later release is a different dataset even with an identical config.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from ..config import ChemblConfig

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
MOLECULE_ENDPOINT = f"{BASE_URL}/molecule.json"
STATUS_ENDPOINT = f"{BASE_URL}/status.json"
USER_AGENT = "docking-surrogate-al/0.1 (research pipeline)"


class ChemblDownloadError(RuntimeError):
    """Raised when the ChEMBL API cannot be queried successfully."""


def _get(url: str, params: dict[str, Any], cfg: ChemblConfig) -> dict[str, Any]:
    """GET with bounded exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(cfg.retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=cfg.timeout_s,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            if response.status_code == 200:
                return dict(response.json())
            # 429/5xx are worth retrying; 4xx other than 429 are not.
            if response.status_code != 429 and response.status_code < 500:
                raise ChemblDownloadError(
                    f"ChEMBL returned HTTP {response.status_code} for {response.url}"
                )
            last_error = ChemblDownloadError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        sleep_s = 2.0 * (2**attempt)
        logger.warning(
            "ChEMBL request failed (attempt %d/%d): %s. Retrying in %.0fs.",
            attempt + 1,
            cfg.retries,
            last_error,
            sleep_s,
        )
        time.sleep(sleep_s)
    raise ChemblDownloadError(f"ChEMBL request failed after {cfg.retries} attempts: {last_error}")


def chembl_release() -> str | None:
    """Return the ChEMBL release string, or None if the status call fails."""
    try:
        response = requests.get(
            STATUS_ENDPOINT, timeout=60, headers={"User-Agent": USER_AGENT}
        )
        if response.status_code == 200:
            return str(response.json().get("chembl_db_version"))
    except requests.RequestException:
        return None
    return None


def download_molecules(cfg: ChemblConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download drug-like small molecules from ChEMBL.

    Args:
        cfg: ChEMBL query parameters from the run config.

    Returns:
        ``(frame, metadata)``. ``frame`` has ``chembl_id`` and ``smiles``
        columns; ``metadata`` records the query and ChEMBL release for the
        manifest.
    """
    query: dict[str, Any] = {
        "molecule_type": "Small molecule",
        "molecule_properties__mw_freebase__gte": cfg.min_mw,
        "molecule_properties__mw_freebase__lte": cfg.max_mw,
        "molecule_properties__alogp__lte": cfg.max_alogp,
        "molecule_properties__num_ro5_violations": cfg.max_ro5_violations,
        "only": "molecule_chembl_id,molecule_structures",
        "limit": min(cfg.page_size, 1000),
    }

    records: list[dict[str, str]] = []
    offset = 0
    # Over-fetch, because some records carry no structure and some are
    # multi-component; sorting then truncating happens after collection.
    target = int(cfg.n_molecules * 1.5) + 100

    while len(records) < target:
        page_params = {**query, "offset": offset}
        payload = _get(MOLECULE_ENDPOINT, page_params, cfg)
        molecules = payload.get("molecules", [])
        if not molecules:
            logger.info("ChEMBL returned no further molecules at offset %d.", offset)
            break
        for mol in molecules:
            structures = mol.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles")
            chembl_id = mol.get("molecule_chembl_id")
            if smiles and chembl_id:
                records.append({"chembl_id": chembl_id, "smiles": smiles})
        offset += len(molecules)
        logger.info("Downloaded %d candidate molecules (offset %d).", len(records), offset)
        if payload.get("page_meta", {}).get("next") is None:
            break

    if not records:
        raise ChemblDownloadError("ChEMBL returned no usable molecules for this query.")

    frame = pd.DataFrame(records).drop_duplicates(subset="chembl_id")
    # Deterministic truncation: sort by ID so "the first N" is reproducible.
    frame = frame.sort_values("chembl_id", kind="stable").head(cfg.n_molecules)
    frame = frame.reset_index(drop=True)

    metadata = {
        "source": "chembl_rest_api",
        "chembl_release": chembl_release(),
        "query": f"{MOLECULE_ENDPOINT}?{urlencode(query)}",
        "requested_n": cfg.n_molecules,
        "candidates_seen": len(records),
        "returned_n": len(frame),
    }
    logger.info(
        "ChEMBL download complete: %d molecules (release %s).",
        len(frame),
        metadata["chembl_release"],
    )
    return frame, metadata


def load_local_csv(
    path: Path, smiles_column: str, id_column: str | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load molecules from a committed CSV instead of the network.

    Used by the smoke config so that CI and a first-time user do not depend on
    the ChEMBL API being reachable. The file itself was produced from ChEMBL
    by ``scripts/freeze_smoke_set.py``, and its provenance is recorded there.
    """
    if not path.exists():
        raise FileNotFoundError(f"data.csv_path does not exist: {path}")
    frame = pd.read_csv(path)
    if smiles_column not in frame.columns:
        raise KeyError(f"{path} has no {smiles_column!r} column; got {list(frame.columns)}")
    keep = [c for c in (id_column, smiles_column) if c and c in frame.columns]
    frame = frame[keep].rename(columns={smiles_column: "smiles"})
    if id_column and id_column in frame.columns:
        frame = frame.rename(columns={id_column: "chembl_id"})
    metadata = {"source": "local_csv", "path": str(path), "returned_n": len(frame)}
    logger.info("Loaded %d molecules from %s.", len(frame), path)
    return frame.reset_index(drop=True), metadata


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write the download provenance sidecar next to the raw CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)
