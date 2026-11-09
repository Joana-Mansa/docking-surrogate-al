"""Regenerate the frozen 200-molecule smoke input from ChEMBL.

The smoke config reads ``data/smoke/chembl_smoke_raw.csv`` from disk instead
of querying ChEMBL, so that CI and a first-time clone do not depend on the
ChEMBL API being reachable or on the live database release. That file is
committed, and this script is how it was produced and how it can be refreshed.

Refreshing it is a deliberate act, not part of any pipeline stage: the
molecules will differ from the committed set, so smoke-test numbers before and
after a refresh are not comparable.

Usage:
    python scripts/freeze_smoke_set.py [--n 200] [--out data/smoke/chembl_smoke_raw.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docking_surrogate_al.config import ChemblConfig, REPO_ROOT
from docking_surrogate_al.data.chembl import download_molecules


def main() -> None:
    """Download molecules and write the frozen CSV plus its provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="number of molecules to freeze")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "smoke" / "chembl_smoke_raw.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    frame, metadata = download_molecules(ChemblConfig(n_molecules=args.n))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    sidecar = args.out.with_suffix(".provenance.json")
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)

    print(f"Wrote {len(frame)} molecules to {args.out}")
    print(f"Provenance written to {sidecar}")


if __name__ == "__main__":
    main()
