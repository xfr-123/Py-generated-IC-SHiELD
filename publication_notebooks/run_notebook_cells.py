#!/usr/bin/env python3
"""Execute a notebook with a real Jupyter kernel and save outputs in place."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--status-dir", type=Path, required=True)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    root = notebook_path.parents[1]
    os.chdir(root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/james-publication-mpl")
    os.environ.setdefault("PUBLICATION_REBUILD_FIGURES", os.environ.get("PUBLICATION_FULL_REBUILD", "1"))

    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [(index, cell) for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"]
    started = time.time()
    status = {
        "notebook": str(notebook_path.relative_to(root)),
        "python": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "rebuild_figures": os.environ.get("PUBLICATION_REBUILD_FIGURES"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "code_cells": len(code_cells),
        "completed_cells": 0,
        "cells": [],
        "success": False,
    }

    print(f"NOTEBOOK_START {notebook_path.relative_to(root)} code_cells={len(code_cells)}", flush=True)
    try:
        client = NotebookClient(
            notebook,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
            allow_errors=False,
            record_timing=True,
        )
        with client.setup_kernel():
            for ordinal, (cell_index, cell) in enumerate(code_cells, start=1):
                cell_started = time.time()
                print(f"CELL_START ordinal={ordinal} index={cell_index}", flush=True)
                try:
                    client.execute_cell(cell, cell_index, execution_count=ordinal)
                except Exception:
                    elapsed = time.time() - cell_started
                    status["cells"].append({"index": cell_index, "seconds": elapsed, "success": False})
                    print(f"CELL_FAIL index={cell_index} seconds={elapsed:.3f}", flush=True)
                    traceback.print_exc()
                    raise
                elapsed = time.time() - cell_started
                status["completed_cells"] += 1
                status["cells"].append({"index": cell_index, "seconds": elapsed, "success": True})
                print(f"CELL_OK index={cell_index} seconds={elapsed:.3f}", flush=True)
        status["success"] = True
        return_code = 0
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        return_code = 1
    finally:
        nbformat.write(notebook, notebook_path)
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status["elapsed_seconds"] = time.time() - started
        args.status_dir.mkdir(parents=True, exist_ok=True)
        status_path = args.status_dir / f"{notebook_path.stem}.json"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"NOTEBOOK_END success={status['success']} seconds={status['elapsed_seconds']:.3f} status={status_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
