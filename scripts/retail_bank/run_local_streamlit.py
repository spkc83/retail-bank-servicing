#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "bitsandbytes==0.50.0",
#   "huggingface-hub==1.22.0",
#   "safetensors==0.8.0",
#   "streamlit==1.48.0",
#   "torch==2.12.1",
#   "transformers==5.13.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    allow_test_skip = "--allow-test-skip" in sys.argv[1:]
    skip_flags = (
        "LOCAL_POC_SKIP_MODEL_LOAD",
        "LOCAL_POC_SKIP_ROUTER_LOAD",
    )
    active_skip_flags = [name for name in skip_flags if os.environ.get(name) == "1"]
    if active_skip_flags and not allow_test_skip:
        raise RuntimeError(
            "Refusing to launch with local test skip flags active: "
            + ", ".join(active_skip_flags)
        )

    from streamlit.web import cli as streamlit_cli

    app_path = (
        Path(__file__).resolve().parents[2]
        / "poc"
        / "retail-bank-customer-service-poc"
        / "streamlit_app.py"
    )
    port = int(os.environ.get("LOCAL_STREAMLIT_PORT", "8501"))
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
        "--browser.gatherUsageStats=false",
    ]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
