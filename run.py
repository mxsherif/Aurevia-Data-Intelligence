#!/usr/bin/env python
"""Launch the Aurevia Streamlit app.

    python run.py
    python run.py --port 8502
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENTRYPOINT = PROJECT_ROOT / "app" / "main.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Aurevia Streamlit app.")
    parser.add_argument("--port", type=int, default=8501, help="port to serve on")
    parser.add_argument("--host", default="localhost", help="address to bind")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="do not open a browser window (useful in containers / CI)",
    )
    args = parser.parse_args()

    if not ENTRYPOINT.exists():
        print(f"error: entry point not found: {ENTRYPOINT}", file=sys.stderr)
        return 1

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(
            "error: streamlit is not installed.\n"
            "       install the dependencies with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    sys.argv = [
        "streamlit",
        "run",
        str(ENTRYPOINT),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.headless:
        sys.argv += ["--server.headless", "true"]

    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
