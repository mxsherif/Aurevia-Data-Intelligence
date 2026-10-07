"""DataPilot's Streamlit entry point.

Run it with ``python run.py`` (preferred) or ``streamlit run app/main.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app/main.py` from the project root, where the parent
# directory is not automatically on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.config import APP_NAME, APP_TAGLINE, configure_logging, get_settings  # noqa: E402
from app.ui.overview import STATE_ERROR, STATE_LOAD, STATE_PROFILE, render_overview, render_sidebar  # noqa: E402

PAGE_ICON = "📊"


def _init_state() -> None:
    for key in (STATE_LOAD, STATE_PROFILE, STATE_ERROR):
        st.session_state.setdefault(key, None)


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} — {APP_TAGLINE}",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    settings = get_settings()
    configure_logging(settings)
    _init_state()

    with st.sidebar:
        st.title(f"{PAGE_ICON} {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.markdown("---")

    render_sidebar(settings)
    render_overview(settings)


main()
