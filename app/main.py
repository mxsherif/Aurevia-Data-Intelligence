"""Aurevia's Streamlit entry point.

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
from app.ui.explore import render_explore  # noqa: E402
from app.ui.overview import render_overview  # noqa: E402
from app.ui.state import (  # noqa: E402
    STATE_PAGE,
    init_state,
    render_data_source_sidebar,
)
from app.ui.visualize import render_visualize  # noqa: E402

PAGE_ICON = "📊"

#: Page label -> renderer. Navigation is a plain radio rather than Streamlit's
#: multipage files so that every page shares one session state and one sidebar.
PAGES = {
    "Overview": render_overview,
    "Explore": render_explore,
    "Visualize": render_visualize,
}


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} — {APP_TAGLINE}",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    settings = get_settings()
    configure_logging(settings)
    init_state()

    with st.sidebar:
        st.title(f"{PAGE_ICON} {APP_NAME}")
        st.caption(APP_TAGLINE)
        page = st.radio("Page", list(PAGES), key=STATE_PAGE, label_visibility="collapsed")
        st.markdown("---")

    render_data_source_sidebar(settings)
    PAGES[page](settings)


main()
