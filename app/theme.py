"""BlueAlpha-inspired theme: inject custom CSS (light + optional night)."""

from pathlib import Path

import streamlit as st


def inject_theme_css(*, night: bool = False) -> None:
    base_path = Path(__file__).resolve().parent / "static" / "theme.css"
    if not base_path.is_file():
        return
    css = base_path.read_text(encoding="utf-8")
    if night:
        night_path = Path(__file__).resolve().parent / "static" / "theme_night.css"
        if night_path.is_file():
            css += "\n" + night_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
