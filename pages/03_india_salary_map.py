from __future__ import annotations

from typing import Dict, List

import json
from pathlib import Path

import plotly.express as px
import streamlit as st

from data_loader import load_data

GEOJSON_FILE = "india_states.geojson"
GEOJSON_FEATURE_KEY = "properties.ST_NM"

STATE_NAME_NORMALIZATION: Dict[str, str] = {
    "Orissa": "Odisha",
    "Union Territory": "Chandigarh",
    "Jammu and Kashmir": "Jammu & Kashmir",
}


@st.cache_data(show_spinner=False)
def load_india_geojson():
    geojson_path = Path(__file__).resolve().parent.parent / GEOJSON_FILE
    if not geojson_path.exists():
        st.error(
            f"GeoJSON file not found at {geojson_path}. "
            "Please ensure india_states.geojson is in the project root."
        )
        raise FileNotFoundError(f"{geojson_path} missing.")
    with open(geojson_path, encoding="utf-8") as f:
        return json.load(f)


def get_color_scale_for_gender(selected_gender: List[str]):
    if selected_gender == ["M"]:
        return ["#A1C4FD", "#0091FF", "#003B7A"]
    if selected_gender == ["F"]:
        return ["#FFD1DC", "#FF69B4", "#C71585"]
    return "Viridis"


def aggregate_salary_by_state(df):
    subset = df.dropna(subset=["CollegeState", "Salary"]).copy()
    if subset.empty:
        return subset

    subset["CollegeState"] = (
        subset["CollegeState"]
        .astype(str)
        .str.strip()
        .str.title()
        .replace(STATE_NAME_NORMALIZATION)
    )

    grouped = (
        subset.groupby("CollegeState")
        .agg(avg_salary=("Salary", "mean"), student_count=("Salary", "size"))
        .reset_index()
    )
    return grouped


def india_salary_choropleth(df_state, india_geojson, selected_gender):
    color_scale = get_color_scale_for_gender(selected_gender)
    fig = px.choropleth(
        df_state,
        geojson=india_geojson,
        locations="CollegeState",
        featureidkey=GEOJSON_FEATURE_KEY,
        color="avg_salary",
        color_continuous_scale=color_scale,
        hover_data={
            "CollegeState": True,
            "avg_salary": ":.0f",
            "student_count": True,
        },
        labels={
            "CollegeState": "State",
            "avg_salary": "Average Salary (₹)",
            "student_count": "Students",
        },
        title="Average Graduate Salary by Indian State",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        height=600,
        modebar=dict(
            bgcolor="rgba(0,0,0,0.3)",
            color="white",
            activecolor="#00D9FF"
        )
    )
    return fig


def apply_filter(df, column, selections):
    if not selections:
        return df
    return df[df[column].isin(selections)]


def main() -> None:
    st.set_page_config(page_title="India Salary Map", layout="wide")
    df = load_data()

    if "CollegeState" not in df.columns or "Salary" not in df.columns:
        st.warning("Dataset must contain both CollegeState and Salary columns.")
        return

    st.subheader("India Salary Choropleth Map")
    st.markdown(
        "Explore how average graduate salaries vary across Indian states. "
        "Use the filters to focus on specific degree programs, departments, or genders. "
        "The color palette adapts depending on the selected gender filter."
    )

    geojson = load_india_geojson()

    degree_opts = sorted(df["Degree"].dropna().unique()) if "Degree" in df.columns else []
    spec_opts = (
        sorted(df["Specialization"].dropna().unique())
        if "Specialization" in df.columns
        else []
    )
    gender_opts = sorted(df["Gender"].dropna().unique()) if "Gender" in df.columns else []

    filter_cols = st.columns(3)
    with filter_cols[0]:
        selected_degree = st.multiselect("Degree", degree_opts, default=degree_opts)
    with filter_cols[1]:
        selected_spec = st.multiselect("Specialization", spec_opts, default=spec_opts)
    with filter_cols[2]:
        selected_gender = st.multiselect("Gender", gender_opts, default=gender_opts)

    df_filtered = apply_filter(df, "Degree", selected_degree) if "Degree" in df.columns else df
    df_filtered = (
        apply_filter(df_filtered, "Specialization", selected_spec)
        if "Specialization" in df.columns
        else df_filtered
    )
    df_filtered = (
        apply_filter(df_filtered, "Gender", selected_gender)
        if "Gender" in df.columns
        else df_filtered
    )

    df_state = aggregate_salary_by_state(df_filtered)
    if df_state.empty:
        st.warning("No data available for the current filter selection.")
        return

    fig = india_salary_choropleth(df_state, geojson, selected_gender)
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'india_salary_map',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)

    st.caption(
        "States are colored by average graduate salary. The color scale switches to blue "
        "tones for male selections, pink tones for female selections, and a neutral palette "
        "when both genders are included."
    )


if __name__ == "__main__":
    main()


