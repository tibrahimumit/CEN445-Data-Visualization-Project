from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = "Cleaned_Engineering_graduate_salary.csv"

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the dataset and guarantee required derived columns exist."""
    df = pd.read_csv(DATA_PATH)

    if "PlacementStatus" not in df.columns:
        df["PlacementStatus"] = df["Salary"].apply(lambda s: "Placed" if s > 0 else "Not Placed")

    df["CollegeTierLabel"] = df["CollegeTier"].apply(
        lambda tier: f"Tier {int(tier)}" if pd.notna(tier) else "Tier Unknown"
    )

    def categorize_salary(value: float) -> str:
        if pd.isna(value):
            return "Salary Unknown"
        if value < 50_000:
            return "Low (<50k)"
        if value <= 100_000:
            return "Medium (50k-100k)"
        return "High (>100k)"

    df["SalaryCategory"] = df["Salary"].apply(categorize_salary)

    return df


