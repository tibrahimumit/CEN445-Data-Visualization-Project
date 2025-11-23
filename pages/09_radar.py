import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_data

def prepare_radar_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for radar chart analysis.
    Filters to top 6 specializations and creates success tier categories.
    """

    df['Specialization'] = df['Specialization'].str.lower()
    top_specs = df['Specialization'].value_counts().nlargest(6).index.tolist()
    df = df[df['Specialization'].isin(top_specs)]
    
    spec_map = {
        'electronics and communication engineering': 'Electronics',
        'computer science & engineering': 'CS',
        'information technology': 'IT',
        'computer engineering': 'Comp. Eng.',
        'mechanical engineering': 'Mechanical',
        'electronics and electrical engineering': 'Electrical'
    }
    df['Specialization'] = df['Specialization'].replace(spec_map)
    
    salary_threshold = df['Salary'].quantile(0.75)
    df['Success_Tier'] = df['Salary'].apply(
        lambda x: 'Top 25% (High Income)' if x >= salary_threshold else 'Standard Income'
    )
    
    return df


NUMERIC_COLS = [
    'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism', 'openess_to_experience',
    'Quant', 'Logical', 'English', 'ComputerProgramming',
    'collegeGPA', '12percentage'
]

COL_MAP = {
    'conscientiousness': 'Conscientiousness',
    'agreeableness': 'Agreeableness',
    'extraversion': 'Extraversion',
    'nueroticism': 'Neuroticism',
    'openess_to_experience': 'Openness',
    'Quant': 'Quantitative',
    'Logical': 'Logical',
    'English': 'English',
    'ComputerProgramming': 'Programming',
    'collegeGPA': 'College GPA',
    '12percentage': '12th Grade Score'
}

def normalize_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize specified columns to 0-100 scale."""
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    return df_normalized


def build_radar_chart(radar_data: pd.DataFrame, split_mode: str, selected_axes: list, col_map: dict):
    """Build the interactive radar chart."""
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    for i, row in radar_data.iterrows():
        group_name = row[split_mode]
        
        r_values = row[selected_axes].values.tolist()
        r_values += r_values[:1]
        
        theta_values = [col_map[c] for c in selected_axes]
        theta_values += theta_values[:1]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            fill='toself',
            name=str(group_name),
            line_color=colors[i % len(colors)],
            opacity=0.7
        ))
    
    split_display = {
        'Specialization': 'Specialization',
        'Success_Tier': 'Success Tier',
        'Gender': 'Gender'
    }
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(240,240,240, 0.1)'
        ),
        showlegend=True,
        modebar=dict(
            bgcolor="rgba(0,0,0,0.3)",
            color="white",
            activecolor="#00D9FF"
        ),
        title=f"Skills and Personality Profile by {split_display.get(split_mode, split_mode)}",
        height=600
    )
    
    return fig


def main() -> None:
    st.set_page_config(
        page_title="360° Candidate Profile - Radar",
        page_icon="📡",
        layout="wide"
    )
    
    df = load_data()
    df = prepare_radar_data(df)
    
    st.title("📡 360° Candidate Profile Analysis (Radar Chart)")
    st.markdown(
        """
        This visualization combines **Intelligence, Academic Performance, and Personality** into one unified view.
        All features are **normalized to 0-100 scale** for comparability (e.g., GPA and personality scores 
        originally have different ranges).
        """
    )
    
    df_normalized = normalize_features(df, NUMERIC_COLS)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Settings")

        split_mode = st.radio(
            "Compare By:",
            ["Specialization", "Success_Tier", "Gender"],
            format_func=lambda x: {
                "Specialization": "Specialization",
                "Success_Tier": "Success Level (Salary)",
                "Gender": "Gender"
            }[x]
        )
        
        if split_mode != "Specialization":
            selected_spec_filter = st.selectbox(
                "Filter by Specialization:",
                ["All"] + list(df['Specialization'].unique())
            )
            if selected_spec_filter != "All":
                df_normalized = df_normalized[df_normalized['Specialization'] == selected_spec_filter]
        
        st.divider()
        
        default_axes = ['Quant', 'Logical', 'ComputerProgramming', 'conscientiousness', 'extraversion', 'collegeGPA']
        selected_axes = st.multiselect(
            "Chart Axes:",
            list(COL_MAP.keys()),
            default=default_axes,
            format_func=lambda x: COL_MAP[x]
        )

    with col2:
        if not selected_axes:
            st.warning("Please select at least one axis.")
        else:
            radar_data = df_normalized.groupby(split_mode)[selected_axes].mean().reset_index()

            fig = build_radar_chart(radar_data, split_mode, selected_axes, COL_MAP)

            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'radar_profile',
                    'height': 1080,
                    'width': 1920,
                    'scale': 2
                }
            }
            
            st.plotly_chart(fig, use_container_width=True, config=config)

            st.info(
                """
                **What This Chart Reveals:**
                * **Select "Success Level":** See which traits (e.g., Programming or Logical skills) distinguish 
                  high earners (Top 25%) from the rest.
                * **Select "Specialization":** Compare how Computer Engineers might have higher Quantitative scores 
                  while Mechanical Engineers score higher in Conscientiousness.
                * **Change Axes:** Add or remove features from the left panel to explore different dimensions 
                  (e.g., add English or 12th Grade Score).
                """
            )

if __name__ == "__main__":
    main()