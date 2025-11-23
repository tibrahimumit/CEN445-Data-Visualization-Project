import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_data

AVAILABLE_COLS = {
    'Specialization': 'Specialization',
    'Gender': 'Gender',
    'GraduationYear': 'Graduation Year',
    'Degree': 'Degree (B.Tech/M.Tech)',
    'CollegeCityTier': 'City Tier',
    'CollegeState': 'College State',
    'CollegeTier': 'College Tier'
}

METRICS = {
    'Salary': 'Salary',
    'collegeGPA': 'College GPA',
    'Quant': 'Quantitative Score'
}

AGG_FUNCS = {
    'avg': 'Average',
    'max': 'Maximum',
    'min': 'Minimum',
    'count': 'Count'
}


def main() -> None:
    st.set_page_config(
        page_title="Multi-Dimensional Heatmap & Treemap",
        page_icon="🎛️",
        layout="wide"
    )
    
    df = load_data()

    st.subheader("🌳 Hierarchical Treemap")
    st.markdown(
        "View the data as nested rectangles (e.g., Degree → Specialization → Gender):"
    )
    
    path_options = st.multiselect(
        "Select Hierarchy Order:",
        list(AVAILABLE_COLS.keys()),
        default=['Degree', 'Specialization', 'Gender']
    )
    
    if path_options:
        fig_tree = px.treemap(
            df,
            path=path_options,
            values='Salary',
            color='Salary',
            color_continuous_scale='RdBu',
            title="Salary Distribution Hierarchy"
        )
        fig_tree.data[0].marker.colorscale = 'Viridis'
        fig_tree.update_layout(
            modebar=dict(
                bgcolor="rgba(0,0,0,0.3)",
                color="white",
                activecolor="#00D9FF"
            )
        )

        config_tree = {
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'salary_treemap',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
        
        st.plotly_chart(fig_tree, use_container_width=True, config=config_tree)
    else:
        st.warning("Please select at least one hierarchy level for the treemap.")
    
    st.info(
        """
        **How to Use:**
        * **Heatmap (Top):** Shows density at the intersection of two variables. For example, 
          set X to "Year", Y to "Specialization", and Color to "Salary" to see how salaries 
          evolved by specialization over time.
        * **Treemap (Bottom):** Displays groups as proportional rectangles. Large box = 
          many students or high total salary.
        """
    )

if __name__ == "__main__":
    main()