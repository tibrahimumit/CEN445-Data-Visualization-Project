import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_data

def prepare_box_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for box plot analysis.
    Filters to top 8 specializations and normalizes names.
    """
    df['Specialization'] = df['Specialization'].str.lower()
    top_specs = df['Specialization'].value_counts().nlargest(8).index.tolist()
    df = df[df['Specialization'].isin(top_specs)]
    
    spec_map = {
        'electronics and communication engineering': 'Electronics',
        'computer science & engineering': 'CS',
        'information technology': 'IT',
        'computer engineering': 'Comp. Eng.',
        'computer application': 'MCA',
        'mechanical engineering': 'Mechanical',
        'electronics and electrical engineering': 'Electrical',
        'electronics & telecommunications': 'Telecom'
    }
    df['Specialization'] = df['Specialization'].replace(spec_map)
    return df

TRAITS = {
    'conscientiousness': 'Conscientiousness',
    'agreeableness': 'Agreeableness',
    'extraversion': 'Extraversion',
    'nueroticism': 'Neuroticism',
    'openess_to_experience': 'Openness to Experience'
}

def main() -> None:
    st.set_page_config(
        page_title="Personality Profile & Salary Analysis",
        page_icon="🎛️",
        layout="wide"
    )
    
    df = load_data()
    df = prepare_box_data(df)
    
    st.title("🎛️ Multi-Filter Personality Profile & Salary Analysis")
    st.markdown(
        "Use the sidebar filters to create a **specific personality profile** and compare "
        "salaries against the general market."
    )

    st.sidebar.header("Build Personality Profile")
    st.sidebar.info(
        "Select the desired score range for each trait. By default, all ranges are selected."
    )
    
    filtered_df = df.copy()
    current_filters = {}
    
    for col_name, display_name in TRAITS.items():
        min_val = float(df[col_name].min())
        max_val = float(df[col_name].max())
        
        user_range = st.sidebar.slider(
            display_name,
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=0.1
        )

        filtered_df = filtered_df[
            (filtered_df[col_name] >= user_range[0]) & 
            (filtered_df[col_name] <= user_range[1])
        ]
        current_filters[display_name] = user_range

    col1, col2, col3 = st.columns(3)

    n_selected = len(filtered_df)
    n_total = len(df)
    avg_salary_selected = filtered_df['Salary'].mean() if n_selected > 0 else 0
    avg_salary_total = df['Salary'].mean()
    
    with col1:
        st.metric("Matching Candidates", f"{n_selected}", f"Out of {n_total} total")
    with col2:
        delta_val = avg_salary_selected - avg_salary_total
        st.metric(
            "Avg Salary (Filtered)",
            f"₹{avg_salary_selected:,.0f}",
            f"{delta_val:,.0f} (vs. Overall)"
        )
    with col3:
        ratio = (n_selected / n_total) * 100
        st.metric("% of Population", f"{ratio:.1f}%")
    
    if n_selected == 0:
        st.error("No candidates match the selected criteria. Please broaden the filters.")
    else:
        st.subheader("📊 Salary Performance by Specialization (Filtered vs. Overall)")
        
        comp_df_selected = filtered_df[['Specialization', 'Salary']].copy()
        comp_df_selected['Group'] = 'Filtered Profile'
        
        comp_df_all = df[['Specialization', 'Salary']].copy()
        comp_df_all['Group'] = 'Overall Population'
        
        final_comp_df = pd.concat([comp_df_all, comp_df_selected])
        
        fig = px.box(
            final_comp_df,
            x="Specialization",
            y="Salary",
            color="Group",
            title="Overall Population vs. Your Filtered Profile",
            color_discrete_map={
                'Overall Population': 'lightgray',
                'Filtered Profile': '#EF553B'
            },
            category_orders={"Group": ["Overall Population", "Filtered Profile"]}
        )
        
        fig.update_layout(
            boxmode='group',
            height=500,
            modebar=dict(
                bgcolor="rgba(0,0,0,0.3)",
                color="white",
                activecolor="#00D9FF"
            )
        )

        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'personality_box_plot',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
        
        st.plotly_chart(fig, use_container_width=True, config=config)

        st.subheader("🧠 Average Personality Profile of Filtered Group")
        
        avg_traits = filtered_df[list(TRAITS.keys())].mean().reset_index()
        avg_traits.columns = ['Trait', 'Score']
        avg_traits['Trait Name'] = avg_traits['Trait'].map(TRAITS)
        
        fig_radar = px.line_polar(
            avg_traits,
            r='Score',
            theta='Trait Name',
            line_close=True,
            range_r=[-2, 2],
            title="Personality Profile of Filtered Group"
        )
        fig_radar.update_traces(fill='toself')
        fig_radar.update_layout(
            modebar=dict(
                bgcolor="rgba(0,0,0,0.3)",
                color="white",
                activecolor="#00D9FF"
            )
        )

        config_radar = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'personality_radar',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
        
        st.plotly_chart(fig_radar, use_container_width=True, config=config_radar)
        
        st.info(
            """
            **Radar Chart Tips:**
            - If the shape extends outward from center, that trait is dominant in your selected group.
            - If it's sunken toward the center, the group has weak or negative scores in that trait.
            """
        )

if __name__ == "__main__":
    main()