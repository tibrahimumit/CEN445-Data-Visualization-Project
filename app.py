"""Landing page for the Engineering Salary Dashboard."""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Engineering Salary Dashboard", layout="wide")

    # Header
    st.title("🎓 Engineering Graduate Salary Dashboard")
    st.markdown("### CEN445 Data Visualization Project — 2025/26")
    
    # Team section with better formatting
    st.info(
        """
        **👥 Project Team**  
        
        • **İbrahim Ümit TAŞ** — 2021555061  
        • **Emre Karaman** — 2021555034  
        • **Abdurrahman Gülmez** — 2021555028
        """
    )
    
    st.divider()
    
    # Introduction
    st.markdown(
        """
        ## 📊 About This Dashboard
        
        This interactive dashboard visualizes the **Engineering Graduate Salary Dataset** from India, 
        featuring **9 advanced data visualizations** that reveal how academic performance, personality traits, 
        specialization, and geography influence graduate salaries.
        
        Use the **sidebar navigation** (☰) to explore each visualization.
        """
    )
    
    # Visualization cards - Row 1
    st.subheader("🎯 Advanced Visualizations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            ### 1. 📈 Parallel Coordinates
            
            Multi-dimensional view of:
            - Academic scores (10th/12th, GPA)
            - Aptitude tests (English, Quant)
            - Salary levels (color-coded)
            
            **Interactive brushing** to filter cohorts.
            """
        )
    
    with col2:
        st.markdown(
            """
            ### 2. 🌊 Salary Sankey
            
            Flow diagram showing:
            - Degree → Specialization → College Tier → Salary
            
            Discover which paths lead to high salaries (Low / Medium / High brackets).
            """
        )
    
    with col3:
        st.markdown(
            """
            ### 3. 🗺️ India Salary Map
            
            State-level choropleth of average salaries.
            
            **Dynamic colors:**
            - 🔵 Blue (male-only)
            - 🩷 Pink (female-only)
            - 🌈 Neutral (both)
            """
        )
    
    # Row 2
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown(
            """
            ### 4. 📊 Specialization Bar Chart
            
            Student distribution by specialization.
            
            **Features:** Icons, glowing hover effects, dynamic colors.
            """
        )
    
    with col5:
        st.markdown(
            """
            ### 5. 🧬 GPA & Skills Clustering
            
            Correlation heatmap with hierarchical clustering.
            
            See which skills and academic scores **cluster together** (e.g., GPA + 10th/12th scores).
            """
        )
    
    with col6:
        st.markdown(
            """
            ### 6. 🔬 K-Means Talent Clustering
            
            Machine learning-powered segmentation.
            
            Groups graduates into **talent profiles** based on skills, then visualizes with PCA in 2D.
            """
        )
    
    # Row 3
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown(
            """
            ### 7. 🎛️ Personality Profile & Box Plot
            
            Filter by **Big Five personality traits**, then compare salaries.
            
            Includes box plot and radar chart showing personality profile.
            """
        )
    
    with col8:
        st.markdown(
            """
            ### 8. 🗺️ Multi-Dimensional Heatmap
            
            Density heatmap (bitmap) + Treemap.
            
            Choose any 2 dimensions (e.g., Year × Specialization) and see aggregate values.
            """
        )
    
    with col9:
        st.markdown(
            """
            ### 9. 📡 360° Radar Chart
            
            Candidate profile analysis combining:
            - Personality traits
            - Academic scores
            - Aptitude tests
            
            Compare by Specialization, Success Tier, or Gender.
            """
        )
    
    st.divider()
    
    st.success("💡 **Get Started**: Open the sidebar menu and select a visualization page.")


if __name__ == "__main__":
    main()
