# Engineering Salary Dashboard

Interactive Streamlit dashboard visualizing the engineering graduate salary dataset (India) with advanced charts.

## Project Team

- **İbrahim Ümit TAŞ** — 2021555061
- **Emre Karaman** — 2021555034
- **Abdurrahman Gülmez** — 2021555028

**Course**: CEN445 Data Visualization (2025–26)

---

## Features

This dashboard includes **9 advanced interactive visualizations**:

### Core Visualizations

1. **Parallel Coordinates**  
   Multi-dimensional view of academic scores (10th/12th grade, GPA), aptitude tests (English, Quant), and salary. Interactive brushing and filtering by placement status, degree, and specialization. Electric color scale with adjustable line density.

2. **Salary Sankey Diagram**  
   Flow diagram showing: Degree → Specialization → College Tier → Salary Category (Low/Medium/High). Interactive filters reveal which academic paths lead to higher salaries.

3. **India Salary Map (Choropleth)**  
   State-level average salary visualization with **dynamic color palettes**:
   - Blue tones when filtering by male only
   - Pink tones when filtering by female only
   - Neutral (Viridis) when both/neither selected
   
   Includes filters for Degree, Specialization, and Gender.

### Advanced Analytics

4. **GPA & Skills Clustering**  
   Correlation heatmap with hierarchical clustering. Automatically groups related features (e.g., academic scores cluster together, aptitude tests form another block). Interactive feature selection and filter by Degree/Specialization.

5. **K-Means Talent Clustering**  
   Machine learning-powered segmentation using K-Means clustering + PCA dimensionality reduction. Groups graduates into talent profiles based on skills (English, Logical, Quant, Domain, Programming, GPA). Visualizes clusters in 2D with interactive controls for cluster count and quality metrics (silhouette score, Davies-Bouldin).

6. **Specialization Bar Chart**  
   Student distribution across specializations with:
   - Category icons (CS, Mechanical, etc.)
   - Glowing hover effects
   - Dynamic color gradients
   - Filters for Degree, College Tier, Placement Status, Gender

### Personality & Profile Analysis

7. **Personality Profile & Box Plot**  
   Multi-filter analysis using Big Five personality traits (Conscientiousness, Agreeableness, Extraversion, Neuroticism, Openness). Compare filtered personality profiles against general population with side-by-side box plots. Includes radar chart showing personality profile shape.

8. **Treemap**  
   Hierarchical nested rectangles showing proportional data (e.g., Degree → Specialization → Gender). Choose any 2 dimensions and see aggregate values with customizable visualization.

9. **360° Radar Chart**  
   Comprehensive candidate profile combining:
   - Personality traits (Big Five)
   - Academic scores (GPA, 10th/12th grade)
   - Aptitude tests (Quant, Logical, English, Programming)
   
   All features normalized to 0-100 scale. Compare by Specialization, Success Tier (top 25% vs. standard income), or Gender. Customizable axis selection.

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/tibrahimumit/CEN445-Data-Visualization-Project.git
cd CEN445-Data-Visualization-Project

# Create virtual environment
python3 -m venv .venv

# Activate venv (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## Dataset

- **Name**: Engineering Graduate Salary Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/lovishbansal123/engineering-graduate-salary)
- **Size**: 2,998 students with multiple features
- **Column Counts**: 34 columns
- **Content**: Academic performance, skills, personality traits, college details, and salary information

---

## Technologies

- **Streamlit** — Web framework for interactive dashboards
- **Plotly** — Advanced charting library (parallel coordinates, Sankey, choropleth, radar, etc.)
- **Pandas** — Data manipulation and aggregation
- **Scikit-learn** — Machine learning (K-Means clustering, PCA, StandardScaler, MinMaxScaler)
- **SciPy** — Scientific computing (hierarchical clustering for correlation heatmap)

---

## Team Contributions

- **İbrahim Ümit Taş**: Charts 1-3 (Parallel Coordinates, Sankey, India Map), Streamlit application integration and deployment, landing page design
- **Emre Karaman**: Charts 4-6 (GPA Clustering Heatmap, K-Means Clustering, Bar Chart)
- **Abdurrahman Gülmez**: Charts 7-9 (Box Plot, Heatmap/Treemap, Radar Chart), data preprocessing and cleaning
- **Report Writing**: Collaborative effort by all team members

---

## Assignment Report

**One-page assignment report**: [`CEN445_Report.docx`](CEN445_Report.docx) | [`CEN445_Report.pdf`](CEN445_Report.pdf)

The report summarizes:
- Dataset information and source
- Analysis goals and objectives
- All 9 visualization techniques used
- Key insights and findings
- Technical implementation details
- Individual team member contributions

---

## License

Academic project for CEN445 course. Not licensed for commercial use.
