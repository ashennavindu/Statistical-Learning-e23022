# data-inspector-tool

A modular Python toolkit designed for data sanitization, exploration, and interactive visualization within Google Colab environments. Built and demonstrated using the **Adult Census Income Dataset**, this tool automates common preprocessing tasks such as data sanitization, missing value imputation, outlier detection, and advanced statistical association mapping.

---

## Features

- **Intelligent Data Loading:** Automatically handles common null strings (e.g., `'?'`, `'N/A'`, `'NULL'`) and attempts auto-conversion of columns to correct numeric types.
- **Comprehensive Inspection:** Quickly view data dimensions, column type breakdowns, and statistical summaries for both numerical and categorical data.
- **Automated Cleaning:**
  - Identify and impute missing values using `mean`, `median`, `mode`, or `constant` strategies.
  - Remove exact duplicate rows and specific outliers using IQR logic.
  - Interactive row and column deletion.
- **Advanced Scaling & Encoding:**
  - Numeric: Min-Max, Standard (Z-score), and Robust scaling.
  - Categorical: One-Hot, Ordinal, and Uniform encoding.
- **Interactive Visualizations:** Powered by Plotly, including horizontal violin plots, scatter plots, histograms, and grouped bar charts.
- **Deep Statistical Insights:**
  - Pearson Correlation heatmaps for numeric data.
  - Cramér's V heatmaps for categorical associations.
  - Unified Association Heatmaps combining Numeric and Categorical data (using Point-Biserial and Eta/ANOVA).
- **Advanced Statistical Tests:**
  - MANOVA Wilks' Lambda for mean homogeneity.
  - Box's M-Test for covariance homogeneity.
  - Multivariate Ljung-Box for row independence.
- **Dimensionality Reduction:**
  - PCA Dashboard (Loadings, Eigenvalues, T², Q/SPE).
  - Factor Analysis Dashboard (Communality, Uniqueness, Latent Scores).

---

## Dataset Used

**Adult Census Income Dataset (UCI)**
- **Source:** https://www.kaggle.com/datasets/uciml/adult-census-income
- **File:** `adult.csv`
- **Rows:** 32,561 | **Columns:** 15

| Type | Columns |
|---|---|
| Numeric | age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week |
| Categorical | workclass, education, marital-status, occupation, relationship, race, sex, native-country, income |

> This dataset contains real `'?'` garbage strings, mixed numeric and categorical columns, natural outliers in `capital-gain` and `capital-loss`, and missing values — making it an ideal real-world test for all features of this tool.

---

## Installation

```bash
# Basic installation
pip install "git+https://github.com/fratlеader/data-inspector-tool.git"

# Install all dependencies
pip install numpy pandas scikit-learn scipy pydantic plotly
```

---

## Quick Start (Google Colab)

Upload `core.py` to your Colab session before running.

### 1. Data Cleaning and Imputation

```python
from core import DataInspector

di = DataInspector()

# Step 1: Upload adult.csv (interactive file picker in Colab)
# '?' strings are automatically converted to NaN
di.upload_data()

# Step 2: Impute missing values using mode (best for categorical columns)
di.handle_missing_values(strategy='mode')

# Step 3: Remove duplicate rows
di.remove_duplicates()
```

### 2. Exploratory Data Analysis

```python
# View data dimensions, column types, and first 20 rows
di.get_summary()

# Visualize numeric distributions (Violin + Scatter + Histogram)
di.plot_numerical(['age', 'hours-per-week', 'education-num'])

# Visualize categorical frequency with percentage labels
di.plot_categorical(['income', 'sex', 'workclass', 'education'])
```

### 3. Relationship Plots

```python
# Auto-selects chart type based on column data types
di.plot_relationship('age', 'hours-per-week')       # Num vs Num → Scatter + OLS
di.plot_relationship('income', 'age')               # Cat vs Num → Box plot
di.plot_relationship('sex', 'income')               # Cat vs Cat → Grouped bar
```

### 4. Feature Engineering & Normalization

```python
# Scale numeric columns using MinMax [0, 1]
num_scaled = di.extract_normalized_numeric_data(method='minmax')

# Encode categorical columns using Ordinal encoding
cat_encoded = di.extract_normalized_categorical_data(method='ordinal')

# Create a single merged DataFrame ready for ML training
final_df = di.create_normalized_data_df()
```

### 5. Advanced Correlation Mapping

```python
# Pearson r heatmap (numeric only)
di.plot_numerical_correlation()

# Cramér's V heatmap (categorical only)
di.plot_categorical_correlation()

# Unified heatmap for ALL column type pairs
di.plot_all_associations_heatmap()
```

### 6. Statistical Tests

```python
numeric_cols = ['age', 'education-num', 'hours-per-week']

# MANOVA Wilks' Lambda — mean homogeneity across data chunks
di.test_constant_mean(columns=numeric_cols, chunks=10)

# Box's M-test — covariance homogeneity
di.test_constant_covariance(columns=numeric_cols, chunks=5)

# Multivariate Ljung-Box — row-to-row independence
di.test_row_independence(columns=numeric_cols)
```

### 7. PCA & Factor Analysis Dashboards

```python
# PCA — 2x3 dashboard: Loadings | Eigenvalues | Variance | T² | Q(SPE)
pca_results = di.compute_empirical_pca(columns=numeric_cols, show_plot=True)

# Factor Analysis — 2x2 dashboard: Loadings | Communality | Uniqueness | Scores
fa_results = di.compute_empirical_fa(k=2, columns=numeric_cols, show_plot=True)
```

### 8. Custom Chart Generation (PlottingMethods)

```python
from core import PlottingMethods
pm = PlottingMethods()

# Bar chart — income distribution
income_counts = di.df['income'].value_counts().reset_index()
income_counts.columns = ['income', 'count']
bar = pm.plot_bar_chart(
    x='income', y='count',
    title='Income Distribution (<=50K vs >50K)',
    barmode='group',
    data=income_counts.to_json(orient='records')
)
pm.display_image(bar)

# Pie chart — gender breakdown
sex_counts = di.df['sex'].value_counts().reset_index()
sex_counts.columns = ['sex', 'count']
pie = pm.plot_pie_chart(
    names='sex', values='count',
    title='Gender Breakdown',
    data=sex_counts.to_json(orient='records')
)
pm.display_image(pie)

# Histogram — age distribution
hist = pm.plot_histogram(
    x='age',
    title='Age Distribution of Census Respondents',
    data=di.df[['age']].to_json(orient='records')
)
pm.display_image(hist)

# Heatmap — avg hours-per-week by education and sex
heatmap = pm.plot_heat_map(
    values='hours-per-week',
    index='education',
    columns='sex',
    aggregade_method='mean',
    title='Avg Hours-per-Week by Education & Sex',
    data=di.df[['education', 'sex', 'hours-per-week']].to_json(orient='records')
)
pm.display_image(heatmap)
```

### 9. Export Cleaned Data

```python
di.export_cleaned_data(filename='adult_cleaned.csv')
```

---

## author
W.N ASHEN
  
