from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List
from pydantic import BaseModel, ValidationError, field_validator
 
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files
import scipy
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway, multivariate_normal
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import FactorAnalysis
 
 
class DataInspector:
    """
    A comprehensive data cleaning and exploration tool for Google Colab.
    Provides interactive visualizations using Plotly and robust data sanitization.
    """
 
    def __init__(self):
        self.df = None
        self.numeric_df = None
        self.categorical_df = None
        self.categorical_normalized_df = None
        self.normalized_data_df = None
        self.numeric_normalized_df = None
 
    def upload_data(self):
        """
        Prompts user to upload a CSV, handles common null strings,
        and attempts to auto-convert columns to their correct numeric types.
        """
        uploaded = files.upload()
        if not uploaded:
            return print("No file uploaded.")
 
        file_name = list(uploaded.keys())[0]
        self.df = pd.read_csv(io.BytesIO(uploaded[file_name]),
                              na_values=['?', 'n/a', 'N/A', 'NULL', 'null', ' '])
        self.df['count'] = 1
 
        for col in self.df.columns:
            numeric_col = pd.to_numeric(self.df[col], errors='coerce')
            if not numeric_col.isna().all():
                self.df[col] = numeric_col
 
        print(f"\n✅ File '{file_name}' loaded and types sanitized!")
 
    def get_summary(self):
        """
        Prints data dimensions and column type breakdown.
        Displays the first 20 rows of the DataFrame.
        """
        if self.df is None: return print("Error: No data loaded.")
 
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
 
        print(f"--- Data Summary ---")
        print(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}")
        print(f"Numerical ({len(num_cols)}): {num_cols}")
        print(f"Categorical ({len(cat_cols)}): {cat_cols}")
        display(self.df.head(20))
 
    def show_missing_data(self):
        """
        Filters the DataFrame to show only rows containing at least one missing (NaN) value.
        """
        if self.df is None: return
        missing_mask = self.df.isnull().any(axis=1) | (self.df == "").any(axis=1)
        missing_rows = self.df[missing_mask]
 
        if missing_rows.empty:
            print("✨ No missing data found!")
        else:
            print(f"🔍 Found {len(missing_rows)} rows with missing values:")
            display(missing_rows)
 
    def delete_rows(self):
        """
        Deletes rows based on a comma-separated list of indices provided via user input.
        """
        if self.df is None: return
        try:
            user_input = input("Enter row indices to delete (e.g., 1, 3, 15): ")
            indices_to_drop = [int(i.strip()) for i in user_input.split(',') if i.strip().isdigit()]
            existing_indices = [i for i in indices_to_drop if i in self.df.index]
            self.df = self.df.drop(index=existing_indices).reset_index(drop=True)
            print(f"🗑️ Deleted {len(existing_indices)} rows. New count: {len(self.df)}")
        except Exception as e:
            print(f"❌ Error: {e}")
 
    def delete_columns(self):
        """
        Deletes columns based on a comma-separated list of names provided via user input.
        """
        if self.df is None:
            return print("No data loaded.")
        try:
            print(f"Current columns: {', '.join(self.df.columns)}")
            user_input = input("Enter column names to delete (e.g., Column1, Column2): ")
            cols_to_drop = [c.strip() for c in user_input.split(',')]
            existing_cols = [c for c in cols_to_drop if c in self.df.columns]
            if not existing_cols:
                return print("⚠️ None of the provided column names were found.")
            self.df = self.df.drop(columns=existing_cols)
            print(f"🗑️ Deleted {len(existing_cols)} columns. Remaining: {len(self.df.columns)}")
        except Exception as e:
            print(f"❌ Error: {e}")
 
    def handle_missing_values(self, columns=None, strategy='median', fill_value=None):
        """
        Imputes missing values in specified columns.
 
        Parameters:
        - columns: List of strings. If None, applies to all columns with NaNs.
        - strategy: 'mean', 'median', 'mode', or 'constant'.
        - fill_value: Used only if strategy is 'constant'.
        """
        if self.df is None: return
        target_cols = columns if columns else self.df.columns[self.df.isnull().any()].tolist()
 
        for col in target_cols:
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)
 
        print(f"🛠️ Imputation complete using '{strategy}' strategy for: {target_cols}")
 
    def remove_duplicates(self):
        """
        Identifies and removes exact duplicate rows from the DataFrame.
        """
        if self.df is None: return
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        dropped = initial_count - len(self.df)
        print(f"✨ Removed {dropped} duplicate rows. New row count: {len(self.df)}")
 
    def export_cleaned_data(self, filename='cleaned_data.csv'):
        """
        Converts the current DataFrame to CSV and triggers a browser download in Colab.
        """
        if self.df is None: return
        self.df.to_csv(filename, index=False)
        files.download(filename)
        print(f"💾 '{filename}' has been generated and download triggered.")
 
    def column_details(self):
        """
        Iterates through all columns to show numeric ranges or categorical unique value counts.
        """
        if self.df is None: return
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"🔹 {col} (Numeric): Range [{self.df[col].min()} to {self.df[col].max()}]")
            else:
                print(f"🔸 {col} (Categorical): {self.df[col].nunique()} unique values")
 
    def get_categorical_summary(self):
        """
        Generates a detailed statistical summary for categorical columns,
        including unique counts, mode, and mode frequency.
        """
        if self.df is None: return
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("No categorical columns found.")
        summary = cat_df.describe().T[['unique', 'top', 'freq']]
        print("--- Categorical Deep Dive ---")
        display(summary)
 
    def extract_numeric_data(self):
        """
        Filters the DataFrame to include only numeric columns.
        """
        if self.df is None: return print("Error: No data loaded.")
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        return self.numeric_df
 
    def extract_categorical_data(self):
        """
        Filters the DataFrame to include only categorical (non-numeric) columns.
        """
        if self.df is None: return print("Error: No data loaded.")
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])
        return self.categorical_df
 
    def extract_normalized_numeric_data(self, method='minmax'):
        """
        Extracts numerical columns and scales them using the specified method.
 
        Parameters:
        - method: 'minmax'   → scales to [0, 1]
                  'standard' → Z-score (mean=0, std=1)
                  'robust'   → IQR-based, handles outliers well
        """
        if self.df is None: return print("Error: No data loaded.")
        num_df = self.df.select_dtypes(include=[np.number]).copy()
 
        if num_df.empty:
            print("⚠️ No numerical columns found to scale.")
            self.numeric_normalized_df = pd.DataFrame()
            return self.numeric_normalized_df
 
        if num_df.isnull().any().any():
            print("ℹ️ Imputing with column medians before scaling...")
            num_df = num_df.fillna(num_df.median())
 
        method_lower = method.lower().strip()
        if method_lower == 'minmax':
            scaler = MinMaxScaler()
        elif method_lower == 'standard':
            scaler = StandardScaler()
        elif method_lower == 'robust':
            scaler = RobustScaler()
        else:
            print(f"❌ Unknown method '{method}'. Defaulting to 'minmax'.")
            return self.extract_normalized_numeric_data(method='minmax')
 
        scaled_data = scaler.fit_transform(num_df)
        self.numeric_normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)
        print(f"✨ Scaled numerical data using '{method_lower}' method.")
        return self.numeric_normalized_df
 
    def extract_normalized_categorical_data(self, method='uniform'):
        """
        Extracts categorical columns and applies the specified encoding method.
 
        Parameters:
        - method: 'uniform'       → codes scaled to [0, 1]
                  'ordinal'       → integer codes (0, 1, 2 ...)
                  'onehot'        → binary columns per category
                  'minmax_ordinal'→ ordinal then MinMax scaled
        """
        if self.df is None: return print("Error: No data loaded.")
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
 
        if cat_df.empty:
            print("⚠️ No categorical columns found.")
            self.categorical_normalized_df = pd.DataFrame()
            return self.categorical_normalized_df
 
        method_lower = method.lower().strip()
 
        if method_lower == 'uniform':
            for col in cat_df.columns:
                codes = cat_df[col].astype('category').cat.codes
                max_code = codes.max()
                cat_df[col] = codes / max_code if max_code > 0 else 0.0
            self.categorical_normalized_df = cat_df
 
        elif method_lower == 'ordinal':
            encoder = OrdinalEncoder()
            encoded_data = encoder.fit_transform(cat_df.fillna("Missing"))
            self.categorical_normalized_df = pd.DataFrame(encoded_data, columns=cat_df.columns, index=cat_df.index)
 
        elif method_lower == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(cat_df.fillna("Missing"))
            feature_names = encoder.get_feature_names_out(cat_df.columns)
            self.categorical_normalized_df = pd.DataFrame(encoded_data, columns=feature_names, index=cat_df.index)
 
        elif method_lower == 'minmax_ordinal':
            encoder = OrdinalEncoder()
            scaler = MinMaxScaler()
            encoded_data = encoder.fit_transform(cat_df.fillna("Missing"))
            scaled_data = scaler.fit_transform(encoded_data)
            self.categorical_normalized_df = pd.DataFrame(scaled_data, columns=cat_df.columns, index=cat_df.index)
 
        else:
            print(f"❌ Unknown method '{method}'. Defaulting to 'uniform'.")
            return self.extract_normalized_categorical_data(method='uniform')
 
        print(f"✨ Encoded categorical data using '{method_lower}' method.")
        return self.categorical_normalized_df
 
    def create_normalized_data_df(self):
        """
        Creates a single DataFrame containing original numeric columns
        merged side-by-side with normalized categorical columns.
        """
        if self.df is None: return print("Error: No data loaded.")
        num_df = self.extract_numeric_data()
        cat_norm_df = self.extract_normalized_categorical_data()
 
        if cat_norm_df is None or (isinstance(cat_norm_df, pd.DataFrame) and cat_norm_df.empty):
            print("ℹ️ No categorical columns. Returning numeric DataFrame only.")
            self.normalized_data_df = num_df
            return self.normalized_data_df
 
        if num_df is None or (isinstance(num_df, pd.DataFrame) and num_df.empty):
            print("ℹ️ No numeric columns. Returning categorical DataFrame only.")
            self.normalized_data_df = cat_norm_df
            return self.normalized_data_df
 
        self.normalized_data_df = pd.concat([num_df, cat_norm_df], axis=1)
        print(f"✅ Created merged DataFrame with {self.normalized_data_df.shape[1]} columns.")
        return self.normalized_data_df
 
    def plot_numerical(self, column_names):
        """
        Generates a 3-panel interactive subplot for each numeric column:
        Horizontal Violin/Box | Scatter (Index vs Value) | Histogram.
        """
        if self.df is None: return
        if isinstance(column_names, str): column_names = [column_names]
        valid_cols = [c for c in column_names if c in self.df.columns
                      and pd.api.types.is_numeric_dtype(self.df[c])]
 
        for col in valid_cols:
            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=(f"Horizontal Violin/Box: {col}",
                                                f"Scatter Plot: {col}",
                                                f"Distribution: {col}"))
            fig.add_trace(go.Violin(x=self.df[col], box_visible=True, meanline_visible=True,
                                    name=col, orientation='h', line_color='lightseagreen'), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.df[col], mode='markers',
                                     marker=dict(opacity=0.5, color='royalblue'), name=col), row=1, col=2)
            fig.add_trace(go.Histogram(x=self.df[col], name=col, marker_color='indianred'), row=1, col=3)
            fig.update_layout(height=450, title_text=f"<b>Statistical Analysis: {col}</b>",
                               showlegend=False, template="plotly_white")
            fig.update_xaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=2)
            fig.update_xaxes(title_text="Value", row=1, col=3)
            fig.show()
 
    def plot_categorical(self, column_names):
        """
        Generates interactive Bar charts for categorical columns
        showing raw counts and percentage labels.
        """
        if self.df is None: return
        if isinstance(column_names, str): column_names = [column_names]
 
        for col in column_names:
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, 'count']
            counts['percentage'] = (counts['count'] / counts['count'].sum() * 100).round(1).astype(str) + '%'
            fig = px.bar(counts, x=col, y='count', text='percentage',
                         title=f"Frequency: {col}", color=col,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.show()
 
    def handle_outliers(self, columns=None, find_and_delete=False):
        """
        Flags outliers using the IQR method.
        Optionally deletes the flagged rows if find_and_delete=True.
        """
        if self.df is None: return
        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        all_outliers = set()
 
        for col in target_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))]
            all_outliers.update(outliers.index.tolist())
            print(f"🚨 {col}: Found {len(outliers)} outliers.")
 
        if all_outliers:
            display(self.df.loc[list(all_outliers)])
            if find_and_delete:
                self.df = self.df.drop(index=list(all_outliers)).reset_index(drop=True)
                print(f"🗑️ Deleted {len(all_outliers)} outlier rows.")
 
    def plot_relationship(self, col1, col2):
        """
        Intelligently selects the best plot based on column data types:
        - Num vs Num : Scatter with OLS Trendline
        - Cat vs Num : Box plot with all data points
        - Cat vs Cat : Grouped Bar chart
        """
        if self.df is None: return
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
 
        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols",
                             title=f"Correlation: {col1} vs {col2}")
        elif is_num1 != is_num2:
            num, cat = (col1, col2) if is_num1 else (col2, col1)
            fig = px.box(self.df, x=cat, y=num, points="all", color=cat,
                         title=f"Distribution of {num} by {cat}")
        else:
            fig = px.histogram(self.df, x=col1, color=col2, barmode="group",
                               title=f"Relationship: {col1} vs {col2}")
        fig.show()
 
    def plot_numerical_correlation(self):
        """
        Displays an interactive Pearson Correlation Heatmap for all numeric columns.
        """
        if self.df is None: return
        numerical_df = self.df.select_dtypes(include=[np.number])
        corr = numerical_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Pearson Correlation Heatmap")
        fig.show()
 
    def plot_categorical_correlation(self):
        """
        Calculates Cramér's V association matrix for all categorical columns
        and displays it as an interactive Plotly Heatmap.
        """
        if self.df is None: return print("Error: No data loaded.")
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("⚠️ No categorical columns found.")
 
        cols = cat_df.columns
        n_cols = len(cols)
        corr_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
 
        for i in range(n_cols):
            for j in range(i, n_cols):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    corr_matrix.loc[col1, col2] = 1.0
                    continue
                confusion_matrix = pd.crosstab(cat_df[col1], cat_df[col2])
                if confusion_matrix.size == 0 or min(confusion_matrix.shape) <= 1:
                    corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = 0.0
                    continue
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                v = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1))) if n > 0 else 0.0
                corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = v
 
        print("--- Cramér's V Association Matrix ---")
        display(corr_matrix.round(3))
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="<b>Cramér's V Categorical Association Heatmap</b>",
                        labels=dict(color="Cramér's V"))
        fig.update_layout(height=max(400, n_cols * 80), width=max(500, n_cols * 80),
                          template="plotly_white")
        fig.show()
        return corr_matrix
 
    def correlate_num_to_cat(self):
        """
        Computes associations between all numeric and categorical columns.
        - Binary categories   → Point-Biserial correlation (-1 to 1)
        - Multi-class categories → Eta from ANOVA (0 to 1)
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
 
        if len(num_cols) == 0 or len(cat_cols) == 0:
            print("⚠️ Requires both numerical and categorical columns.")
            return pd.DataFrame()
 
        results = []
        for cat in cat_cols:
            for num in num_cols:
                valid_data = self.df[[cat, num]].dropna()
                if valid_data.empty: continue
                categories = valid_data[cat].unique()
                if len(categories) < 2: continue
 
                if len(categories) == 2:
                    binary_cat = pd.get_dummies(valid_data[cat], drop_first=True).iloc[:, 0]
                    corr, p_val = pointbiserialr(binary_cat, valid_data[num])
                    results.append({'Categorical': cat, 'Numerical': num,
                                    'Type': 'Point-Biserial (Binary)',
                                    'Correlation': round(corr, 3), 'P-Value': round(p_val, 4)})
                else:
                    groups = [g for g in [valid_data[valid_data[cat] == val][num]
                               for val in categories] if len(g) > 0]
                    if len(groups) > 1:
                        f_val, p_val = f_oneway(*groups)
                        grand_mean = valid_data[num].mean()
                        ss_total = ((valid_data[num] - grand_mean) ** 2).sum()
                        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                        eta = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                        results.append({'Categorical': cat, 'Numerical': num,
                                        'Type': 'Eta (Multi-class ANOVA)',
                                        'Correlation': round(eta, 3), 'P-Value': round(p_val, 4)})
        return pd.DataFrame(results)
 
    def plot_all_associations_heatmap(self):
        """
        Creates a unified association matrix covering ALL column type pairs
        and displays it as a single interactive Plotly Heatmap.
        - Numeric-Numeric   : Pearson's r  (absolute value)
        - Cat-Cat           : Cramér's V
        - Mixed (Num-Cat)   : Correlation Ratio Eta
        """
        if self.df is None: return print("Error: No data loaded.")
        cols = self.df.columns
        n_cols = len(cols)
        assoc_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
 
        for i in range(n_cols):
            for j in range(i, n_cols):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    assoc_matrix.loc[col1, col2] = 1.0
                    continue
                valid_data = self.df[[col1, col2]].dropna()
                if valid_data.empty: continue
 
                is_num1 = pd.api.types.is_numeric_dtype(valid_data[col1])
                is_num2 = pd.api.types.is_numeric_dtype(valid_data[col2])
 
                if is_num1 and is_num2:
                    val = abs(valid_data[col1].corr(valid_data[col2], method='pearson'))
                elif not is_num1 and not is_num2:
                    cm = pd.crosstab(valid_data[col1], valid_data[col2])
                    if cm.size > 0 and min(cm.shape) > 1:
                        chi2 = chi2_contingency(cm)[0]
                        n = cm.sum().sum()
                        val = np.sqrt(chi2 / (n * (min(cm.shape) - 1))) if n > 0 else 0.0
                    else:
                        val = 0.0
                else:
                    cat_col, num_col = (col1, col2) if not is_num1 else (col2, col1)
                    categories = valid_data[cat_col].unique()
                    if len(categories) > 1:
                        groups = [g for g in [valid_data[valid_data[cat_col] == c][num_col]
                                   for c in categories] if len(g) > 0]
                        grand_mean = valid_data[num_col].mean()
                        ss_total = ((valid_data[num_col] - grand_mean) ** 2).sum()
                        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                        val = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                    else:
                        val = 0.0
 
                assoc_matrix.loc[col1, col2] = assoc_matrix.loc[col2, col1] = round(val, 3)
 
        print("--- Global Association Matrix ---")
        display(assoc_matrix)
        fig = px.imshow(assoc_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale="viridis",
                        title="<b>Unified Association Heatmap (Numeric & Categorical)</b>",
                        labels=dict(color="Association Strength"))
        fig.update_layout(height=max(500, n_cols * 45), width=max(600, n_cols * 45),
                          template="plotly_white")
        fig.show()
        return assoc_matrix
 
    def test_constant_mean(self, columns: Optional[Sequence[str]] = None, chunks: int = 10) -> Any:
        """
        Evaluates first moment homogeneity across sequential data blocks
        using MANOVA via Wilks' Lambda. Numerically stabilized via
        log-determinant tracking and shrinkage regularization.
 
        Parameters:
        - columns: Sequence of strings. If None, uses all numerical columns.
        - chunks: int, number of sequential blocks to split data into.
        """
        if self.df is None: raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
            if not target_cols: raise ValueError("No numerical columns found.")
        else:
            if isinstance(columns, str): columns = [columns]
            non_numeric = [c for c in columns if c not in self.df.columns
                           or not pd.api.types.is_numeric_dtype(self.df[c])]
            if non_numeric: raise TypeError(f"Non-numerical columns: {non_numeric}")
            target_cols = list(columns)
 
        analysis_df = self.df[target_cols].copy().dropna()
        n, m = len(analysis_df), len(target_cols)
        chunk_size = n // chunks
        if chunk_size < m:
            raise ValueError(f"Chunk size ({chunk_size}) must exceed features ({m}).")
 
        analysis_df['_chunk_label'] = np.minimum(np.arange(n) // chunk_size, chunks - 1)
        global_mean = analysis_df[target_cols].mean().values
        W = np.zeros((m, m))
        B = np.zeros((m, m))
 
        for label, group in analysis_df.groupby('_chunk_label'):
            X_chunk = group[target_cols].values
            chunk_mean = X_chunk.mean(axis=0)
            n_j = len(X_chunk)
            W += np.dot((X_chunk - chunk_mean).T, (X_chunk - chunk_mean))
            mean_diff = (chunk_mean - global_mean).reshape(-1, 1)
            B += n_j * np.dot(mean_diff, mean_diff.T)
 
        epsilon = 1e-6 * np.eye(m)
        W_stable, T_stable = W + epsilon, W + B + epsilon
        sign_W, log_det_W = np.linalg.slogdet(W_stable)
        sign_T, log_det_T = np.linalg.slogdet(T_stable)
        if sign_W <= 0 or sign_T <= 0:
            raise np.linalg.LinAlgError("Matrices are poorly scaled or non-invertible.")
 
        log_wilks = log_det_W - log_det_T
        wilks_lambda = np.exp(log_wilks)
        df_stat = m * (chunks - 1)
        scale_factor = n - 1 - (m + chunks) / 2
        chi2_calc = max(0.0, -scale_factor * log_wilks)
        p_value = 1.0 - scipy.stats.chi2.cdf(chi2_calc, df_stat)
 
        print(f"\n--- MANOVA Mean Homogeneity Test (g={chunks} chunks, m={m} features) ---")
        print(f"Wilks' Lambda (Λ): {wilks_lambda:.5f}")
        print(f"Chi-Square: {chi2_calc:.4f} | df: {df_stat} | P-Value: {p_value:.6f}")
        if p_value > 0.05:
            print("✅ Fail to reject H0. No structural mean drift detected.")
        else:
            print("🚨 Reject H0. Significant mean drift detected across rows.")
 
        return {"wilks_lambda": wilks_lambda, "chi2": chi2_calc, "p_value": p_value, "df": df_stat}
 
    def test_constant_covariance(self, columns: Optional[Sequence[str]] = None, chunks: int = 5) -> Any:
        """
        Evaluates second moment homogeneity across sequential data blocks
        using Box's M-test. Numerically stabilized via shrinkage regularization.
 
        Parameters:
        - columns: Sequence of strings. If None, uses all numerical columns.
        - chunks: int, number of sequential blocks to split data into.
        """
        if self.df is None: raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
            if not target_cols: raise ValueError("No numerical columns found.")
        else:
            if isinstance(columns, str): columns = [columns]
            target_cols = list(columns)
 
        analysis_df = self.df[target_cols].copy().dropna()
        n, m = len(analysis_df), len(target_cols)
        chunk_size = n // chunks
        if chunk_size <= m:
            raise ValueError(f"Degrees of freedom per chunk must exceed dimensions. Reduce chunks.")
 
        analysis_df['_chunk_label'] = np.minimum(np.arange(n) // chunk_size, chunks - 1)
        log_det_S, pooled_S, total_df = 0.0, np.zeros((m, m)), 0
        n_chunks = []
        epsilon = 1e-6 * np.eye(m)
 
        for label, group in analysis_df.groupby('_chunk_label'):
            X_chunk = group[target_cols].values
            n_j = len(X_chunk)
            S_j = np.cov(X_chunk, rowvar=False, ddof=1) + epsilon
            n_chunks.append(n_j)
            df_j = n_j - 1
            pooled_S += df_j * S_j
            total_df += df_j
            sign, logdet = np.linalg.slogdet(S_j)
            if sign <= 0: raise np.linalg.LinAlgError(f"Chunk {label} covariance non-positive definite.")
            log_det_S += df_j * logdet
 
        pooled_S /= total_df
        sign_p, log_det_Sp = np.linalg.slogdet(pooled_S)
        if sign_p <= 0: raise np.linalg.LinAlgError("Pooled covariance is non-positive definite.")
 
        M = total_df * log_det_Sp - log_det_S
        sum_inv_df = sum(1.0 / (nj - 1) for nj in n_chunks)
        inv_total_df = 1.0 / total_df
        C = (sum_inv_df - inv_total_df) * ((2.0 * m**2 + 3.0 * m - 1.0) / (6.0 * (m + 1.0) * (chunks - 1.0)))
        chi2_calc = max(0.0, M * (1.0 - C))
        df_stat = int((m * (m + 1) * (chunks - 1)) / 2.0)
        p_value = 1.0 - scipy.stats.chi2.cdf(chi2_calc, df_stat)
 
        print(f"\n--- Box's M Covariance Homogeneity Test (g={chunks} chunks, m={m} features) ---")
        print(f"Box's M: {M:.4f} | Chi-Square: {chi2_calc:.4f} | df: {df_stat} | P-Value: {p_value:.6f}")
        if p_value > 0.001:
            print("✅ Fail to reject H0. Covariance structure is stable.")
        else:
            print("🚨 Reject H0. Significant covariance drift detected.")
 
        return {"M": M, "chi2": chi2_calc, "p_value": p_value, "df": df_stat}
 
    def test_row_independence(self, columns: Optional[Sequence[str]] = None, max_lag: Optional[int] = None) -> Any:
        """
        Evaluates row-to-row statistical independence using the
        Multivariate Ljung-Box Portmanteau test.
 
        Parameters:
        - columns: Sequence of strings. If None, uses all numerical columns.
        - max_lag: int, maximum lag to check. Defaults to ln(n).
        """
        if self.df is None: raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
            if not target_cols: raise ValueError("No numerical columns found.")
        else:
            if isinstance(columns, str): columns = [columns]
            target_cols = list(columns)
 
        analysis_df = self.df[target_cols].copy().dropna()
        n, m = len(analysis_df), len(target_cols)
        if max_lag is None: max_lag = int(np.ceil(np.log(n)))
        if max_lag >= n: raise ValueError(f"Max lag ({max_lag}) must be less than sample size ({n}).")
 
        X = analysis_df[target_cols].values
        X_centered = X - X.mean(axis=0)
        epsilon = 1e-6 * np.eye(m)
        Gamma_0 = (np.dot(X_centered.T, X_centered) / n) + epsilon
 
        try:
            inv_Gamma_0 = np.linalg.inv(Gamma_0)
        except np.linalg.LinAlgError:
            inv_Gamma_0 = np.linalg.pinv(Gamma_0)
 
        Q_m = 0.0
        for k in range(1, max_lag + 1):
            Gamma_k = np.dot(X_centered[k:].T, X_centered[:-k]) / n
            M_k = np.dot(np.dot(np.dot(Gamma_k.T, inv_Gamma_0), Gamma_k), inv_Gamma_0)
            Q_m += np.trace(M_k) / (n - k)
 
        Q_m = max(0.0, Q_m * (n ** 2))
        df_stat = (m ** 2) * max_lag
        p_value = 1.0 - scipy.stats.chi2.cdf(Q_m, df_stat)
 
        print(f"\n--- Multivariate Ljung-Box Test (Lags = {max_lag}) ---")
        print(f"Q_m: {Q_m:.4f} | df: {df_stat} | P-Value: {p_value:.6f}")
        if p_value > 0.05:
            print("✅ Fail to reject H0. Rows are statistically independent.")
        else:
            print("🚨 Reject H0. Significant row-to-row serial dependency detected.")
 
        return {"Q_m": Q_m, "p_value": p_value, "df": df_stat}
 
    def estimate_joint_normal(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Fits a parametric Multivariate Normal Distribution X_i ~ N(mu_hat, S)
        using unbiased MLE with Bessel's correction.
        """
        if self.df is None: raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
        else:
            target_cols = list(columns)
 
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("Sample size n must be larger than feature dimensions m.")
 
        mu_hat = np.mean(X, axis=0)
        epsilon = 1e-6 * np.eye(m)
        S_matrix = np.cov(X, rowvar=False, ddof=1) + epsilon
        joint_dist = multivariate_normal(mean=mu_hat, cov=S_matrix, allow_singular=True)
        log_likelihoods = joint_dist.logpdf(X)
        total_log_likelihood = np.sum(log_likelihoods)
        k_parameters = m + (m * (m + 1)) // 2
        aic = 2 * k_parameters - 2 * total_log_likelihood
 
        print(f"\n--- X_i ~ N(mu_hat, S) | m={m} features, n={n} samples ---")
        for col, val in zip(target_cols, mu_hat):
            print(f"  • {col}: {val:.4f}")
        print(f"Log-Likelihood: {total_log_likelihood:.4f} | AIC: {aic:.4f}")
 
        return {"mean_vector": mu_hat, "covariance_matrix": S_matrix,
                "log_likelihood": total_log_likelihood, "aic": aic,
                "distribution_object": joint_dist, "features": target_cols}
 
    def instantiate_macro_clt_distribution(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Operationalizes the macro-scale CLT model: mu_hat_n ~ N(mu_hat_n, (1/n)*S).
        Models parameter uncertainty rather than raw data variation.
        """
        if self.df is None: raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
        else:
            target_cols = list(columns)
 
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("Sample size n must be larger than feature dimensions m.")
 
        mu_hat = np.mean(X, axis=0)
        S_matrix = np.cov(X, rowvar=False, ddof=1)
        epsilon = 1e-10 * np.eye(m)
        clt_covariance = (1.0 / n) * S_matrix + epsilon
        macro_clt_dist = multivariate_normal(mean=mu_hat, cov=clt_covariance, allow_singular=True)
        total_parameter_variance = np.trace(clt_covariance)
 
        print(f"\n--- μ_hat_n ~ N(μ_hat_n, (1/n)S) | n={n} samples ---")
        for col, val in zip(target_cols, mu_hat):
            print(f"  • {col}: {val:.4f}")
        print(f"Total Trace Error Variance (Tr[(1/n)S]): {total_parameter_variance:.8f}")
 
        return {"mean_vector": mu_hat, "clt_covariance_matrix": clt_covariance,
                "total_parameter_variance": total_parameter_variance,
                "distribution_object": macro_clt_dist, "features": target_cols}
 
    def compute_empirical_pca(self, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Decomposes the empirical covariance matrix S = P * Lambda * P^T via PCA.
        Computes Hotelling's T^2 and Q (SPE) statistics across all truncation
        boundaries k. Generates a 2x3 Plotly Dashboard.
        """
        if self.df is None: raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
        else:
            target_cols = list(columns)
 
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("Sample size must exceed feature dimensions.")
 
        mu_hat = np.mean(X, axis=0)
        X_centered = X - mu_hat
        S_matrix = np.cov(X, rowvar=False, ddof=1)
        eigenvalues, eigenvectors = np.linalg.eigh(S_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        lambda_hat = np.clip(eigenvalues[idx], a_min=1e-15, a_max=None)
        P_hat = eigenvectors[:, idx]
        total_variance = np.sum(lambda_hat)
        if total_variance == 0: raise ValueError("Total variance is zero.")
 
        explained_variance_ratio = lambda_hat / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        unexplained_variance_ratio = 1.0 - cumulative_variance_ratio
        Z_scores = np.dot(X_centered, P_hat)
        S_Z = np.cov(Z_scores, rowvar=False, ddof=1)
 
        k_range = np.arange(1, m)
        mean_T2_vs_k, mean_Q_vs_k = [], []
        T2_matrix = np.zeros((n, len(k_range)))
        Q_matrix = np.zeros((n, len(k_range)))
 
        for idx_k, k in enumerate(k_range):
            Z_k = Z_scores[:, :k]
            lambda_k = lambda_hat[:k]
            T2_samples = np.sum((Z_k ** 2) / lambda_k, axis=1)
            T2_matrix[:, idx_k] = T2_samples
            mean_T2_vs_k.append(np.mean(T2_samples))
            Q_samples = np.sum(Z_scores[:, k:] ** 2, axis=1)
            Q_matrix[:, idx_k] = Q_samples
            mean_Q_vs_k.append(np.mean(Q_samples))
 
        print(f"\n--- PCA | {m} features, {n} samples | Total Variance: {total_variance:.4f} ---")
 
        if show_plot:
            pc_labels = [f"PC {i+1}" for i in range(m)]
            k_labels = [f"k={k}" for k in k_range]
            fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.18, vertical_spacing=0.28,
                                subplot_titles=("Feature Loading Matrix |P_hat|", "Eigenvalues λ",
                                                "Explained Variance", "Unexplained Variance",
                                                "Mean T² vs k", "Mean Q (SPE) vs k"))
            fig.add_trace(go.Heatmap(z=np.abs(P_hat), x=pc_labels, y=target_cols,
                                     colorscale='YlOrRd',
                                     colorbar=dict(title="Loading Weight", x=-0.12, len=0.38,
                                                   y=0.78, yanchor="middle", xanchor="right",
                                                   titleside="top")), row=1, col=1)
            fig.add_trace(go.Bar(x=pc_labels, y=lambda_hat, name="Eigenvalue (λ_j)",
                                 marker=dict(color='#1f77b4')), row=1, col=2)
            fig.add_trace(go.Bar(x=pc_labels, y=explained_variance_ratio * 100,
                                 name="Marginal Explained",
                                 marker=dict(color='#ff7f0e', opacity=0.75)), row=1, col=3)
            fig.add_trace(go.Scatter(x=pc_labels, y=cumulative_variance_ratio * 100,
                                     mode='lines+markers', name='Cumulative Captured',
                                     line=dict(color='#d62728', width=2.5, dash='dash')), row=1, col=3)
            fig.add_trace(go.Bar(x=pc_labels, y=unexplained_variance_ratio * 100,
                                 name="Remaining Noise",
                                 marker=dict(color='#2ca02c')), row=2, col=1)
            fig.add_trace(go.Scatter(x=k_labels, y=mean_T2_vs_k, mode='lines+markers',
                                     name='Mean T²', line=dict(color='#9467bd', width=2.5),
                                     marker=dict(size=6, symbol='diamond')), row=2, col=2)
            fig.add_trace(go.Scatter(x=k_labels, y=mean_Q_vs_k, mode='lines+markers',
                                     name='Mean Q (SPE)', line=dict(color='#e377c2', width=2.5),
                                     marker=dict(size=6, symbol='square')), row=2, col=3)
            fig.update_layout(
                title=dict(text="PCA Optimization & Feature Loading Dashboard", x=0.5, y=0.97),
                template="plotly_white", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=150, b=60, l=140, r=80), height=750, width=1250)
            fig.show()
 
        return {"mean_vector": mu_hat, "covariance_matrix_S": S_matrix,
                "eigenvalues_lambda": lambda_hat, "eigenvectors_P": P_hat,
                "explained_variance_ratio": explained_variance_ratio,
                "cumulative_variance_ratio": cumulative_variance_ratio,
                "unexplained_variance_ratio": unexplained_variance_ratio,
                "transformed_scores_Z": Z_scores, "score_covariance_diagonal": np.diag(S_Z),
                "features": target_cols, "k_values": k_range,
                "T2_matrix_vs_k": T2_matrix, "Q_matrix_vs_k": Q_matrix,
                "mean_T2_profile": np.array(mean_T2_vs_k), "mean_Q_profile": np.array(mean_Q_vs_k)}
 
    def compute_empirical_fa(self, k: int, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Decomposes the empirical correlation matrix R into shared structural
        variance (Lambda * Lambda^T) and uniqueness (Psi) via Factor Analysis.
        Estimates latent factor scores using Thomson's MMSE regression method.
        Generates a 2x2 Plotly Dashboard.
        """
        if self.df is None: raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
        else:
            target_cols = list(columns)
 
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("Sample size must exceed feature dimensions.")
        if k >= m: raise ValueError(f"k ({k}) must be strictly less than features m ({m}).")
 
        mu_hat = np.mean(X, axis=0)
        std_hat = np.std(X, axis=0, ddof=1)
        std_hat[std_hat == 0] = 1e-15
        Z = (X - mu_hat) / std_hat
        R_matrix = np.corrcoef(X, rowvar=False)
 
        fa = FactorAnalysis(n_components=k, rotation='varimax', random_state=42)
        fa.fit(Z)
        lambda_matrix = fa.components_.T
        psi_diagonal = fa.noise_variance_
        communality = np.sum(lambda_matrix**2, axis=1)
        uniqueness = psi_diagonal
        F_scores = fa.transform(Z)
 
        print(f"\n--- FA | {m} sensors → {k} latent factors ---")
        print(f"Avg Communality: {np.mean(communality)*100:.2f}%")
        print(f"Avg Uniqueness: {np.mean(uniqueness)*100:.2f}%")
 
        if show_plot:
            factor_labels = [f"Factor {j+1}" for j in range(k)]
            fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.24, vertical_spacing=0.28,
                                subplot_titles=("Structural Loadings Matrix |λ|",
                                                "Variance Partitioning (Communality vs Uniqueness)",
                                                "Sensor Uniqueness Noise Floor (φ²)",
                                                "Latent Factor Scores Empirical Variance"))
            fig.add_trace(go.Heatmap(z=np.abs(lambda_matrix), x=factor_labels, y=target_cols,
                                     colorscale='YlOrRd',
                                     colorbar=dict(title="Sensitivity Score", x=-0.15, len=0.38,
                                                   y=0.78, yanchor="middle", xanchor="right",
                                                   titleside="top")), row=1, col=1)
            fig.add_trace(go.Bar(y=target_cols, x=communality * 100,
                                 name="Communality (h² - Shared Structure)",
                                 orientation='h', marker=dict(color='#1f77b4')), row=1, col=2)
            fig.add_trace(go.Bar(y=target_cols, x=uniqueness * 100,
                                 name="Uniqueness (φ² - Channel Noise)",
                                 orientation='h', marker=dict(color='#ff7f0e')), row=1, col=2)
            fig.add_trace(go.Scatter(x=target_cols, y=uniqueness, mode='lines+markers',
                                     name='Uniqueness Profile (φ²)',
                                     line=dict(color='#d62728', width=2, dash='dot'),
                                     marker=dict(size=8, symbol='x')), row=2, col=1)
            factor_variances = np.var(F_scores, axis=0, ddof=1)
            fig.add_trace(go.Bar(x=factor_labels, y=factor_variances,
                                 name="Factor Empirical Variance",
                                 marker=dict(color='#2ca02c')), row=2, col=2)
            fig.update_layout(
                title=dict(text="Factor Analysis Latent Subspace Diagnostics Dashboard", x=0.5, y=0.97),
                template="plotly_white", showlegend=True, barmode='stack',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=150, b=60, l=140, r=80), height=750, width=1250)
            fig.show()
 
        return {"mean_vector_mu": mu_hat, "std_vector_D": std_hat,
                "correlation_matrix_R": R_matrix, "factor_loadings_lambda": lambda_matrix,
                "uniqueness_psi": uniqueness, "communality_h2": communality,
                "latent_factor_scores_F": F_scores, "sensors": target_cols}
 
 
# =============================================================================
 
import time
from datetime import datetime
import uuid
import json
import copy
import inspect
import plotly.io as pio
import networkx as nx
import graphviz
import base64
from IPython.display import HTML
 
 
class PlottingMethods:
 
    def get_methods_info(self, user_id=None):
        method_dicts = []
        methods = inspect.getmembers(self, inspect.ismethod)
        for name, method in methods:
            if name.startswith('_'): continue
            signature = inspect.signature(method)
            docstring = method.__doc__
            method_dicts += [{"method": name, "signature": str(signature),
                               "description": docstring.strip() if docstring else "No description available"}]
        return {'status': 'success', 'response': method_dicts}
 
    def _data_validate(self, data, message_dict):
        if data is None or (isinstance(data, str) and not data):
            message_dict.update({'message': 'No data'})
            return {'status': 'error', 'message_dict': message_dict}
        if isinstance(data, pd.DataFrame):
            if data.empty:
                message_dict.update({'message': 'No data'})
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': data.to_dict(orient='records')}
        if isinstance(data, list):
            if not data:
                message_dict.update({'message': 'No data'})
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': data}
        try:
            parsed_data = json.loads(data)
            records = parsed_data.get('records') if isinstance(parsed_data, dict) else parsed_data
            if not records:
                message_dict.update({'message': 'No data'})
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': records}
        except (json.JSONDecodeError, TypeError):
            message_dict.update({'message': 'Invalid data format'})
            return {'status': 'error', 'message_dict': message_dict}
 
    def plot_bar_chart(self, x='date', y='value', color=None, text=None, title='',
                       barmode='stack', hover_data=None, data_id=None,
                       data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly px bar chart with x as the x values,
        y as the y values, and color as the categories.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)
            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                return {'status': 'error', 'response': {'meta_data': message_dict,
                        'data': json.dumps({'figure': ''})},
                        'message': message_dict.get('message', 'Error')}
            data = validated_response.get('data')
 
            if isinstance(hover_data, str):
                try:
                    parsed = json.loads(hover_data)
                    hover_data = parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    hover_data = hover_data.split(',') if ',' in hover_data else None
 
            df = pd.DataFrame(data)
            df[y] = pd.to_numeric(df[y])
            c_categories_labels = None
            if color is not None:
                df.dropna(subset=[color], inplace=True)
                c_categories_labels = df[color].unique()
                if not any(sub in color.lower() for sub in ['month', 'week']):
                    c_categories_labels = sorted(c_categories_labels)
                df[color] = pd.Categorical(df[color], categories=c_categories_labels, ordered=True)
 
            x_categories_labels = df[x].unique()
            df[x] = pd.Categorical(df[x], categories=x_categories_labels, ordered=True)
            if hover_data: hover_data = [col for col in hover_data if col in df.columns]
 
            cat_orders = {x: x_categories_labels}
            if color is not None and c_categories_labels is not None:
                cat_orders[color] = c_categories_labels
 
            fig = px.bar(df, x=x, y=y, color=color, title=title, text=text,
                         hover_data=hover_data, category_orders=cat_orders)
            fig.update_layout(xaxis_title=x, yaxis_title=y,
                              uniformtext_minsize=8, uniformtext_mode='hide', barmode=barmode)
 
            plot_html = pio.to_html(fig, full_html=False,
                                    config={"displaylogo": False, "responsive": True},
                                    include_plotlyjs=True)
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')
            message_dict.update({'message': 'Bar chart plotted'})
            return {'status': 'success', 'response': {'meta_data': message_dict,
                    'data': json.dumps({'figure': fig_return}),
                    'message': json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            return {'status': 'error', 'response': {'meta_data': message_dict,
                    'data': json.dumps({'figure': ''})}, 'message': json.dumps(message_dict)}
 
    def plot_pie_chart(self, names='date', values='value', title='', hole=None,
                       data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates a responsive Plotly pie chart based on provided data.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)
            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                return {'status': 'error', 'response': {'meta_data': message_dict,
                        'data': json.dumps({'figure': ''})},
                        'message': message_dict.get('message', 'Error')}
            data = validated_response.get('data')
 
            df = pd.DataFrame(data)
            fig = px.pie(df, names=names, values=values, title=title, hole=hole)
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(title=title, uniformtext_minsize=10, uniformtext_mode='hide')
 
            plot_html = pio.to_html(fig, full_html=False,
                                    config={"displaylogo": False, "responsive": True},
                                    include_plotlyjs=True)
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')
            message_dict.update({'message': 'Pie chart plotted'})
            return {'status': 'success', 'response': {'meta_data': message_dict,
                    'data': json.dumps({'figure': fig_return}),
                    'message': json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            return {'status': 'error', 'response': {'meta_data': message_dict,
                    'data': json.dumps({'figure': ''})}, 'message': json.dumps(message_dict)}
 
    def plot_histogram(self, x='value', title='', bins=None, data='{"records":[]}',
                       meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly histogram with x as the x-axis.
        """
        message_dict = {'message': meta_data}
        validated_response = self._data_validate(data, message_dict)
        if not validated_response.get('status') == 'success':
            message_dict = validated_response.get('message_dict', {})
            return {'status': 'error', 'response': {'meta_data': message_dict,
                    'data': json.dumps({'figure': ''})},
                    'message': message_dict.get('message', 'Error')}
        data = validated_response.get('data')
 
        df = pd.DataFrame(data)
        if bins:
            if not isinstance(bins, list) or len(bins) < 2:
                return {'status': 'error', 'response': {'meta_data': 'Invalid bins', 'data': {'figure': ''}}}
            df[x] = pd.cut(df[x], bins=bins, right=False).astype(str)
 
        fig = px.histogram(df, x=x, title=title,
                           category_orders={x: [f"[{bins[i]}, {bins[i+1]})"
                                                for i in range(len(bins) - 1)]} if bins else None)
        fig.update_layout(title=title, xaxis_title=x, yaxis_title='Count', bargap=0.2)
 
        plot_html = pio.to_html(fig, full_html=False,
                                config={"displaylogo": False, "responsive": True},
                                include_plotlyjs=True)
        fig_id = str(uuid.uuid4())[:8]
        fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')
        message_dict.update({'message': 'Histogram plotted'})
        return {'status': 'success', 'response': {'meta_data': json.dumps(message_dict),
                'data': json.dumps({'figure': fig_return})}}
 
    def plot_heat_map(self, values='Sales', index='Region', columns='Category',
                      aggregade_method='sum', fill_value=0, title='Heatmap',
                      width=None, data_id=None, data='{"records":[]}',
                      meta_data={}, user_id=None):
        """
        Generates an interactive Plotly heatmap from tabular data
        with optional aggregation and layout control.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)
            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                return {'status': 'error', 'response': {'meta_data': message_dict,
                        'data': json.dumps({'figure': ''})},
                        'message': message_dict.get('message', 'Error')}
            data = validated_response.get('data')
 
            df = pd.DataFrame(data)
            col_labels = df[columns].unique()
            row_labels = df[index].unique()
            pivot_data = df.pivot_table(index=index, columns=columns, values=values,
                                        aggfunc=aggregade_method, fill_value=0)
            pivot_data = pivot_data.reindex(index=row_labels, columns=col_labels)
 
            fig = px.imshow(pivot_data, color_continuous_scale='Jet',
                            labels=dict(y=index, x=columns, color=values), text_auto=True)
            fig.update_layout(title=title, autosize=True, template='plotly', width=width)
 
            plot_html = pio.to_html(fig, full_html=False,
                                    config={"displaylogo": False, "responsive": True},
                                    include_plotlyjs=True)
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')
            message_dict.update({'message': 'Heat map plotted'})
            return {'status': 'success', 'response': {'meta_data': message_dict,
                    'data': json.dumps({'figure': fig_return}),
                    'message': json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            return {'status': 'error', 'response': {'meta_data': message_dict,
                    'data': json.dumps({'figure': ''})}, 'message': json.dumps(message_dict)}
 
    def display_image(self, result):
        """
        Renders a Plotly HTML figure inside the Colab output cell.
        """
        if result['status'] == 'success':
            response_data = json.loads(result['response']['data'])
            plot_html = response_data['figure']
            display(HTML(plot_html))
        else:
            print(f"Failed to plot: {result['response']['message']}")
