import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any
import networkx as nx


class CorrelationAnalyzer:
    """Analyzes correlations and dependencies between variables"""

    def __init__(self, schema):
        self.schema = schema
        self.correlation_matrix = None
        self.dependency_graph = None

    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between variables in the dataset"""
        # Separate numeric and categorical columns
        numeric_cols = [col for col, info in self.schema.items()
                        if info.get('type', '') == 'numeric' and col in df.columns]
        categorical_cols = [col for col, info in self.schema.items()
                            if info.get('type', '') == 'categorical' and col in df.columns]

        # Initialize results
        results = {
            'pearson': None,
            'spearman': None,
            'cramer_v': {},
            'mutual_info': None
        }

        # Calculate Pearson correlation for numeric variables
        if len(numeric_cols) > 1:
            pearson_corr = df[numeric_cols].corr(method='pearson')
            results['pearson'] = pearson_corr

        # Calculate Spearman rank correlation (works with non-linear relationships)
        if len(numeric_cols) > 1:
            spearman_corr = df[numeric_cols].corr(method='spearman')
            results['spearman'] = spearman_corr

        # Calculate Cramér's V for categorical variables
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 != col2 and f"{col2}_{col1}" not in results['cramer_v']:
                    key = f"{col1}_{col2}"
                    results['cramer_v'][key] = self._cramers_v(df[col1], df[col2])

        # Store correlation results
        self.correlation_matrix = results
        return results

    def build_dependency_graph(self, df: pd.DataFrame, threshold: float = 0.2) -> nx.DiGraph:
        """Build a directed graph of variable dependencies"""
        # Ensure correlations have been analyzed
        if self.correlation_matrix is None:
            self.analyze_correlations(df)

        # Create directed graph
        G = nx.DiGraph()

        # Add all columns as nodes
        for column in df.columns:
            G.add_node(column)

        # Add edges based on correlation strengths
        if self.correlation_matrix['pearson'] is not None:
            pearson = self.correlation_matrix['pearson'].abs()
            for col1 in pearson.columns:
                for col2 in pearson.index:
                    if col1 != col2 and pearson.loc[col2, col1] > threshold:
                        # Use conditional mutual information to determine direction
                        direction = self._determine_direction(df, col1, col2)
                        if direction == 1:
                            G.add_edge(col1, col2, weight=pearson.loc[col2, col1])
                        else:
                            G.add_edge(col2, col1, weight=pearson.loc[col2, col1])

        # Add edges for categorical relationships
        for key, value in self.correlation_matrix['cramer_v'].items():
            if value > threshold:
                col1, col2 = key.split('_')
                # Determine direction for categorical variables
                direction = self._determine_direction(df, col1, col2)
                if direction == 1:
                    G.add_edge(col1, col2, weight=value)
                else:
                    G.add_edge(col2, col1, weight=value)

        # Store dependency graph
        self.dependency_graph = G
        return G

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Cramér's V statistic between two categorical variables"""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def _determine_direction(self, df: pd.DataFrame, col1: str, col2: str) -> int:
        """
        Determine the direction of dependency between two variables
        Returns 1 if col1->col2, -1 if col2->col1
        """
        # This is a simplified approach; more sophisticated methods exist
        # For example, conditional independence tests or Bayesian approaches

        # Here we'll use a heuristic based on entropy reduction
        entropy1 = self._entropy(df[col1])
        entropy2 = self._entropy(df[col2])

        # The variable with higher entropy might be influencing the other
        if entropy1 > entropy2:
            return 1
        else:
            return -1

    def _entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy of a series"""
        if pd.api.types.is_numeric_dtype(series):
            # For numeric, bin the data first
            hist, _ = np.histogram(series, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))
        else:
            # For categorical, use value counts
            value_counts = series.value_counts(normalize=True)
            return stats.entropy(value_counts)

    def get_conditional_distribution(self, df: pd.DataFrame, target_col: str, condition_col: str,
                                     condition_value: Any) -> Dict[str, Any]:
        """
        Get the conditional distribution of target_col given condition_col=condition_value
        """
        # Filter data based on condition
        filtered_df = df[df[condition_col] == condition_value]

        # If no data matches the condition, return None
        if len(filtered_df) == 0:
            return None

        # Analyze the distribution of the target column in the filtered data
        analyzer = DistributionAnalyzer()
        return analyzer.analyze_column(filtered_df[target_col])