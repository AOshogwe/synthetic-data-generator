import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import networkx as nx
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from stats.correlations import CorrelationAnalyzer

class RelationshipDiscoverer:
    """Discovers complex relationships and conditional dependencies in data"""

    def __init__(self, schema):
        self.schema = schema
        self.dependency_graph = None
        self.conditional_relationships = {}

    def discover_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Discover relationships between variables in the dataset"""
        # Create correlation analyzer
        corr_analyzer = CorrelationAnalyzer(self.schema)

        # Build initial dependency graph based on correlations
        self.dependency_graph = corr_analyzer.build_dependency_graph(df)

        # Discover mutual information between variables
        mutual_info = self._calculate_mutual_information(df)

        # Discover conditional dependencies
        self.conditional_relationships = self._discover_conditional_dependencies(df)

        return {
            'dependency_graph': self.dependency_graph,
            'mutual_information': mutual_info,
            'conditional_relationships': self.conditional_relationships
        }

    def _calculate_mutual_information(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate mutual information between all pairs of variables"""
        # Prepare data
        columns = df.columns
        result = {col: {} for col in columns}

        for target_col in columns:
            # Determine if target is categorical or continuous
            is_categorical = self.schema.get(target_col, {}).get('type') == 'categorical'

            for feature_col in columns:
                if target_col == feature_col:
                    continue

                # Skip if both columns are identical
                if df[target_col].equals(df[feature_col]):
                    result[target_col][feature_col] = 1.0
                    continue

                # Prepare feature data
                X = df[[feature_col]]
                y = df[target_col]

                # Calculate mutual information
                try:
                    if is_categorical:
                        mi = mutual_info_classif(X, y, discrete_features='auto')[0]
                    else:
                        mi = mutual_info_regression(X, y, discrete_features='auto')[0]

                    result[target_col][feature_col] = mi
                except:
                    # Handle error cases
                    result[target_col][feature_col] = 0.0

        return result

    def _discover_conditional_dependencies(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Discover conditional dependencies between variables"""
        # Get all possible edges from dependency graph
        relationships = {}

        # Get columns with significant relationships
        sig_cols = list(self.dependency_graph.nodes())

        # For each potential relationship A->B, check if it's conditional on C
        for col_a in sig_cols:
            relationships[col_a] = {}

            for col_b in sig_cols:
                if col_a == col_b or not self.dependency_graph.has_edge(col_a, col_b):
                    continue

                # Look for third variables that might explain the relationship
                for col_c in sig_cols:
                    if col_c == col_a or col_c == col_b:
                        continue

                    # Check if col_c affects the relationship between col_a and col_b
                    conditional_effect = self._measure_conditional_effect(df, col_a, col_b, col_c)

                    if conditional_effect > 0.1:  # Threshold for significant effect
                        if col_b not in relationships[col_a]:
                            relationships[col_a][col_b] = {}

                        relationships[col_a][col_b][col_c] = conditional_effect

        return relationships

    def _measure_conditional_effect(self, df: pd.DataFrame, col_a: str, col_b: str, col_c: str) -> float:
        """
        Measure how much col_c affects the relationship between col_a and col_b
        Returns a value between 0 (no effect) and 1 (completely explains relationship)
        """
        # This is a simplified approach; more sophisticated methods exist

        try:
            # Calculate correlation between A and B
            if pd.api.types.is_numeric_dtype(df[col_a]) and pd.api.types.is_numeric_dtype(df[col_b]):
                corr_ab = df[[col_a, col_b]].corr().iloc[0, 1]
            else:
                # For non-numeric, use Cramer's V through correlation analyzer
                corr_analyzer = CorrelationAnalyzer(self.schema)
                corr_ab = corr_analyzer._cramers_v(df[col_a], df[col_b])

            # If col_c is continuous, bin it into discrete categories
            if pd.api.types.is_numeric_dtype(df[col_c]):
                c_bins = pd.qcut(df[col_c], 5, duplicates='drop')
            else:
                c_bins = df[col_c]

            # Calculate conditional correlations for each value of C
            c_values = c_bins.unique()
            conditional_corrs = []

            for c_val in c_values:
                # Filter data for this value of C
                subset = df[c_bins == c_val]

                # Need enough data for meaningful correlation
                if len(subset) < 10:
                    continue

                # Calculate correlation in this subset
                if pd.api.types.is_numeric_dtype(subset[col_a]) and pd.api.types.is_numeric_dtype(subset[col_b]):
                    try:
                        subset_corr = subset[[col_a, col_b]].corr().iloc[0, 1]
                        if not np.isnan(subset_corr):
                            conditional_corrs.append(subset_corr)
                    except:
                        pass
                else:
                    try:
                        corr_analyzer = CorrelationAnalyzer(self.schema)
                        subset_corr = corr_analyzer._cramers_v(subset[col_a], subset[col_b])
                        conditional_corrs.append(subset_corr)
                    except:
                        pass

            # If we couldn't calculate any conditional correlations, assume no effect
            if not conditional_corrs:
                return 0.0

            # Calculate average conditional correlation
            avg_cond_corr = np.mean(np.abs(conditional_corrs))

            # Measure effect as reduction in correlation magnitude
            effect = 1.0 - (avg_cond_corr / abs(corr_ab) if corr_ab != 0 else 0)

            # Clamp effect to [0, 1] range
            return max(0.0, min(1.0, effect))

        except:
            # If any errors occur, assume no effect
            return 0.0