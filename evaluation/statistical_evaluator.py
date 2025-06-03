import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from evaluation.ml_evaluator import MLUtilityEvaluator
from evaluation.privacy_evaluator import PrivacyRiskEvaluator

class StatisticalSimilarityEvaluator:
    """
    Evaluates how well synthetic data preserves statistical properties of original data
    """

    def __init__(self, schema):
        self.schema = schema

    def evaluate(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate statistical similarity between original and synthetic data
        Returns a dictionary of metrics
        """
        # Ensure dataframes have the same columns
        common_columns = set(original_df.columns).intersection(set(synthetic_df.columns))
        # Convert set to list before using as indexer
        common_columns_list = list(common_columns)
        original_df = original_df[common_columns_list].copy()
        synthetic_df = synthetic_df[common_columns_list].copy()

        # Initialize results dictionary
        results = {
            'column_metrics': {},
            'correlation_similarity': None,
            'pca_explained_variance_similarity': None,
            'overall_score': None
        }

        # Calculate per-column metrics
        column_scores = []
        for column in common_columns_list:  # Use list here too for consistency
            column_type = self.schema.get(column, {}).get('type', 'unknown')

            if column_type == 'categorical':
                metrics = self._evaluate_categorical_column(
                    original_df[column], synthetic_df[column]
                )
            else:  # Assume numeric
                metrics = self._evaluate_numeric_column(
                    original_df[column], synthetic_df[column]
                )

            results['column_metrics'][column] = metrics
            column_scores.append(metrics['similarity_score'])

        # Calculate correlation matrix similarity
        results['correlation_similarity'] = self._evaluate_correlation_similarity(
            original_df, synthetic_df
        )

        # Calculate PCA explained variance similarity
        results['pca_explained_variance_similarity'] = self._evaluate_pca_similarity(
            original_df, synthetic_df
        )

        # Calculate overall score
        # Add safeguard against empty lists or None values
        valid_scores = [score for score in [
            np.mean(column_scores) if column_scores else None,
            results['correlation_similarity'],
            results['pca_explained_variance_similarity']
        ] if score is not None]

        results['overall_score'] = np.mean(valid_scores) if valid_scores else None

        return results


    def _evaluate_categorical_column(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """Evaluate similarity between categorical columns"""
        # Convert all values to strings to avoid Timestamp comparison errors
        original = original.astype(str)
        synthetic = synthetic.astype(str)

        # Calculate frequency distributions
        orig_freq = original.value_counts(normalize=True).sort_index()
        synth_freq = synthetic.value_counts(normalize=True).sort_index()

        # Align frequencies (add missing categories with zero frequency)
        all_categories = sorted(set(orig_freq.index) | set(synth_freq.index))
        for cat in all_categories:
            if cat not in orig_freq:
                orig_freq[cat] = 0
            if cat not in synth_freq:
                synth_freq[cat] = 0

        orig_freq = orig_freq.sort_index()
        synth_freq = synth_freq.sort_index()

        # Calculate Jensen-Shannon divergence
        js_divergence = self._jensen_shannon_divergence(orig_freq.values, synth_freq.values)

        # Chi-squared test
        try:
            chi2, p_value = stats.chisquare(
                f_obs=synth_freq.values * len(synthetic),
                f_exp=orig_freq.values * len(synthetic)
            )
        except:
            chi2, p_value = np.nan, np.nan

        # TV distance (Total Variation distance)
        tv_distance = 0.5 * np.sum(np.abs(orig_freq.values - synth_freq.values))

        # Calculate similarity score (1 - normalized distance)
        # JS divergence is already between 0 and 1
        similarity_score = 1 - js_divergence

        return {
            'js_divergence': js_divergence,
            'tv_distance': tv_distance,
            'chi2_stat': chi2,
            'p_value': p_value,
            'similarity_score': similarity_score
        }

    def _evaluate_numeric_column(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """
           Evaluate similarity between original and synthetic numeric columns
           """
        # Check if we're actually dealing with numeric data
        if not pd.api.types.is_numeric_dtype(original) or not pd.api.types.is_numeric_dtype(synthetic):
            print(f"Warning: Non-numeric data detected in column {original.name}. Treating as categorical.")
            return self._evaluate_categorical_column(original, synthetic)

        # Calculate statistics
        stats = {
            'mean': original.mean(),
            'median': original.median(),
            'std': original.std(),
            'min': original.min(),
            'max': original.max(),
            'skew': original.skew() if len(original) > 2 else 0.0
        }

        # Calculate statistics for synthetic data
        synth_stats = {
            'mean': synthetic.mean(),
            'median': synthetic.median(),
            'std': synthetic.std(),
            'min': synthetic.min(),
            'max': synthetic.max(),
            'skew': synthetic.skew() if len(synthetic) > 2 else 0.0
        }

        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(original.dropna(), synthetic.dropna())
        except:
            ks_stat, ks_pvalue = np.nan, np.nan

        # Wasserstein distance (Earth Mover's Distance)
        try:
            wasserstein = stats.wasserstein_distance(original.dropna(), synthetic.dropna())
            # Normalize by range
            range_val = max(original.max() - original.min(), 1e-10)
            normalized_wasserstein = wasserstein / range_val
        except:
            wasserstein = np.nan
            normalized_wasserstein = np.nan

        # Calculate statistics similarity score
        stat_diffs = [
            abs(stats['mean'] - synth_stats['mean']) / max(abs(stats['mean']), 1e-10),
            abs(stats['std'] - synth_stats['std']) / max(abs(stats['std']), 1e-10),
            abs(stats['median'] - synth_stats['median']) / max(abs(stats['median']), 1e-10),
            abs(stats['skew'] - synth_stats['skew']) / max(abs(stats['skew']), 1.0)
        ]
        stats_similarity = 1 - min(1.0, np.mean([min(diff, 1.0) for diff in stat_diffs]))

        # Distribution similarity (1 - normalized wasserstein)
        dist_similarity = 1 - min(1.0, normalized_wasserstein)

        # Overall similarity score
        similarity_score = 0.5 * stats_similarity + 0.5 * dist_similarity

        return {
            'original_stats': stats,
            'synthetic_stats': synth_stats,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'wasserstein': wasserstein,
            'normalized_wasserstein': normalized_wasserstein,
            'stats_similarity': stats_similarity,
            'dist_similarity': dist_similarity,
            'similarity_score': similarity_score
        }

    def _evaluate_date_column(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """
        Evaluate similarity between original and synthetic date columns
        """
        # Try to convert to datetime if they're not already
        try:
            if not pd.api.types.is_datetime64_dtype(original):
                original = pd.to_datetime(original, errors='coerce')
            if not pd.api.types.is_datetime64_dtype(synthetic):
                synthetic = pd.to_datetime(synthetic, errors='coerce')
        except:
            # If conversion fails, treat as categorical
            print(f"Warning: Could not convert column to datetime. Treating as categorical.")
            return self._evaluate_categorical_column(original, synthetic)

        # Calculate date range statistics
        orig_min = original.min()
        orig_max = original.max()
        orig_range = (orig_max - orig_min).days

        synth_min = synthetic.min()
        synth_max = synthetic.max()
        synth_range = (synth_max - synth_min).days

        # Calculate similarity in range
        range_sim = 1.0 - abs(orig_range - synth_range) / max(orig_range, 1)

        # Calculate similarity in distribution of dates
        # (e.g., same proportion of dates in each month)

        # Calculate overall similarity score
        similarity_score = range_sim

        return {
            'min_date': str(orig_min),
            'max_date': str(orig_max),
            'synth_min_date': str(synth_min),
            'synth_max_date': str(synth_max),
            'date_range_similarity': range_sim,
            'similarity_score': similarity_score
        }

    def _evaluate_correlation_similarity(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """Evaluate similarity between correlation matrices"""
        # Select numeric columns
        numeric_cols = [col for col in original_df.columns
                        if pd.api.types.is_numeric_dtype(original_df[col])]

        if len(numeric_cols) < 2:
            return 1.0  # Not enough numeric columns for correlation

        # Calculate correlation matrices
        orig_corr = original_df[numeric_cols].corr().fillna(0)
        synth_corr = synthetic_df[numeric_cols].corr().fillna(0)

        # Calculate Frobenius norm of difference
        diff_norm = np.linalg.norm(orig_corr.values - synth_corr.values)

        # Normalize by number of elements
        n_elements = len(numeric_cols) ** 2
        normalized_diff = diff_norm / np.sqrt(n_elements)

        # Convert to similarity score
        similarity = 1 - min(1.0, normalized_diff)

        return similarity

    def _evaluate_pca_similarity(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """Evaluate similarity in principal component explained variance"""
        from sklearn.decomposition import PCA

        # Select numeric columns
        numeric_cols = [col for col in original_df.columns
                        if pd.api.types.is_numeric_dtype(original_df[col])]

        if len(numeric_cols) < 2:
            return 1.0  # Not enough numeric columns for PCA

        # Standardize data
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(original_df[numeric_cols].fillna(0))
        synth_scaled = scaler.transform(synthetic_df[numeric_cols].fillna(0))

        # Calculate PCA for both datasets
        n_components = min(len(numeric_cols), 10)  # Use at most 10 components

        pca_orig = PCA(n_components=n_components)
        pca_orig.fit(orig_scaled)
        orig_explained_var = pca_orig.explained_variance_ratio_

        pca_synth = PCA(n_components=n_components)
        pca_synth.fit(synth_scaled)
        synth_explained_var = pca_synth.explained_variance_ratio_

        # Calculate similarity in explained variance
        diff = np.abs(orig_explained_var - synth_explained_var)
        mse = np.mean(diff ** 2)

        # Convert to similarity score
        similarity = 1 - min(1.0, np.sqrt(mse))

        return similarity

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        # Avoid zero probabilities
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # JS divergence
        m = 0.5 * (p + q)
        js = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

        return js


class SyntheticDataEvaluator:
    """
    Comprehensive evaluator for synthetic data quality
    """

    def __init__(self):
        self.schema = None

    def set_schema(self, schema):
        """Set the data schema"""
        self.schema = schema

    def evaluate_statistical_similarity(self, original_df: pd.DataFrame,
                                        synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate statistical similarity"""
        # Infer schema if not provided
        if self.schema is None:
            self.schema = self._infer_schema(original_df)

        # Use statistical similarity evaluator
        evaluator = StatisticalSimilarityEvaluator(self.schema)
        return evaluator.evaluate(original_df, synthetic_df)

    def evaluate_privacy_risk(self, original_df: pd.DataFrame,
                              synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate privacy risks in synthetic data"""
        # Infer schema if not provided
        if self.schema is None:
            self.schema = self._infer_schema(original_df)

        # Use privacy risk evaluator
        evaluator = PrivacyRiskEvaluator(self.schema)
        return evaluator.evaluate(original_df, synthetic_df)

    def evaluate_ml_utility(self, original_df: pd.DataFrame,
                            synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate machine learning utility of synthetic data"""
        # Infer schema if not provided
        if self.schema is None:
            self.schema = self._infer_schema(original_df)

        # Use ML utility evaluator
        evaluator = MLUtilityEvaluator(self.schema)
        return evaluator.evaluate(original_df, synthetic_df)

    def _infer_schema(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Infer schema from dataframe"""
        schema = {}

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                schema[column] = {'type': 'numeric'}
            else:
                schema[column] = {'type': 'categorical'}

        return schema