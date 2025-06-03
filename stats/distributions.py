import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Any
from sklearn.neighbors import KernelDensity


class DistributionAnalyzer:
    """Analyzes and fits distributions to numerical and categorical data"""

    def __init__(self):
        # Candidate parametric distributions to test for continuous variables
        self.continuous_distributions = [
            stats.norm, stats.beta, stats.gamma, stats.lognorm,
            stats.expon, stats.weibull_min, stats.uniform
        ]

    def analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column and determine its distribution"""
        if pd.api.types.is_numeric_dtype(series):
            return self._analyze_continuous(series)
        else:
            return self._analyze_categorical(series)

    def _analyze_continuous(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze continuous variable distribution"""
        # Remove nulls for analysis
        series = series.dropna()

        # Initialize results dictionary
        results = {
            'type': 'continuous',
            'stats': {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skew': series.skew(),
                'kurtosis': series.kurtosis()
            },
            'best_fit': None,
            'kde_fit': None
        }

        # Skip if too few samples
        if len(series) < 10:
            return results

        # Try parametric distribution fitting
        best_dist = None
        best_params = None
        best_sse = float('inf')

        # Normalize data to [0,1] range for better fitting
        normalized = (series - series.min()) / (series.max() - series.min())

        for distribution in self.continuous_distributions:
            try:
                # Fit distribution
                params = distribution.fit(normalized)

                # Generate theoretical PDF
                x = np.linspace(0, 1, 100)
                pdf = distribution.pdf(x, *params)

                # Compare with empirical histogram
                hist, bin_edges = np.histogram(normalized, bins=20, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Interpolate PDF at bin centers
                pdf_at_bins = np.interp(bin_centers, x, pdf)

                # Calculate sum of squared errors
                sse = np.sum((hist - pdf_at_bins) ** 2)

                if sse < best_sse:
                    best_dist = distribution
                    best_params = params
                    best_sse = sse
            except:
                # Skip distributions that fail to fit
                continue

        # Store best parametric fit
        if best_dist is not None:
            results['best_fit'] = {
                'distribution': best_dist.name,
                'parameters': best_params,
                'sse': best_sse
            }

        # Always fit KDE as fallback
        try:
            # Standardize data for KDE
            data = (series - series.mean()) / series.std()
            data = data.values.reshape(-1, 1)

            # Fit KDE
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
            kde.fit(data)

            # Store KDE model
            results['kde_fit'] = {
                'model': kde,
                'mean': series.mean(),
                'std': series.std()
            }
        except:
            pass

        return results

    def _analyze_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical variable distribution"""
        # Remove nulls for analysis
        series = series.dropna()

        # Calculate frequency distribution
        value_counts = series.value_counts(normalize=True)

        # Calculate entropy
        entropy = stats.entropy(value_counts)

        # Store results
        return {
            'type': 'categorical',
            'value_counts': value_counts.to_dict(),
            'entropy': entropy,
            'n_categories': len(value_counts)
        }

    def generate_synthetic_values(self, analysis: Dict[str, Any], n_samples: int) -> np.ndarray:
        """Generate synthetic values based on distribution analysis"""
        if analysis['type'] == 'continuous':
            return self._generate_continuous(analysis, n_samples)
        else:
            return self._generate_categorical(analysis, n_samples)

    def _generate_continuous(self, analysis: Dict[str, Any], n_samples: int) -> np.ndarray:
        """Generate continuous values based on distribution analysis"""
        # First try parametric fit if available
        if analysis['best_fit'] is not None:
            try:
                dist_name = analysis['best_fit']['distribution']
                params = analysis['best_fit']['parameters']

                # Find the distribution object
                for dist in self.continuous_distributions:
                    if dist.name == dist_name:
                        # Generate random samples
                        samples = dist.rvs(*params, size=n_samples)

                        # Rescale to original range
                        stats_dict = analysis['stats']
                        min_val, max_val = stats_dict['min'], stats_dict['max']
                        samples = samples * (max_val - min_val) + min_val
                        return samples
            except:
                pass

        # Fall back to KDE if parametric generation fails
        if analysis['kde_fit'] is not None:
            try:
                kde = analysis['kde_fit']['model']
                mean = analysis['kde_fit']['mean']
                std = analysis['kde_fit']['std']

                # Sample from KDE
                samples = kde.sample(n_samples=n_samples).flatten()

                # Transform back to original scale
                samples = samples * std + mean
                return samples
            except:
                pass

        # Last resort: sample directly from normal distribution based on stats
        mean = analysis['stats']['mean']
        std = analysis['stats']['std']
        return np.random.normal(mean, std, n_samples)

    def _generate_categorical(self, analysis: Dict[str, Any], n_samples: int) -> np.ndarray:
        """Generate categorical values based on distribution analysis"""
        categories = list(analysis['value_counts'].keys())
        probabilities = list(analysis['value_counts'].values())

        # Sample from multinomial distribution
        return np.random.choice(categories, size=n_samples, p=probabilities)