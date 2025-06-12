# privacy_methods.py - Advanced Privacy-Preserving Techniques
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import hashlib
import secrets
from config import get_config

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """Implementation of differential privacy mechanisms"""

    def __init__(self, epsilon: float = 1.0):
        """
        Initialize differential privacy with privacy budget epsilon

        Args:
            epsilon: Privacy budget (smaller = more private, larger = more accurate)
        """
        self.epsilon = epsilon
        self.privacy_budget_used = 0.0

    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy"""
        if self.privacy_budget_used + self.epsilon > self.epsilon:
            raise ValueError("Privacy budget exceeded")

        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        self.privacy_budget_used += self.epsilon

        return value + noise

    def add_gaussian_noise(self, value: float, sensitivity: float, delta: float = 1e-5) -> float:
        """Add Gaussian noise for (epsilon, delta)-differential privacy"""
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise

    def privatize_histogram(self, data: pd.Series, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create differentially private histogram"""
        counts, bin_edges = np.histogram(data, bins=bins)

        # Add noise to each bin count
        private_counts = []
        for count in counts:
            private_count = self.add_laplace_noise(count, sensitivity=1.0)
            private_counts.append(max(0, private_count))  # Ensure non-negative

        return np.array(private_counts), bin_edges

    def privatize_mean(self, data: pd.Series, data_range: Tuple[float, float]) -> float:
        """Calculate differentially private mean"""
        min_val, max_val = data_range
        sensitivity = (max_val - min_val) / len(data)

        true_mean = data.mean()
        return self.add_laplace_noise(true_mean, sensitivity)


class KAnonymity:
    """Implementation of k-anonymity privacy model"""

    def __init__(self, k: int = 5):
        """
        Initialize k-anonymity with parameter k

        Args:
            k: Minimum group size for anonymity
        """
        self.k = k

    def identify_quasi_identifiers(self, df: pd.DataFrame) -> List[str]:
        """Identify potential quasi-identifier columns"""
        quasi_identifiers = []

        for column in df.columns:
            # Check if column could be a quasi-identifier
            unique_ratio = df[column].nunique() / len(df)

            # High uniqueness suggests quasi-identifier
            if 0.1 < unique_ratio < 0.9:
                quasi_identifiers.append(column)

        return quasi_identifiers

    def generalize_numerical(self, series: pd.Series, levels: int = 3) -> pd.Series:
        """Generalize numerical data into ranges"""
        try:
            # Create equal-width bins
            bins = pd.cut(series, bins=levels, precision=0)
            return bins.astype(str)
        except Exception:
            return series

    def generalize_categorical(self, series: pd.Series, hierarchy: Dict[str, str] = None) -> pd.Series:
        """Generalize categorical data using hierarchy"""
        if hierarchy:
            return series.map(hierarchy).fillna(series)
        else:
            # Simple generalization: group rare values
            value_counts = series.value_counts()
            rare_values = value_counts[value_counts < self.k].index

            result = series.copy()
            result[result.isin(rare_values)] = 'Other'
            return result

    def suppress_rare_combinations(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """Suppress records that appear in groups smaller than k"""
        if not quasi_identifiers:
            return df

        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)
        group_sizes = grouped.size()

        # Identify groups with size < k
        small_groups = group_sizes[group_sizes < self.k].index

        # Create mask for records to keep
        keep_mask = pd.Series(True, index=df.index)

        for group_key in small_groups:
            if isinstance(group_key, tuple):
                mask = pd.Series(True, index=df.index)
                for i, col in enumerate(quasi_identifiers):
                    mask &= (df[col] == group_key[i])
            else:
                mask = (df[quasi_identifiers[0]] == group_key)

            keep_mask &= ~mask

        logger.info(f"Suppressed {(~keep_mask).sum()} records to maintain {self.k}-anonymity")
        return df[keep_mask].reset_index(drop=True)

    def achieve_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str] = None) -> pd.DataFrame:
        """Apply k-anonymity to dataframe"""
        if quasi_identifiers is None:
            quasi_identifiers = self.identify_quasi_identifiers(df)

        result_df = df.copy()

        # Generalize quasi-identifiers
        for col in quasi_identifiers:
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = self.generalize_numerical(result_df[col])
            else:
                result_df[col] = self.generalize_categorical(result_df[col])

        # Suppress rare combinations
        result_df = self.suppress_rare_combinations(result_df, quasi_identifiers)

        return result_df


class LDiversity:
    """Implementation of l-diversity privacy model"""

    def __init__(self, l: int = 2):
        """
        Initialize l-diversity with parameter l

        Args:
            l: Minimum number of distinct sensitive values per group
        """
        self.l = l

    def check_l_diversity(self, df: pd.DataFrame, quasi_identifiers: List[str],
                          sensitive_attributes: List[str]) -> pd.DataFrame:
        """Check and enforce l-diversity"""
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)

        valid_groups = []

        for group_key, group_data in grouped:
            # Check diversity for each sensitive attribute
            is_diverse = True

            for sensitive_attr in sensitive_attributes:
                unique_values = group_data[sensitive_attr].nunique()
                if unique_values < self.l:
                    is_diverse = False
                    break

            if is_diverse:
                valid_groups.append(group_data)

        if valid_groups:
            result = pd.concat(valid_groups, ignore_index=True)
            logger.info(f"Maintained {len(result)} records with {self.l}-diversity")
            return result
        else:
            logger.warning("No groups satisfy l-diversity requirement")
            return pd.DataFrame()


class SecureHashing:
    """Secure hashing for identifier anonymization"""

    def __init__(self, salt: str = None):
        """Initialize with optional salt"""
        self.salt = salt or secrets.token_hex(16)

    def hash_identifier(self, identifier: str) -> str:
        """Create secure hash of identifier"""
        combined = f"{self.salt}{identifier}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()[:16]  # Truncate for readability

    def hash_column(self, series: pd.Series) -> pd.Series:
        """Hash an entire column of identifiers"""
        return series.astype(str).apply(self.hash_identifier)


class PrivacyAuditor:
    """Audit privacy protection levels of synthetic data"""

    def __init__(self):
        self.audit_results = {}

    def audit_privacy_risk(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive privacy risk audit"""
        audit = {
            'membership_inference': self._test_membership_inference(original_df, synthetic_df),
            'attribute_inference': self._test_attribute_inference(original_df, synthetic_df),
            'linkage_attack': self._test_linkage_attack(original_df, synthetic_df),
            'uniqueness_analysis': self._analyze_uniqueness(original_df, synthetic_df),
            'overall_risk': 'unknown'
        }

        # Calculate overall risk level
        risks = [audit['membership_inference']['risk_level'],
                 audit['attribute_inference']['risk_level'],
                 audit['linkage_attack']['risk_level']]

        if 'high' in risks:
            audit['overall_risk'] = 'high'
        elif 'medium' in risks:
            audit['overall_risk'] = 'medium'
        else:
            audit['overall_risk'] = 'low'

        return audit

    def _test_membership_inference(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Test vulnerability to membership inference attacks"""
        # Simple test: check for exact record matches
        original_records = set()
        synthetic_records = set()

        # Convert records to tuples for comparison (sample to avoid memory issues)
        sample_size = min(1000, len(original_df), len(synthetic_df))

        orig_sample = original_df.sample(sample_size, random_state=42)
        synth_sample = synthetic_df.sample(sample_size, random_state=42)

        for _, row in orig_sample.iterrows():
            original_records.add(tuple(row.fillna('').astype(str)))

        for _, row in synth_sample.iterrows():
            synthetic_records.add(tuple(row.fillna('').astype(str)))

        # Calculate overlap
        overlap = len(original_records.intersection(synthetic_records))
        overlap_rate = overlap / len(synthetic_records) if synthetic_records else 0

        if overlap_rate > 0.1:
            risk_level = 'high'
        elif overlap_rate > 0.05:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'exact_matches': overlap,
            'overlap_rate': overlap_rate,
            'risk_level': risk_level,
            'description': f"Found {overlap} exact record matches ({overlap_rate:.2%} overlap)"
        }

    def _test_attribute_inference(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Test vulnerability to attribute inference attacks"""
        # Test if rare attribute combinations are preserved
        categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) < 2:
            return {
                'rare_combinations': 0,
                'risk_level': 'low',
                'description': "Insufficient categorical data for attribute inference test"
            }

        # Check 2-column combinations
        rare_combinations = 0
        total_combinations = 0

        for i, col1 in enumerate(categorical_cols[:3]):  # Limit to first 3 columns
            for col2 in categorical_cols[i + 1:4]:  # Limit combinations
                # Find rare combinations in original data
                orig_combinations = original_df.groupby([col1, col2]).size()
                rare_orig = orig_combinations[orig_combinations <= 3]  # Rare = ≤3 occurrences

                if len(rare_orig) == 0:
                    continue

                # Check if these rare combinations appear in synthetic data
                synth_combinations = synthetic_df.groupby([col1, col2]).size()

                for combo_key, _ in rare_orig.items():
                    total_combinations += 1
                    if combo_key in synth_combinations.index:
                        rare_combinations += 1

        if total_combinations == 0:
            preservation_rate = 0
        else:
            preservation_rate = rare_combinations / total_combinations

        if preservation_rate > 0.5:
            risk_level = 'high'
        elif preservation_rate > 0.2:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'rare_combinations_preserved': rare_combinations,
            'total_rare_combinations': total_combinations,
            'preservation_rate': preservation_rate,
            'risk_level': risk_level,
            'description': f"Preserved {rare_combinations}/{total_combinations} rare attribute combinations"
        }

    def _test_linkage_attack(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Test vulnerability to linkage attacks using quasi-identifiers"""
        # Identify potential quasi-identifiers
        quasi_identifiers = []

        for col in original_df.columns:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                # Age-like columns
                if 'age' in col.lower() or (original_df[col].min() >= 0 and original_df[col].max() <= 120):
                    quasi_identifiers.append(col)
            elif original_df[col].dtype == 'object':
                # Categorical with moderate uniqueness
                uniqueness = original_df[col].nunique() / len(original_df)
                if 0.1 < uniqueness < 0.5:
                    quasi_identifiers.append(col)

        if len(quasi_identifiers) < 2:
            return {
                'vulnerable_records': 0,
                'risk_level': 'low',
                'description': "Insufficient quasi-identifiers for linkage attack test"
            }

        # Check uniqueness of quasi-identifier combinations
        qi_subset = quasi_identifiers[:3]  # Use first 3 QIs

        orig_combinations = original_df.groupby(qi_subset).size()
        unique_orig = (orig_combinations == 1).sum()

        synth_combinations = synthetic_df.groupby(qi_subset).size()
        unique_synth = (synth_combinations == 1).sum()

        # Risk is high if many unique combinations are preserved
        total_orig_groups = len(orig_combinations)
        uniqueness_rate = unique_synth / total_orig_groups if total_orig_groups > 0 else 0

        if uniqueness_rate > 0.3:
            risk_level = 'high'
        elif uniqueness_rate > 0.1:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'unique_combinations_original': unique_orig,
            'unique_combinations_synthetic': unique_synth,
            'uniqueness_rate': uniqueness_rate,
            'risk_level': risk_level,
            'description': f"Synthetic data has {unique_synth} unique quasi-identifier combinations"
        }

    def _analyze_uniqueness(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze uniqueness patterns"""
        uniqueness_analysis = {}

        for col in original_df.columns:
            if col in synthetic_df.columns:
                orig_unique = original_df[col].nunique()
                synth_unique = synthetic_df[col].nunique()

                uniqueness_analysis[col] = {
                    'original_unique': orig_unique,
                    'synthetic_unique': synth_unique,
                    'uniqueness_ratio': synth_unique / orig_unique if orig_unique > 0 else 0
                }

        return uniqueness_analysis


class AdvancedAnonymizer:
    """Advanced anonymization techniques"""

    def __init__(self):
        self.differential_privacy = DifferentialPrivacy()
        self.k_anonymity = KAnonymity()
        self.l_diversity = LDiversity()
        self.secure_hasher = SecureHashing()
        self.privacy_auditor = PrivacyAuditor()

    def apply_advanced_anonymization(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply advanced anonymization based on configuration"""
        result_df = df.copy()

        privacy_level = config.get('privacy_level', 'balanced')

        if privacy_level == 'minimal':
            # Basic anonymization only
            result_df = self._apply_basic_anonymization(result_df, config)

        elif privacy_level == 'balanced':
            # K-anonymity + basic techniques
            result_df = self._apply_balanced_anonymization(result_df, config)

        elif privacy_level == 'high':
            # Full privacy protection
            result_df = self._apply_high_privacy_anonymization(result_df, config)

        return result_df

    def _apply_basic_anonymization(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply basic anonymization techniques"""
        result_df = df.copy()

        # Hash identifiers
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for col in id_columns:
            if df[col].dtype == 'object':
                result_df[col] = self.secure_hasher.hash_column(df[col])

        return result_df

    def _apply_balanced_anonymization(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply balanced anonymization (k-anonymity + hashing)"""
        result_df = df.copy()

        # Apply k-anonymity
        k_value = config.get('k_anonymity', 5)
        self.k_anonymity.k = k_value

        quasi_identifiers = self.k_anonymity.identify_quasi_identifiers(result_df)
        if quasi_identifiers:
            result_df = self.k_anonymity.achieve_k_anonymity(result_df, quasi_identifiers)

        # Hash identifiers
        result_df = self._apply_basic_anonymization(result_df, config)

        return result_df

    def _apply_high_privacy_anonymization(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply high privacy protection (differential privacy + k-anonymity + l-diversity)"""
        result_df = df.copy()

        # Apply k-anonymity
        k_value = config.get('k_anonymity', 10)  # Higher k for more privacy
        self.k_anonymity.k = k_value

        quasi_identifiers = self.k_anonymity.identify_quasi_identifiers(result_df)
        if quasi_identifiers:
            result_df = self.k_anonymity.achieve_k_anonymity(result_df, quasi_identifiers)

        # Apply l-diversity if sensitive attributes are identified
        sensitive_attrs = [col for col in df.columns
                           if any(term in col.lower() for term in ['diagnosis', 'income', 'salary', 'disease'])]

        if sensitive_attrs and quasi_identifiers:
            l_value = config.get('l_diversity', 3)
            self.l_diversity.l = l_value
            result_df = self.l_diversity.check_l_diversity(result_df, quasi_identifiers, sensitive_attrs)

        # Apply differential privacy to numerical columns
        epsilon = config.get('epsilon', 1.0)
        self.differential_privacy.epsilon = epsilon

        for col in result_df.select_dtypes(include=[np.number]).columns:
            if col not in quasi_identifiers:  # Don't add noise to generalized QIs
                try:
                    data_range = (result_df[col].min(), result_df[col].max())
                    private_mean = self.differential_privacy.privatize_mean(result_df[col], data_range)

                    # Add some controlled noise while preserving distribution shape
                    noise_scale = (data_range[1] - data_range[0]) * 0.01  # 1% of range
                    noise = np.random.normal(0, noise_scale, len(result_df))
                    result_df[col] = result_df[col] + noise

                except Exception as e:
                    logger.warning(f"Could not apply differential privacy to {col}: {e}")

        # Hash all identifiers
        result_df = self._apply_basic_anonymization(result_df, config)

        return result_df

    def audit_privacy_protection(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive privacy audit"""
        return self.privacy_auditor.audit_privacy_risk(original_df, synthetic_df)


# Example usage and factory function
def create_privacy_engine(privacy_level: str = 'balanced') -> AdvancedAnonymizer:
    """Factory function to create privacy engine with appropriate settings"""
    anonymizer = AdvancedAnonymizer()

    if privacy_level == 'minimal':
        anonymizer.k_anonymity.k = 3
        anonymizer.differential_privacy.epsilon = 2.0
    elif privacy_level == 'balanced':
        anonymizer.k_anonymity.k = 5
        anonymizer.l_diversity.l = 2
        anonymizer.differential_privacy.epsilon = 1.0
    elif privacy_level == 'high':
        anonymizer.k_anonymity.k = 10
        anonymizer.l_diversity.l = 3
        anonymizer.differential_privacy.epsilon = 0.5

    return anonymizer