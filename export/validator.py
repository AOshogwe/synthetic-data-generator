import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json


class DataValidator:
    """
    Validates synthetic data against schema and constraints
    """

    def __init__(self, schema):
        self.schema = schema
        self.constraints = {}

    def add_constraint(self, column: str, constraint_type: str, params: Dict[str, Any]):
        """
        Add a constraint to validate

        Parameters:
        column: Column name
        constraint_type: Type of constraint ('range', 'unique', 'regex', 'dependent', etc.)
        params: Parameters for the constraint
        """
        if column not in self.constraints:
            self.constraints[column] = []

        self.constraints[column].append({
            'type': constraint_type,
            'params': params
        })

    def validate(self, df):
        """Validate synthetic data against schema"""
        issues = []

        # Check column presence
        for column in self.schema:
            if column not in df.columns:
                issues.append(f"Missing column: {column}")

        # Check data types and constraints
        for column in df.columns:
            if column not in self.schema:
                continue

            col_info = self.schema[column]
            col_type = col_info.get('type')

            # Check for nulls if required
            if col_info.get('nullable') is False and df[column].isnull().any():
                issues.append(f"Column {column} contains nulls but is marked as non-nullable")

            # Type-specific validation
            if col_type == 'numeric':
                non_numeric = df[column].apply(lambda x: not (isinstance(x, (int, float)) or pd.isna(x)))
                if non_numeric.any():
                    issues.append(f"Column {column} contains non-numeric values")

            elif col_type == 'date':
                try:
                    pd.to_datetime(df[column], errors='raise')
                except:
                    issues.append(f"Column {column} contains invalid date values")

            # Range validation for numeric columns
            if col_type == 'numeric' and 'min' in col_info and 'max' in col_info:
                out_of_range = df[column].apply(
                    lambda x: x < col_info['min'] or x > col_info['max'] if not pd.isna(x) else False
                )
                if out_of_range.any():
                    issues.append(f"Column {column} contains values outside valid range")

        return issues

    def _validate_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate dataframe against schema"""
        errors = []

        # Check columns match schema
        schema_columns = set(self.schema.keys())
        df_columns = set(df.columns)

        missing_columns = schema_columns - df_columns
        extra_columns = df_columns - schema_columns

        if missing_columns:
            errors.append(f"Missing columns in data: {', '.join(missing_columns)}")

        if extra_columns:
            errors.append(f"Extra columns in data: {', '.join(extra_columns)}")

        # Check data types
        for column in schema_columns.intersection(df_columns):
            expected_type = self.schema[column]['type']

            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(df[column]):
                    errors.append(f"Column '{column}' should be numeric but isn't")
            elif expected_type == 'categorical':
                # Categorical data could be any type, but check for excessive unique values
                n_unique = df[column].nunique()
                if n_unique > 10000:  # Arbitrary large threshold
                    errors.append(
                        f"Column '{column}' has {n_unique} unique values, which is unusually high for a categorical variable")
            elif expected_type == 'datetime':
                try:
                    pd.to_datetime(df[column])
                except:
                    errors.append(f"Column '{column}' contains values that can't be converted to datetime")

        return errors

    def _validate_constraints(self, df: pd.DataFrame) -> List[str]:
        """Validate dataframe against defined constraints"""
        errors = []

        for column, column_constraints in self.constraints.items():
            if column not in df.columns:
                continue

            for constraint in column_constraints:
                constraint_type = constraint['type']
                params = constraint['params']

                # Range constraint for numeric columns
                if constraint_type == 'range':
                    if 'min' in params and (df[column] < params['min']).any():
                        errors.append(f"Column '{column}' has values below minimum {params['min']}")
                    if 'max' in params and (df[column] > params['max']).any():
                        errors.append(f"Column '{column}' has values above maximum {params['max']}")

                # Unique constraint
                elif constraint_type == 'unique':
                    if not df[column].is_unique:
                        errors.append(f"Column '{column}' should have unique values but has duplicates")

                # Regex pattern constraint
                elif constraint_type == 'regex':
                    pattern = params.get('pattern')
                    if pattern:
                        non_matching = df[df[column].astype(str).str.match(pattern) == False]
                        if len(non_matching) > 0:
                            errors.append(
                                f"Column '{column}' has {len(non_matching)} values not matching pattern '{pattern}'")

                # Dependent column constraint
                elif constraint_type == 'dependent':
                    dependent_col = params.get('column')
                    condition = params.get('condition')

                    if dependent_col in df.columns and condition:
                        # Convert condition string to actual condition
                        # This is a simplified version - a real implementation would need
                        # a more sophisticated approach to parse and evaluate conditions
                        try:
                            # For simple equal conditions
                            if '==' in condition:
                                left, right = condition.split('==')
                                left = left.strip()
                                right = right.strip()

                                invalid = df[~(df[column] == df[dependent_col])]
                                if len(invalid) > 0:
                                    errors.append(
                                        f"Violated dependency: {len(invalid)} records where {column} != {dependent_col}")
                        except:
                            errors.append(f"Could not evaluate dependent condition '{condition}' for column '{column}'")

        return errors

    def _generate_warnings(self, df: pd.DataFrame) -> List[str]:
        """Generate warnings for potential issues"""
        warnings = []

        # Check for null values
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            for col, count in columns_with_nulls.items():
                warnings.append(f"Column '{col}' has {count} null values ({count / len(df):.1%} of data)")

        # Check for extreme outliers in numeric columns
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr

                n_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

                if n_outliers > 0:
                    warnings.append(
                        f"Column '{column}' has {n_outliers} extreme outliers ({n_outliers / len(df):.1%} of data)")

        return warnings


class AnomalyDetector:
    """
    Detects anomalies in synthetic data
    """

    def __init__(self, schema):
        self.schema = schema

    def detect_anomalies(self, df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect anomalies in the synthetic data

        Parameters:
        df: Synthetic dataframe to check
        reference_df: Original dataframe to use as reference (optional)

        Returns dictionary with anomaly results
        """
        results = {
            'univariate_anomalies': {},
            'multivariate_anomalies': {},
            'total_anomalies': 0,
            'anomaly_indices': []
        }

        # Detect univariate anomalies
        univariate_anomalies = self._detect_univariate_anomalies(df, reference_df)
        results['univariate_anomalies'] = univariate_anomalies

        # Detect multivariate anomalies
        multivariate_anomalies = self._detect_multivariate_anomalies(df, reference_df)
        results['multivariate_anomalies'] = multivariate_anomalies

        # Combine anomalies
        all_anomaly_indices = set()

        for column, indices in univariate_anomalies.items():
            all_anomaly_indices.update(indices)

        for anomaly_type, indices in multivariate_anomalies.items():
            all_anomaly_indices.update(indices)

        results['total_anomalies'] = len(all_anomaly_indices)
        results['anomaly_indices'] = sorted(list(all_anomaly_indices))

        return results

    def _detect_univariate_anomalies(self, df: pd.DataFrame,
                                     reference_df: Optional[pd.DataFrame] = None) -> Dict[str, List[int]]:
        """Detect univariate anomalies"""
        anomalies = {}

        for column in df.columns:
            column_type = self.schema.get(column, {}).get('type', 'unknown')

            if column_type == 'numeric':
                # For numeric columns, detect outliers
                anomaly_indices = self._detect_numeric_outliers(df, column, reference_df)
                if anomaly_indices:
                    anomalies[column] = anomaly_indices

            elif column_type == 'categorical':
                # For categorical columns, detect unusual categories
                anomaly_indices = self._detect_categorical_anomalies(df, column, reference_df)
                if anomaly_indices:
                    anomalies[column] = anomaly_indices

        return anomalies

    def _detect_numeric_outliers(self, df: pd.DataFrame, column: str,
                                 reference_df: Optional[pd.DataFrame] = None) -> List[int]:
        """Detect outliers in numeric column"""
        # If reference data is available, use its statistics
        if reference_df is not None and column in reference_df.columns:
            q1 = reference_df[column].quantile(0.25)
            q3 = reference_df[column].quantile(0.75)
        else:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)

        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        # Find outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        return list(df.index[outliers])

    def _detect_categorical_anomalies(self, df: pd.DataFrame, column: str,
                                      reference_df: Optional[pd.DataFrame] = None) -> List[int]:
        """Detect anomalies in categorical column"""
        # If reference data is available, identify categories not in reference
        if reference_df is not None and column in reference_df.columns:
            valid_categories = set(reference_df[column].unique())

            # Find records with categories not in reference data
            anomalies = ~df[column].isin(valid_categories)

            return list(df.index[anomalies])

        return []

    def _detect_multivariate_anomalies(self, df: pd.DataFrame,
                                       reference_df: Optional[pd.DataFrame] = None) -> Dict[str, List[int]]:
        """Detect multivariate anomalies"""
        anomalies = {}

        # Get numeric columns
        numeric_cols = [col for col in df.columns
                        if self.schema.get(col, {}).get('type') == 'numeric']

        if len(numeric_cols) >= 2:
            # Detect correlation anomalies
            correlation_anomalies = self._detect_correlation_anomalies(df, numeric_cols, reference_df)
            if correlation_anomalies:
                anomalies['correlation'] = correlation_anomalies

        return anomalies

    def _detect_correlation_anomalies(self, df: pd.DataFrame, columns: List[str],
                                      reference_df: Optional[pd.DataFrame] = None) -> List[int]:
        """
        Detect records that violate expected correlations
        This is a simplified implementation - more sophisticated techniques exist
        """
        if reference_df is None or len(columns) < 2:
            return []

        try:
            # Calculate correlation matrix from reference data
            ref_corr = reference_df[columns].corr()

            anomaly_indices = set()

            # Check pairs with strong correlations
            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    if i >= j:
                        continue

                    corr = ref_corr.loc[col1, col2]

                    # If strong correlation exists
                    if abs(corr) > 0.7:
                        # For positive correlation, check if values move in opposite directions
                        if corr > 0:
                            # Standardize both columns
                            df_std1 = (df[col1] - df[col1].mean()) / df[col1].std()
                            df_std2 = (df[col2] - df[col2].mean()) / df[col2].std()

                            # Identify records where standardized values have opposite signs
                            # and the difference is large
                            anomalies = (df_std1 * df_std2 < -1) & (abs(df_std1 - df_std2) > 3)
                            anomaly_indices.update(df.index[anomalies])

                        # For negative correlation, check if values move in same direction
                        elif corr < 0:
                            df_std1 = (df[col1] - df[col1].mean()) / df[col1].std()
                            df_std2 = (df[col2] - df[col2].mean()) / df[col2].std()

                            # Identify records where standardized values have same signs
                            # and the sum is large
                            anomalies = (df_std1 * df_std2 > 1) & (abs(df_std1 + df_std2) > 3)
                            anomaly_indices.update(df.index[anomalies])

            return list(anomaly_indices)

        except:
            return []