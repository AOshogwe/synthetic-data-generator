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

class DMDConsistencyValidator:
    """
    Checks a set of tables against the TREAT-NMD DMD Core Dataset's documented
    consistency rules (see DMD_schema_analysis.md, section 4). Matching is by
    column/table NAME, so this is a safe no-op on datasets that don't use DMD
    naming -- a rule simply produces no findings if its columns aren't present.
    Kept independent of DataValidator/pipeline.py (no cross-import) since
    several of these rules are inherently cross-table (e.g. a patient's
    Genetic confirmation flag living in one table, their Genetic report
    living in another).
    """

    @staticmethod
    def _find_column(df, *name_options):
        lower_map = {c.strip().lower(): c for c in df.columns}
        for name in name_options:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    def validate(self, tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """tables: {table_name: dataframe}. Returns a list of violation dicts:
        {'rule', 'table', 'row_indices', 'message'}."""
        violations = []
        violations += self._check_death_fields_require_not_alive(tables)
        violations += self._check_cause_of_death_classification(tables)
        violations += self._check_episode_exactly_one_stop_or_ongoing(tables)
        violations += self._check_stopping_reason_requires_not_ongoing(tables)
        violations += self._check_hospitalisation_nights_positive(tables)
        violations += self._check_muscle_biopsy_biobank(tables)
        violations += self._check_genetic_confirmation_requires_report(tables)
        violations += self._check_cardiac_treatment_xor_other_drug(tables)
        return violations

    def _check_death_fields_require_not_alive(self, tables):
        """Date of death / Cause of death code may only be provided if Alive = No."""
        violations = []
        for table_name, df in tables.items():
            alive_col = self._find_column(df, 'Alive')
            dod_col = self._find_column(df, 'Date of death')
            cod_col = self._find_column(df, 'Cause of death code')
            if not alive_col or not (dod_col or cod_col):
                continue
            not_no = df[alive_col].astype(str).str.strip().str.lower() != 'no'
            for col in (dod_col, cod_col):
                if not col:
                    continue
                bad = df[not_no & df[col].notna()]
                if not bad.empty:
                    violations.append({
                        'rule': 'death_fields_require_not_alive',
                        'table': table_name,
                        'row_indices': bad.index.tolist(),
                        'message': f"{len(bad)} row(s) have '{col}' set but '{alive_col}' is not 'No'"
                    })
        return violations

    def _check_cause_of_death_classification(self, tables):
        """Cause of death classification must be provided if Cause of death code is."""
        violations = []
        for table_name, df in tables.items():
            code_col = self._find_column(df, 'Cause of death code')
            class_col = self._find_column(df, 'Cause of death classification')
            if not code_col or not class_col:
                continue
            bad = df[df[code_col].notna() & df[class_col].isna()]
            if not bad.empty:
                violations.append({
                    'rule': 'cause_of_death_classification_required',
                    'table': table_name,
                    'row_indices': bad.index.tolist(),
                    'message': f"{len(bad)} row(s) have '{code_col}' set but '{class_col}' is missing"
                })
        return violations

    def _check_episode_exactly_one_stop_or_ongoing(self, tables):
        """Every episode-shaped table must have exactly one of Stop date /
        Ongoing date populated per row, never both, never neither."""
        violations = []
        for table_name, df in tables.items():
            start_col = self._find_column(df, 'Start date')
            stop_col = self._find_column(df, 'Stop date')
            ongoing_col = self._find_column(df, 'Ongoing date')
            if not start_col or not (stop_col or ongoing_col):
                continue
            cols = [c for c in (stop_col, ongoing_col) if c]
            populated_count = df[cols].notna().sum(axis=1)
            bad = df[populated_count != 1]
            if not bad.empty:
                violations.append({
                    'rule': 'episode_exactly_one_stop_or_ongoing',
                    'table': table_name,
                    'row_indices': bad.index.tolist(),
                    'message': f"{len(bad)} row(s) don't have exactly one of {cols} populated"
                })
        return violations

    def _check_stopping_reason_requires_not_ongoing(self, tables):
        """A stopping reason can't be set for a therapy that's still ongoing."""
        violations = []
        reason_names = [
            'Corticosteroid stopping reason',
            'Cardiac treatment stopping reason',
            'Allopathic drug stopping reason',
        ]
        for table_name, df in tables.items():
            ongoing_col = self._find_column(df, 'Ongoing date')
            if not ongoing_col:
                continue
            for reason_name in reason_names:
                reason_col = self._find_column(df, reason_name)
                if not reason_col:
                    continue
                bad = df[df[ongoing_col].notna() & df[reason_col].notna()]
                if not bad.empty:
                    violations.append({
                        'rule': 'stopping_reason_requires_not_ongoing',
                        'table': table_name,
                        'row_indices': bad.index.tolist(),
                        'message': f"{len(bad)} row(s) have '{reason_col}' set while '{ongoing_col}' is also set"
                    })
        return violations

    def _check_hospitalisation_nights_positive(self, tables):
        """Hospitalisation nights must be positive."""
        violations = []
        for table_name, df in tables.items():
            nights_col = self._find_column(df, 'Hospitalisation nights')
            if not nights_col:
                continue
            numeric = pd.to_numeric(df[nights_col], errors='coerce')
            bad = df[numeric.notna() & (numeric <= 0)]
            if not bad.empty:
                violations.append({
                    'rule': 'hospitalisation_nights_positive',
                    'table': table_name,
                    'row_indices': bad.index.tolist(),
                    'message': f"{len(bad)} row(s) have non-positive '{nights_col}'"
                })
        return violations

    def _check_muscle_biopsy_biobank(self, tables):
        """Muscle biopsy biobank requires Muscle biopsy stored in biobank = Yes."""
        violations = []
        for table_name, df in tables.items():
            biobank_col = self._find_column(df, 'Muscle biopsy biobank')
            stored_col = self._find_column(df, 'Muscle biopsy stored in biobank')
            if not biobank_col or not stored_col:
                continue
            stored_yes = df[stored_col].astype(str).str.strip().str.lower() == 'yes'
            bad = df[df[biobank_col].notna() & ~stored_yes]
            if not bad.empty:
                violations.append({
                    'rule': 'muscle_biopsy_biobank_requires_stored_yes',
                    'table': table_name,
                    'row_indices': bad.index.tolist(),
                    'message': f"{len(bad)} row(s) have '{biobank_col}' set but '{stored_col}' isn't 'Yes'"
                })
        return violations

    def _check_genetic_confirmation_requires_report(self, tables):
        """Cross-table: Genetic confirmation must be Yes for any patient who
        has a Genetic report record in another linked table."""
        violations = []
        singular_table, singular_df, confirm_col, key_col = None, None, None, None
        for table_name, df in tables.items():
            c = self._find_column(df, 'Genetic confirmation')
            k = self._find_column(df, 'Patient ID')
            if c and k:
                singular_table, singular_df, confirm_col, key_col = table_name, df, c, k
                break
        if singular_df is None:
            return violations

        for table_name, df in tables.items():
            if table_name == singular_table:
                continue
            report_key = self._find_column(df, 'Patient ID')
            if not report_key or 'genetic report' not in table_name.lower().replace('_', ' '):
                continue
            patients_with_report = set(df[report_key].dropna().unique())
            mask = singular_df[key_col].isin(patients_with_report) & (
                singular_df[confirm_col].astype(str).str.strip().str.lower() != 'yes'
            )
            bad = singular_df[mask]
            if not bad.empty:
                violations.append({
                    'rule': 'genetic_confirmation_requires_report',
                    'table': singular_table,
                    'row_indices': bad.index.tolist(),
                    'message': f"{len(bad)} patient(s) have a genetic report but '{confirm_col}' isn't 'Yes'"
                })
        return violations

    def _check_cardiac_treatment_xor_other_drug(self, tables):
        """Cardiac treatment and Other cardiac drug can't both be set in the same episode."""
        violations = []
        for table_name, df in tables.items():
            treatment_col = self._find_column(df, 'Cardiac treatment')
            other_col = self._find_column(df, 'Other cardiac drug')
            if not treatment_col or not other_col:
                continue
            bad = df[df[treatment_col].notna() & df[other_col].notna()]
            if not bad.empty:
                violations.append({
                    'rule': 'cardiac_treatment_xor_other_drug',
                    'table': table_name,
                    'row_indices': bad.index.tolist(),
                    'message': f"{len(bad)} row(s) have both '{treatment_col}' and '{other_col}' set"
                })
        return violations
