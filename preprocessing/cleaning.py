import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Dict, List


class DataCleaningPipeline:
    def __init__(self, schema):
        self.schema = schema
        self.transformers = {}

    def fit(self, df: pd.DataFrame):
        """Fit preprocessing transformers to data"""
        for column, info in self.schema.items():
            if info.get('type', '') == 'numeric':
                self.transformers[column] = {
                    'imputer': SimpleImputer(strategy='mean').fit(df[[column]]),
                    'outlier_params': self._get_outlier_params(df[column])
                }
            elif info.get('type', '') == 'categorical':
                self.transformers[column] = {
                    'imputer': SimpleImputer(strategy='most_frequent').fit(df[[column]]),
                    'categories': df[column].dropna().unique().tolist()
                }
        for column, info in self.schema.items():
            if info.get('type', '') == 'numeric':
                # Convert to numeric and handle errors
                df[column] = pd.to_numeric(df[column], errors='coerce')
                # Replace NaN with 0 or mean or other strategy
                df[column] = df[column].fillna(df[column].mean())

        for col in df.columns:
            print(f"Column {col} types: {df[col].apply(type).value_counts()}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations"""
        result = df.copy()

        for column, transformer in self.transformers.items():
            if column not in result.columns:
                continue

            # Apply imputation
            result[column] = transformer['imputer'].transform(result[[column]])

            # Handle outliers for numeric columns
            if self.schema[column]['type'] == 'numeric':
                outlier_params = transformer['outlier_params']
                mask = (result[column] < outlier_params['lower']) | (result[column] > outlier_params['upper'])
                if mask.any():
                    # Either clip or remove outliers based on strategy
                    result.loc[mask, column] = np.clip(
                        result.loc[mask, column],
                        outlier_params['lower'],
                        outlier_params['upper']
                    )

            # Validate categorical values
            elif self.schema[column]['type'] == 'categorical':
                valid_categories = transformer['categories']
                result[column] = result[column].apply(
                    lambda x: x if x in valid_categories else valid_categories[0]
                )

        return result

    def _get_outlier_params(self, series):
        """Calculate parameters for outlier detection using IQR method"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        return {
            'lower': q1 - 1.5 * iqr,
            'upper': q3 + 1.5 * iqr
        }
