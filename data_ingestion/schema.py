import pandas as pd
from typing import Dict, List, Tuple


class SchemaInferenceEngine:
    def infer_schema(self, df: pd.DataFrame) -> Dict:
        """Infer schema from dataframe"""
        schema = {}

        for column in df.columns:
            data_type = self._infer_column_type(df[column])
            schema[column] = {
                'type': data_type,
                'nullable': df[column].isnull().any(),
                'unique': df[column].is_unique,
                'stats': self._get_column_stats(df[column], data_type)
            }

        return schema

    def _infer_column_type(self, series: pd.Series) -> str:
        """Improved column type inference"""
        # Check for date columns first
        if pd.api.types.is_object_dtype(series):
            date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']
            sample = series.dropna().head(10).astype(str)
            for date_format in date_formats:
                try:
                    success = 0
                    for val in sample:
                        try:
                            pd.to_datetime(val, format=date_format)
                            success += 1
                        except:
                            pass
                    if success / len(sample) >= 0.7:  # If 70% match the format
                        return 'date'
                except:
                    pass

        # Check numeric types
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'

        # Check categorical with small number of unique values
        n_unique = series.nunique()
        if n_unique <= 20 or (n_unique / len(series) < 0.05):
            return 'categorical'

        # Default to text for other types
        return 'text'

    def _get_column_stats(self, series: pd.Series, data_type: str) -> Dict:
        """Get statistics for a column based on its type"""
        stats = {}

        if data_type == 'numeric':
            stats['min'] = float(series.min()) if not pd.isna(series.min()) else None
            stats['max'] = float(series.max()) if not pd.isna(series.max()) else None
            stats['mean'] = float(series.mean()) if not pd.isna(series.mean()) else None
            stats['median'] = float(series.median()) if not pd.isna(series.median()) else None
            stats['std'] = float(series.std()) if not pd.isna(series.std()) else None

        elif data_type == 'categorical':
            # Get value counts for categorical data
            value_counts = series.value_counts(normalize=True)
            stats['categories'] = value_counts.index.tolist()
            stats['frequencies'] = value_counts.values.tolist()

        elif data_type == 'datetime':
            stats['min'] = series.min().isoformat() if not pd.isna(series.min()) else None
            stats['max'] = series.max().isoformat() if not pd.isna(series.max()) else None

        # Add count of missing values
        stats['missing_count'] = int(series.isnull().sum())
        stats['missing_percentage'] = float(series.isnull().mean() * 100)

        return stats