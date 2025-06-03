from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
import pandas as pd
import numpy as np
from typing import Dict, List, Union


class FeatureTransformer:
    def __init__(self, schema):
        self.schema = schema
        self.transformers = {}
        self.column_types = {}  # Store inferred types

    def fit(self, df: pd.DataFrame):
        """Fit transformation models to the data"""
        # First validate and potentially update schema types
        self._validate_and_update_schema(df)

        for column, info in self.schema.items():
            if column not in df.columns:
                continue

            col_type = info.get('type', 'unknown')
            self.column_types[column] = col_type

            if col_type == 'categorical':
                # For categorical columns, fit a one-hot encoder
                try:
                    # Convert to string to ensure compatibility
                    values = df[column].astype(str).values.reshape(-1, 1)
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit(values)
                    self.transformers[column] = {
                        'type': 'categorical',
                        'transformer': encoder,
                        'categories': encoder.categories_[0]
                    }
                except Exception as e:
                    print(f"Warning: Could not fit encoder for {column}: {str(e)}")
                    # Fallback to handling as text
                    self.transformers[column] = {
                        'type': 'text',
                        'unique_values': df[column].astype(str).unique().tolist()
                    }

            elif col_type == 'numeric':
                # Check if column is actually numeric
                if not pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        # Try to convert to numeric
                        numeric_values = pd.to_numeric(df[column], errors='coerce')
                        # If >20% are NaN after conversion, treat as categorical
                        if numeric_values.isna().mean() > 0.2:
                            print(
                                f"Warning: Column {column} marked as numeric contains many non-numeric values. Treating as categorical.")
                            # Update schema and recursively call fit
                            self.schema[column]['type'] = 'categorical'
                            return self.fit(df)
                        # Otherwise use converted values
                        df[column] = numeric_values
                    except:
                        print(
                            f"Warning: Column {column} marked as numeric contains non-numeric values. Treating as categorical.")
                        # Update schema and recursively call fit
                        self.schema[column]['type'] = 'categorical'
                        return self.fit(df)

                # Handle missing values
                missing_mask = df[column].isna()
                if missing_mask.any():
                    # Store missing value info for later reconstruction
                    self.transformers[column + '_missing'] = {
                        'type': 'indicator',
                        'missing_mask': missing_mask
                    }
                    # Fill missing with median for transformation
                    df[column] = df[column].fillna(df[column].median())

                # For numeric columns, fit appropriate transformers based on distribution
                try:
                    # Check if values are suitable for transformation
                    values = df[column].values.reshape(-1, 1)

                    if self._is_skewed(df[column]):
                        # Use power transform for skewed distributions
                        transformer = PowerTransformer(method='yeo-johnson')
                    else:
                        # Use standard scaling otherwise
                        transformer = StandardScaler()

                    transformer.fit(values)
                    self.transformers[column] = {
                        'type': 'numeric',
                        'transformer': transformer,
                        'original_stats': {
                            'mean': df[column].mean(),
                            'std': df[column].std(),
                            'min': df[column].min(),
                            'max': df[column].max()
                        }
                    }
                except Exception as e:
                    print(f"Warning: Could not fit transformer for {column}: {str(e)}")
                    # Fallback to simple normalization
                    min_val = df[column].min()
                    max_val = df[column].max()
                    self.transformers[column] = {
                        'type': 'numeric_simple',
                        'min': min_val,
                        'max': max_val,
                        'range': max_val - min_val if max_val > min_val else 1.0
                    }

            elif col_type == 'datetime' or col_type == 'date':
                # Convert to datetime
                try:
                    dt = pd.to_datetime(df[column], errors='coerce')
                    missing_mask = dt.isna()

                    if missing_mask.mean() > 0.5:  # If >50% can't be parsed as dates
                        print(f"Warning: Column {column} marked as datetime but contains many invalid values.")
                        # Update schema and recursively call fit
                        self.schema[column]['type'] = 'categorical'
                        return self.fit(df)

                    # Store original date range for validation
                    self.transformers[column] = {
                        'type': 'datetime',
                        'components': ['year', 'month', 'day', 'dayofweek'],
                        'min_date': dt.min(),
                        'max_date': dt.max()
                    }
                except Exception as e:
                    print(f"Warning: Could not process datetime column {column}: {str(e)}")
                    # Fallback to categorical
                    self.schema[column]['type'] = 'categorical'
                    return self.fit(df)

            elif col_type == 'text':
                # For text columns, just store basic stats
                self.transformers[column] = {
                    'type': 'text',
                    'unique_values': df[column].astype(str).unique().tolist()
                }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for model input with better error handling"""
        result = pd.DataFrame(index=df.index)
        df_copy = df.copy()  # Work on a copy to avoid modifying original

        for column, transformer_info in self.transformers.items():
            # Skip missing indicator columns and columns not in the dataframe
            if '_missing' in column or (column not in df_copy.columns and transformer_info['type'] != 'indicator'):
                continue

            try:
                if transformer_info['type'] == 'categorical':
                    # Convert to string to ensure compatibility
                    values = df_copy[column].astype(str).values.reshape(-1, 1)
                    encoder = transformer_info['transformer']
                    encoded = encoder.transform(values)

                    # Create column names for encoded features
                    encoded_columns = [
                        f"{column}_{cat}" for cat in transformer_info['categories']
                    ]

                    # Add encoded columns to result
                    for i, col in enumerate(encoded_columns):
                        result[col] = encoded[:, i]

                elif transformer_info['type'] == 'numeric':
                    # Ensure values are numeric
                    values = pd.to_numeric(df_copy[column], errors='coerce').fillna(0).values.reshape(-1, 1)
                    # Transform numeric with fitted transformer
                    transformer = transformer_info['transformer']
                    result[column] = transformer.transform(values).flatten()

                elif transformer_info['type'] == 'numeric_simple':
                    # Simple min-max normalization
                    values = pd.to_numeric(df_copy[column], errors='coerce').fillna(0)
                    min_val = transformer_info['min']
                    range_val = transformer_info['range']
                    if range_val > 0:
                        result[column] = (values - min_val) / range_val
                    else:
                        result[column] = 0  # Handle case where all values are the same

                elif transformer_info['type'] == 'datetime':
                    # Convert to datetime with error handling
                    dt = pd.to_datetime(df_copy[column], errors='coerce')
                    # Fill missing dates with median to avoid errors
                    dt = dt.fillna(dt.median())

                    # Extract datetime components
                    for component in transformer_info['components']:
                        if component == 'year':
                            result[f"{column}_year"] = dt.dt.year
                        elif component == 'month':
                            result[f"{column}_month"] = dt.dt.month
                        elif component == 'day':
                            result[f"{column}_day"] = dt.dt.day
                        elif component == 'dayofweek':
                            result[f"{column}_dayofweek"] = dt.dt.dayofweek
                        elif component == 'hour':
                            result[f"{column}_hour"] = dt.dt.hour

                elif transformer_info['type'] == 'indicator':
                    # Handle missing value indicators
                    if '_missing' in column:
                        original_col = column.replace('_missing', '')
                        if original_col in df_copy.columns:
                            result[column] = df_copy[original_col].isna().astype(int)

                elif transformer_info['type'] == 'text':
                    # For text, we might want to extract features, but for now, skip
                    pass

            except Exception as e:
                print(f"Warning: Error transforming column {column}: {str(e)}")
                # Skip problematic columns rather than failing the entire transform

        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform with improved error handling"""
        result = pd.DataFrame(index=df.index)

        # Create a mapping of transformed columns back to original columns
        column_mapping = {}
        for original_col, transformer_info in self.transformers.items():
            if '_missing' in original_col:
                continue  # Skip missing indicators in the mapping

            if transformer_info['type'] == 'categorical':
                # Map each one-hot column back to the original
                categories = transformer_info.get('categories', [])
                transformed_cols = []
                for cat in categories:
                    transformed_col = f"{original_col}_{cat}"
                    if transformed_col in df.columns:
                        transformed_cols.append(transformed_col)

                if transformed_cols:
                    column_mapping[original_col] = transformed_cols

            elif transformer_info['type'] in ['numeric', 'numeric_simple']:
                # Direct mapping for numeric columns
                if original_col in df.columns:
                    column_mapping[original_col] = [original_col]

            elif transformer_info['type'] == 'datetime':
                # Map datetime component columns
                components = transformer_info.get('components', [])
                date_columns = [f"{original_col}_{comp}" for comp in components]
                date_columns_present = [col for col in date_columns if col in df.columns]
                if date_columns_present:
                    column_mapping[original_col] = date_columns_present

        # Perform inverse transformation
        for original_col, transformed_cols in column_mapping.items():
            if original_col not in self.transformers:
                continue  # Skip if we don't have a transformer

            transformer_info = self.transformers[original_col]

            try:
                if transformer_info['type'] == 'categorical':
                    # Reconstruct categorical from one-hot
                    if not transformed_cols:
                        continue  # Skip if no columns found

                    one_hot_values = df[transformed_cols].values
                    categories = transformer_info.get('categories', [])

                    if len(transformed_cols) == len(categories):
                        # If all categories present, get index of max value
                        cat_indices = np.argmax(one_hot_values, axis=1)
                        result[original_col] = [
                            categories[idx] if 0 <= idx < len(categories) else "Unknown"
                            for idx in cat_indices
                        ]
                    else:
                        # If some categories missing, use presence of 1 to determine category
                        result[original_col] = "Unknown"  # Default
                        for i, col in enumerate(transformed_cols):
                            if i < len(categories):
                                cat = categories[i]
                                mask = df[col] >= 0.5  # Use threshold for one-hot
                                result.loc[mask, original_col] = cat

                elif transformer_info['type'] == 'numeric':
                    # Inverse transform numeric columns
                    transformer = transformer_info['transformer']
                    values = df[transformed_cols].values.reshape(-1, 1)
                    result[original_col] = transformer.inverse_transform(values).flatten()

                elif transformer_info['type'] == 'numeric_simple':
                    # Reverse min-max normalization
                    min_val = transformer_info['min']
                    range_val = transformer_info['range']
                    values = df[transformed_cols].values.flatten()
                    result[original_col] = values * range_val + min_val

                elif transformer_info['type'] == 'datetime':
                    # Reconstruct datetime from components
                    components = {}
                    for col in transformed_cols:
                        component = col.split('_')[-1]
                        components[component] = df[col]

                    # Create datetime objects with available components
                    if 'year' in components:
                        date_parts = {
                            'year': components['year'],
                            'month': components.get('month', 1),
                            'day': components.get('day', 1),
                            'hour': components.get('hour', 0)
                        }
                        result[original_col] = pd.to_datetime(date_parts, errors='coerce')

                # Apply missing value indicators if available
                missing_col = original_col + '_missing'
                if missing_col in self.transformers and missing_col in df.columns:
                    missing_mask = df[missing_col] > 0.5
                    result.loc[missing_mask, original_col] = np.nan

            except Exception as e:
                print(f"Warning: Error in inverse transform for {original_col}: {str(e)}")
                # Continue with other columns rather than failing completely

        return result

    def _is_skewed(self, series):
        """Determine if a numeric series has a skewed distribution"""
        try:
            # Filter out missing values
            series = series.dropna()
            if len(series) < 10:  # Need enough data to calculate skewness
                return False

            skewness = series.skew()
            return abs(skewness) > 0.5  # Threshold for considering a distribution skewed
        except:
            return False

    def _validate_and_update_schema(self, df):
        """Validate schema against actual data and update if needed"""
        for column in df.columns:
            # If column isn't in schema, add it with inferred type
            if column not in self.schema:
                self.schema[column] = {'type': self._infer_column_type(df[column])}
                print(f"Added missing column {column} to schema with type {self.schema[column]['type']}")
            else:
                # Validate existing schema type
                schema_type = self.schema[column].get('type', 'unknown')
                if schema_type == 'numeric' and not self._can_be_numeric(df[column]):
                    print(
                        f"Warning: Column {column} marked as numeric but contains non-numeric values. Updating schema.")
                    self.schema[column]['type'] = self._infer_column_type(df[column])
                elif schema_type == 'datetime' and not self._can_be_datetime(df[column]):
                    print(f"Warning: Column {column} marked as datetime but contains invalid values. Updating schema.")
                    self.schema[column]['type'] = self._infer_column_type(df[column])