import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import QuantileTransformer


class GaussianCopula:
    """
    Gaussian Copula model for synthetic data generation
    """

    def __init__(self, schema):
        self.schema = schema
        self.columns = None
        self.correlation_matrix = None
        self.transformers = {}
        self.categorical_encoders = {}

    def fit(self, df: pd.DataFrame):
        """Fit Gaussian Copula to the data"""
        # Store columns
        self.columns = df.columns.tolist()

        # If schema is empty or doesn't contain proper type information, infer types
        if not self.schema:
            self.schema = {}
            for column in self.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    self.schema[column] = {'type': 'numeric'}
                else:
                    self.schema[column] = {'type': 'categorical'}

        # Process each column
        uniform_data = {}

        for column in self.columns:
            # Get column type from schema, default to categorical for safety
            column_type = self.schema.get(column, {}).get('type', 'categorical')

            # Print column info for debugging
            print(f"Processing column '{column}' with type '{column_type}'")

            # Handle categorical columns
            if column_type == 'categorical':
                # Encode categorical values to integers
                categorical_values = df[column].astype('category')
                self.categorical_encoders[column] = {
                    'categories': categorical_values.cat.categories.tolist(),
                    'codes': {cat: code for code, cat in enumerate(categorical_values.cat.categories)}
                }

                # Convert to uniform using empirical CDF
                codes = categorical_values.cat.codes.values
                # Add small noise to avoid exact zeros in correlation
                codes = codes + np.random.uniform(-0.01, 0.01, size=len(codes))
                # Scale to [0, 1]
                uniform_data[column] = (codes - codes.min()) / (codes.max() - codes.min())

            # Handle continuous columns
            elif column_type == 'numeric':
                try:
                    # Create transformer for continuous variables
                    transformer = QuantileTransformer(output_distribution='normal')
                    transformed = transformer.fit_transform(df[[column]]).flatten()
                    self.transformers[column] = transformer

                    # Convert to uniform using the standard normal CDF
                    uniform_data[column] = stats.norm.cdf(transformed)
                except ValueError as e:
                    print(f"Error processing numeric column '{column}': {str(e)}")
                    print(f"Sample values: {df[column].head().tolist()}")
                    print(f"Converting column '{column}' to categorical")

                    # Fall back to categorical handling if numeric conversion fails
                    categorical_values = df[column].astype('category')
                    self.categorical_encoders[column] = {
                        'categories': categorical_values.cat.categories.tolist(),
                        'codes': {cat: code for code, cat in enumerate(categorical_values.cat.categories)}
                    }

                    codes = categorical_values.cat.codes.values
                    codes = codes + np.random.uniform(-0.01, 0.01, size=len(codes))
                    uniform_data[column] = (codes - codes.min()) / (codes.max() - codes.min())

            # Handle datetime columns - convert to numeric then handle like continuous
            elif column_type == 'datetime':
                # Convert datetime to numeric (timestamp)
                timestamps = pd.to_datetime(df[column]).astype(np.int64) // 10 ** 9  # Convert to seconds

                # Create transformer
                transformer = QuantileTransformer(output_distribution='normal')
                transformed = transformer.fit_transform(timestamps.values.reshape(-1, 1)).flatten()
                self.transformers[column] = transformer

                # Store datetime information
                self.categorical_encoders[column] = {
                    'type': 'datetime',
                    'min': pd.to_datetime(df[column]).min(),
                    'max': pd.to_datetime(df[column]).max()
                }

                # Convert to uniform
                uniform_data[column] = stats.norm.cdf(transformed)

        # Create uniform dataframe
        uniform_df = pd.DataFrame(uniform_data, index=df.index)

        # Transform to normal using inverse CDF
        normal_data = {}
        for column in self.columns:
            normal_data[column] = stats.norm.ppf(uniform_df[column])

        normal_df = pd.DataFrame(normal_data, index=df.index)

        # Calculate correlation matrix of normal data
        self.correlation_matrix = normal_df.corr().values

        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples using the Gaussian Copula"""
        # Generate multivariate normal samples
        mvn_samples = np.random.multivariate_normal(
            mean=np.zeros(len(self.columns)),
            cov=self.correlation_matrix,
            size=n_samples
        )

        # Convert to DataFrame
        normal_df = pd.DataFrame(mvn_samples, columns=self.columns)

        # Transform back to uniform using standard normal CDF
        uniform_df = pd.DataFrame(index=range(n_samples))
        for column in self.columns:
            uniform_df[column] = stats.norm.cdf(normal_df[column])

        # Transform from uniform back to original distributions
        result = pd.DataFrame(index=range(n_samples))

        for column in self.columns:
            # Check if column is in transformers (numeric) or categorical_encoders (categorical)
            if column in self.transformers:
                # Handle numeric columns
                transformer = self.transformers[column]
                # First convert uniform to normal
                normal_values = stats.norm.ppf(uniform_df[column])
                # Then inverse transform from normal to original
                original_values = transformer.inverse_transform(
                    normal_values.reshape(-1, 1)
                ).flatten()
                result[column] = original_values
            elif column in self.categorical_encoders:
                # Handle categorical columns
                encoder = self.categorical_encoders[column]
                categories = encoder['categories']

                # Scale uniform values to the range of category codes
                scaled = uniform_df[column] * (len(categories) - 1)

                # Round to nearest integer
                codes = np.round(scaled).astype(int)

                # Clip to ensure valid indices
                codes = np.clip(codes, 0, len(categories) - 1)

                # Map back to original categories
                result[column] = [categories[code] for code in codes]
            else:
                # Handle case where column is neither in transformers nor categorical_encoders
                print(
                    f"Warning: Column '{column}' not found in transformers or categorical_encoders. Using random values.")
                if len(self.columns) > 0:
                    # Copy values from another column as fallback
                    other_col = self.columns[0]
                    result[column] = result[other_col] if other_col in result else np.zeros(n_samples)
                else:
                    # No other columns available, use zeros
                    result[column] = np.zeros(n_samples)

        return result
    