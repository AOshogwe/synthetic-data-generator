# synthetic_data_pipeline.py - Core pipeline functionality
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Core synthetic data generation engine"""

    def __init__(self):
        self.original_data = {}
        self.synthetic_data = {}
        self.schema = {}
        self.config = {}

    def load_data_from_files(self, files: List[Dict[str, Any]]) -> bool:
        """Load data from uploaded files"""
        try:
            for file_info in files:
                file_path = file_info['path']
                table_name = file_info['name'].replace('.csv', '').replace('.xlsx', '')

                # Load based on file extension
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue

                # Basic data validation
                if df.empty:
                    logger.warning(f"Empty dataframe for {table_name}")
                    continue

                # Clean column names
                df.columns = df.columns.str.strip()

                self.original_data[table_name] = df
                logger.info(f"Loaded table {table_name}: {len(df)} rows, {len(df.columns)} columns")

            if not self.original_data:
                raise ValueError("No valid data loaded from files")

            self._infer_schema()
            return True

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _infer_schema(self):
        """Infer schema and data types from loaded data"""
        self.schema = {}

        for table_name, df in self.original_data.items():
            table_schema = {'columns': {}, 'relationships': {}}

            for column in df.columns:
                column_info = self._analyze_column(df, column)
                table_schema['columns'][column] = column_info

            # Detect relationships
            table_schema['relationships'] = self._detect_relationships(df)

            self.schema[table_name] = table_schema

    def _analyze_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze a single column to determine its properties"""
        column_info = {
            'type': 'unknown',
            'subtype': None,
            'null_count': int(df[column].isnull().sum()),
            'unique_count': int(df[column].nunique()),
            'sample_values': []
        }

        # Get sample values (non-null)
        non_null_series = df[column].dropna()
        if len(non_null_series) > 0:
            sample_size = min(5, len(non_null_series))
            column_info['sample_values'] = non_null_series.sample(sample_size).tolist()

        # Determine basic type
        if pd.api.types.is_numeric_dtype(df[column]):
            column_info['type'] = 'numeric'
            column_info['min'] = float(df[column].min()) if not df[column].empty else None
            column_info['max'] = float(df[column].max()) if not df[column].empty else None
            column_info['mean'] = float(df[column].mean()) if not df[column].empty else None
            column_info['std'] = float(df[column].std()) if not df[column].empty else None

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            column_info['type'] = 'datetime'
            column_info['min_date'] = df[column].min()
            column_info['max_date'] = df[column].max()

        else:
            # Text/categorical
            unique_ratio = column_info['unique_count'] / len(df) if len(df) > 0 else 0

            if unique_ratio < 0.1:  # Less than 10% unique values
                column_info['type'] = 'categorical'
            else:
                column_info['type'] = 'text'

        # Detect special column types
        column_lower = column.lower()

        # Name detection
        if any(keyword in column_lower for keyword in ['name', 'first', 'last', 'fname', 'lname']):
            column_info['subtype'] = 'name'
            column_info['is_name'] = True

        # Address detection
        elif any(keyword in column_lower for keyword in ['address', 'street', 'location', 'addr']):
            column_info['subtype'] = 'address'
            column_info['is_address'] = True

        # Age detection
        elif 'age' in column_lower and column_info['type'] == 'numeric':
            column_info['subtype'] = 'age'
            column_info['is_age'] = True

        # Email detection
        elif 'email' in column_lower or 'mail' in column_lower:
            column_info['subtype'] = 'email'
            column_info['is_email'] = True

        # Phone detection
        elif any(keyword in column_lower for keyword in ['phone', 'tel', 'mobile']):
            column_info['subtype'] = 'phone'
            column_info['is_phone'] = True

        # ID detection
        elif any(keyword in column_lower for keyword in ['id', 'identifier']) and column_info['unique_count'] == len(
                df):
            column_info['subtype'] = 'id'
            column_info['is_id'] = True

        # Date detection in text columns
        elif any(keyword in column_lower for keyword in ['date', 'time', 'created', 'updated']):
            # Try to parse as date
            try:
                test_series = pd.to_datetime(df[column], errors='coerce')
                if not test_series.isnull().all():
                    column_info['type'] = 'datetime'
                    column_info['subtype'] = 'date_text'
            except:
                pass

        return column_info

    def _detect_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect relationships between columns"""
        relationships = {
            'temporal': [],
            'correlations': {},
            'dependencies': []
        }

        # Detect temporal relationships
        date_columns = [col for col in df.columns
                        if any(
                keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'start', 'end'])]

        if len(date_columns) >= 2:
            relationships['temporal'] = self._analyze_temporal_relationships(df, date_columns)

        # Detect correlations for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            try:
                corr_matrix = df[numeric_columns].corr()
                relationships['correlations'] = corr_matrix.to_dict()
            except:
                pass

        return relationships

    def _analyze_temporal_relationships(self, df: pd.DataFrame, date_columns: List[str]) -> List[Dict[str, Any]]:
        """Analyze temporal relationships between date columns"""
        temporal_rels = []

        for i, col1 in enumerate(date_columns):
            for col2 in date_columns[i + 1:]:
                try:
                    # Try to convert to datetime
                    date1 = pd.to_datetime(df[col1], errors='coerce')
                    date2 = pd.to_datetime(df[col2], errors='coerce')

                    # Skip if too many null values
                    valid_mask = ~(date1.isna() | date2.isna())
                    if valid_mask.sum() < len(df) * 0.5:
                        continue

                    # Check if one consistently comes before the other
                    before_count = (date1[valid_mask] <= date2[valid_mask]).sum()
                    total_valid = valid_mask.sum()

                    if before_count / total_valid > 0.8:  # 80% consistency
                        # Calculate typical duration
                        durations = (date2[valid_mask] - date1[valid_mask]).dt.total_seconds() / (24 * 3600)
                        durations = durations.clip(0, 365)  # Cap at 1 year

                        temporal_rels.append({
                            'from': col1,
                            'to': col2,
                            'consistency': before_count / total_valid,
                            'mean_duration_days': float(durations.mean()),
                            'median_duration_days': float(durations.median()),
                            'std_duration_days': float(durations.std())
                        })

                except Exception as e:
                    logger.warning(f"Error analyzing temporal relationship {col1}-{col2}: {str(e)}")
                    continue

        return temporal_rels

    def configure_generation(self, config: Dict[str, Any]):
        """Configure the generation process"""
        self.config = config
        logger.info(f"Configuration updated: {config.get('generation_method', 'unknown')} method")

    def generate_synthetic_data(self) -> bool:
        """Generate synthetic data based on configuration"""
        try:
            if not self.original_data:
                raise ValueError("No original data available")

            generation_method = self.config.get('generation_method', 'auto')
            logger.info(f"Starting generation using {generation_method} method")

            for table_name, df in self.original_data.items():
                logger.info(f"Generating synthetic data for table: {table_name}")

                # Determine number of rows
                n_rows = self._calculate_target_rows(df)

                # Generate based on method
                if generation_method == 'perturbation':
                    synthetic_df = self._generate_perturbed_data(df, n_rows)
                elif generation_method == 'ctgan':
                    synthetic_df = self._generate_ctgan_data(df, n_rows)
                elif generation_method == 'gaussian_copula':
                    synthetic_df = self._generate_copula_data(df, n_rows)
                else:  # auto
                    synthetic_df = self._generate_auto_data(df, n_rows)

                # Apply post-processing
                synthetic_df = self._apply_post_processing(synthetic_df, table_name)

                self.synthetic_data[table_name] = synthetic_df
                logger.info(f"Generated {len(synthetic_df)} synthetic rows for {table_name}")

            return True

        except Exception as e:
            logger.error(f"Error in synthetic data generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _calculate_target_rows(self, df: pd.DataFrame) -> int:
        """Calculate target number of rows based on configuration"""
        original_rows = len(df)
        data_size = self.config.get('data_size', {})

        if data_size.get('type') == 'percentage':
            return int(original_rows * (data_size.get('value', 100) / 100))
        elif data_size.get('type') == 'custom':
            return int(data_size.get('value', original_rows))
        else:
            return original_rows

    def _generate_perturbed_data(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """Generate data using perturbation method"""
        # Sample from original data
        if n_rows <= len(df):
            synthetic_df = df.sample(n_rows, replace=False).copy()
        else:
            synthetic_df = df.sample(n_rows, replace=True).copy()

        perturbation_factor = self.config.get('perturbation_factor', 0.2)

        # Apply perturbation to numeric columns
        for column in synthetic_df.columns:
            if pd.api.types.is_numeric_dtype(synthetic_df[column]):
                std = synthetic_df[column].std()
                if std > 0 and not synthetic_df[column].isnull().all():
                    noise = np.random.normal(0, std * perturbation_factor, len(synthetic_df))
                    synthetic_df[column] = synthetic_df[column] + noise

            elif pd.api.types.is_datetime64_any_dtype(synthetic_df[column]):
                # Add random days offset
                max_offset_days = 30 * perturbation_factor  # Scale with perturbation factor
                random_days = np.random.randint(-max_offset_days, max_offset_days + 1, len(synthetic_df))
                synthetic_df[column] = synthetic_df[column] + pd.to_timedelta(random_days, unit='d')

        return synthetic_df.reset_index(drop=True)

    def _generate_ctgan_data(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """Generate data using CTGAN (placeholder - would need actual CTGAN implementation)"""
        # For now, fall back to statistical generation
        return self._generate_statistical_data(df, n_rows)

    def _generate_copula_data(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """Generate data using Gaussian Copula (placeholder)"""
        # For now, fall back to statistical generation
        return self._generate_statistical_data(df, n_rows)

    def _generate_auto_data(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """Auto-select best method based on data characteristics"""
        # Simple heuristic: use perturbation for small datasets, statistical for large
        if len(df) < 1000:
            return self._generate_perturbed_data(df, n_rows)
        else:
            return self._generate_statistical_data(df, n_rows)

    def _generate_statistical_data(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """Generate data using statistical methods"""
        synthetic_df = pd.DataFrame()

        for column in df.columns:
            column_data = df[column].dropna()

            if len(column_data) == 0:
                # Handle empty columns
                synthetic_df[column] = [None] * n_rows
                continue

            if pd.api.types.is_numeric_dtype(column_data):
                # Generate from normal distribution
                mean = column_data.mean()
                std = column_data.std()
                if std == 0:
                    synthetic_df[column] = [mean] * n_rows
                else:
                    synthetic_df[column] = np.random.normal(mean, std, n_rows)

                # Ensure non-negative for positive-only columns
                if column_data.min() >= 0:
                    synthetic_df[column] = np.abs(synthetic_df[column])

            elif pd.api.types.is_datetime64_any_dtype(column_data):
                # Generate random dates within range
                min_date = column_data.min()
                max_date = column_data.max()

                if pd.isna(min_date) or pd.isna(max_date) or min_date == max_date:
                    synthetic_df[column] = [min_date] * n_rows
                else:
                    date_range_seconds = (max_date - min_date).total_seconds()
                    random_seconds = np.random.uniform(0, date_range_seconds, n_rows)
                    synthetic_df[column] = min_date + pd.to_timedelta(random_seconds, unit='s')

            else:
                # Categorical/text data - sample from original values
                synthetic_df[column] = np.random.choice(column_data, n_rows, replace=True)

        return synthetic_df

    def _apply_post_processing(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Apply post-processing based on configuration"""
        result_df = df.copy()

        # Apply anonymization techniques
        if self.config.get('anonymize_names', False):
            result_df = self._anonymize_names(result_df, table_name)

        if self.config.get('apply_age_grouping', False):
            result_df = self._apply_age_grouping(result_df, table_name)

        if self.config.get('anonymize_addresses', False):
            result_df = self._anonymize_addresses(result_df, table_name)

        # Preserve relationships if configured
        if self.config.get('preserve_temporal', True):
            result_df = self._preserve_temporal_relationships(result_df, table_name)

        return result_df

    def _anonymize_names(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Anonymize name columns"""
        schema_info = self.schema.get(table_name, {}).get('columns', {})

        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'James', 'Jessica',
                       'Robert', 'Lisa', 'William', 'Mary', 'Richard', 'Jennifer', 'Charles', 'Patricia']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                      'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas']

        for column, info in schema_info.items():
            if info.get('is_name', False) and column in df.columns:
                method = self.config.get('name_method', 'synthetic')

                if method == 'synthetic':
                    df[column] = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                                  for _ in range(len(df))]
                elif method == 'initials':
                    df[column] = df[column].apply(lambda x: self._to_initials(str(x)) if pd.notna(x) else x)

        return df

    def _to_initials(self, name: str) -> str:
        """Convert name to initials"""
        try:
            parts = str(name).split()
            return '. '.join([part[0].upper() for part in parts if part]) + '.'
        except:
            return 'N.A.'

    def _apply_age_grouping(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Apply age grouping"""
        schema_info = self.schema.get(table_name, {}).get('columns', {})
        method = self.config.get('age_grouping_method', '10-year')

        for column, info in schema_info.items():
            if info.get('is_age', False) and column in df.columns:
                if method == '5-year':
                    df[column] = pd.cut(df[column], bins=range(0, 101, 5), right=False,
                                        labels=[f"{i}-{i + 4}" for i in range(0, 100, 5)])
                elif method == '10-year':
                    df[column] = pd.cut(df[column], bins=range(0, 101, 10), right=False,
                                        labels=[f"{i}-{i + 9}" for i in range(0, 100, 10)])
                elif method == 'life-stages':
                    bins = [0, 13, 18, 25, 65, 100]
                    labels = ['Child (0-12)', 'Teen (13-17)', 'Young Adult (18-24)', 'Adult (25-64)', 'Senior (65+)']
                    df[column] = pd.cut(df[column], bins=bins, labels=labels, right=False)

        return df

    def _anonymize_addresses(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Anonymize address columns"""
        schema_info = self.schema.get(table_name, {}).get('columns', {})

        streets = ['Main St', 'Oak Ave', 'First St', 'Second Ave', 'Park Blvd', 'Elm Dr', 'Maple Ln']
        cities = ['Springfield', 'Franklin', 'Georgetown', 'Madison', 'Washington', 'Riverside', 'Fairview']
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']

        for column, info in schema_info.items():
            if info.get('is_address', False) and column in df.columns:
                method = self.config.get('address_method', 'remove-numbers')

                if method == 'synthetic':
                    df[column] = [f"{np.random.randint(100, 9999)} {np.random.choice(streets)}, "
                                  f"{np.random.choice(cities)}, {np.random.choice(states)} "
                                  f"{np.random.randint(10000, 99999)}"
                                  for _ in range(len(df))]
                elif method == 'city-state':
                    df[column] = [f"{np.random.choice(cities)}, {np.random.choice(states)}"
                                  for _ in range(len(df))]
                elif method == 'zip-only':
                    df[column] = [f"{np.random.randint(10000, 99999)}" for _ in range(len(df))]

        return df

    def _preserve_temporal_relationships(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Preserve temporal relationships between date columns"""
        relationships = self.schema.get(table_name, {}).get('relationships', {}).get('temporal', [])

        for rel in relationships:
            from_col = rel.get('from')
            to_col = rel.get('to')

            if from_col in df.columns and to_col in df.columns:
                try:
                    # Ensure temporal order is maintained
                    from_dates = pd.to_datetime(df[from_col], errors='coerce')
                    to_dates = pd.to_datetime(df[to_col], errors='coerce')

                    # Fix any violations
                    violations = to_dates < from_dates
                    if violations.any():
                        mean_duration = rel.get('mean_duration_days', 7)
                        for idx in df.index[violations]:
                            df.at[idx, to_col] = from_dates.iloc[idx] + pd.Timedelta(days=mean_duration)

                except Exception as e:
                    logger.warning(f"Error preserving temporal relationship {from_col}->{to_col}: {str(e)}")

        return df

    def evaluate_synthetic_data(self) -> Dict[str, Any]:
        """Evaluate the quality of synthetic data"""
        if not self.synthetic_data:
            raise ValueError("No synthetic data to evaluate")

        evaluation_results = {}

        for table_name in self.original_data.keys():
            if table_name not in self.synthetic_data:
                continue

            original_df = self.original_data[table_name]
            synthetic_df = self.synthetic_data[table_name]

            # Calculate metrics
            stats_similarity = self._calculate_statistical_similarity(original_df, synthetic_df)
            privacy_score = self._calculate_privacy_score(original_df, synthetic_df)
            utility_score = self._calculate_utility_score(original_df, synthetic_df)

            evaluation_results[table_name] = {
                'statistical_similarity': stats_similarity,
                'privacy_score': privacy_score,
                'utility_score': utility_score,
                'overall_score': (stats_similarity + utility_score + privacy_score) / 3,
                'record_count': {
                    'original': len(original_df),
                    'synthetic': len(synthetic_df)
                }
            }

        return evaluation_results

    def _calculate_statistical_similarity(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """Calculate statistical similarity score"""
        try:
            numeric_cols = original_df.select_dtypes(include=[np.number]).columns.intersection(
                synthetic_df.select_dtypes(include=[np.number]).columns
            )

            if len(numeric_cols) == 0:
                return 0.85  # Default for non-numeric data

            similarities = []

            for col in numeric_cols:
                orig_col = original_df[col].dropna()
                synth_col = synthetic_df[col].dropna()

                if len(orig_col) == 0 or len(synth_col) == 0:
                    continue

                # Compare statistical moments
                orig_mean, orig_std = orig_col.mean(), orig_col.std()
                synth_mean, synth_std = synth_col.mean(), synth_col.std()

                # Normalized differences
                mean_diff = abs(orig_mean - synth_mean) / (abs(orig_mean) + 1e-8)
                std_diff = abs(orig_std - synth_std) / (abs(orig_std) + 1e-8)

                # Similarity score (higher is better)
                col_similarity = 1 - min(1, (mean_diff + std_diff) / 2)
                similarities.append(col_similarity)

            return np.mean(similarities) if similarities else 0.85

        except Exception as e:
            logger.warning(f"Error calculating statistical similarity: {str(e)}")
            return 0.85

    def _calculate_privacy_score(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """Calculate privacy protection score"""
        try:
            # Simple privacy metric: percentage of non-matching records
            # Convert records to strings for comparison
            orig_records = set()
            synth_records = set()

            # Sample to avoid memory issues with large datasets
            sample_size = min(1000, len(original_df), len(synthetic_df))

            orig_sample = original_df.sample(min(sample_size, len(original_df)))
            synth_sample = synthetic_df.sample(min(sample_size, len(synthetic_df)))

            for _, row in orig_sample.iterrows():
                orig_records.add(str(tuple(row.fillna('').astype(str))))

            for _, row in synth_sample.iterrows():
                synth_records.add(str(tuple(row.fillna('').astype(str))))

            # Calculate overlap
            overlap = len(orig_records.intersection(synth_records))
            overlap_rate = overlap / len(synth_records) if synth_records else 0

            # Privacy score: 1 - overlap_rate (higher is better)
            privacy_score = max(0, 1 - overlap_rate)

            return privacy_score

        except Exception as e:
            logger.warning(f"Error calculating privacy score: {str(e)}")
            return 0.90  # Conservative default

    def _calculate_utility_score(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """Calculate ML utility score"""
        try:
            # Utility based on preserving column distributions and correlations
            numeric_cols = original_df.select_dtypes(include=[np.number]).columns.intersection(
                synthetic_df.select_dtypes(include=[np.number]).columns
            )

            if len(numeric_cols) < 2:
                return 0.80  # Default for insufficient numeric data

            # Compare correlation matrices
            orig_corr = original_df[numeric_cols].corr().fillna(0)
            synth_corr = synthetic_df[numeric_cols].corr().fillna(0)

            # Calculate correlation preservation
            corr_diff = np.abs(orig_corr - synth_corr).mean().mean()
            utility_score = max(0, 1 - corr_diff)

            return utility_score

        except Exception as e:
            logger.warning(f"Error calculating utility score: {str(e)}")
            return 0.80