import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import json
import time
import logging
import traceback
import copy
from datetime import datetime
from preprocessing.cleaning import DataCleaningPipeline
from stats.relationships import RelationshipDiscoverer
from models.generation_engine import SyntheticGenerationEngine
from evaluation.statistical_evaluator import SyntheticDataEvaluator
from export.validator import DataValidator
from export.adapters import ExportAdapter
from models.address_synthesis import AddressSynthesizer

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    import usaddress
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False

class SyntheticDataPipeline:
    """
    Main pipeline for synthetic data generation system
    """

    def __init__(self, config_path=None):
        # Initialize components
        self.schema = {}
        self.connectors = {}
        self.transformer = None
        self.analyzer = None
        self.generator = None
        self.evaluator = None
        self.validator = None
        self.exporter = None

        # Pipeline state
        self.original_data = {}
        self.synthetic_data = {}
        self.evaluation_results = None

        # Load configuration if provided
        if config_path:
            self.load_config(config_path)

        # Set up logging
        self._setup_logging()

    def load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load schema
            if 'schema' in config:
                self.schema = config['schema']

            # Configure components based on config
            # This is a simplified version - real implementation would be more complex

            return True

        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return False

    def _setup_logging(self):
        """Set up logging configuration"""
        if not logging.getLogger().hasHandlers():
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler('synthetic_data.log')
                ]
            )

    def connect_database(self, connection_string: str, tables: List[str] = None, chunk_size: int = 10000) -> bool:
        """Connect to database and load data with proper resource management"""
        try:
            from sqlalchemy import create_engine, text
            from urllib.parse import urlparse
            
            # Parse connection string to hide sensitive info in logs
            parsed_url = urlparse(connection_string)
            safe_connection_info = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}/{parsed_url.path.lstrip('/')}"
            logging.info(f"Connecting to database: {safe_connection_info}")

            # Create database connector with connection pooling
            engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            # Use context manager for connection
            with engine.connect() as connection:
                # Get table list if not provided
                if not tables:
                    from sqlalchemy import inspect
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                    
                    # Limit number of tables to prevent memory issues
                    if len(tables) > 50:
                        logging.warning(f"Large number of tables ({len(tables)}). Consider specifying specific tables.")
                        tables = tables[:50]

                # Load each table with chunking for large datasets
                for table in tables:
                    logging.info(f"Loading table: {table}")
                    
                    # First, check table size
                    count_query = text(f"SELECT COUNT(*) FROM {table}")
                    row_count = connection.execute(count_query).scalar()
                    
                    if row_count > 100000:
                        logging.warning(f"Table {table} has {row_count} rows. Loading in chunks.")
                        # Load in chunks for large tables
                        chunks = []
                        for chunk in pd.read_sql_table(table, connection, chunksize=chunk_size):
                            chunks.append(chunk)
                            if len(chunks) * chunk_size > 1000000:  # Limit to 1M rows max
                                logging.warning(f"Table {table} truncated to 1M rows for memory safety")
                                break
                        self.original_data[table] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                    else:
                        # Load entire table for smaller datasets
                        self.original_data[table] = pd.read_sql_table(table, connection)

            # Infer schema if not loaded
            if not self.schema:
                self._infer_schema()

            return True

        except Exception as e:
            logging.error(f"Error connecting to database: {e}")
            return False

    def load_csv_directory(self, directory_path: str) -> bool:
        """Load data from CSV files in directory"""
        try:
            logging.info(f"Loading CSV files from: {directory_path}")

            # Handle case of a single file directly
            if os.path.isfile(directory_path):
                file_name = os.path.basename(directory_path)
                table_name = os.path.splitext(file_name)[0]

                logging.info(f"Loading single CSV file: {file_name}")
                try:
                    self.original_data[table_name] = pd.read_csv(directory_path)

                    # Infer schema if not loaded
                    if not self.schema:
                        self._infer_schema()

                    return True
                except Exception as e:
                    logging.error(f"Error loading {file_name}: {e}")
                    return False

            # Get CSV files in directory
            csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

            # Load each CSV file
            for csv_file in csv_files:
                table_name = os.path.splitext(csv_file)[0]
                file_path = os.path.join(directory_path, csv_file)

                logging.info(f"Loading CSV file: {csv_file}")
                self.original_data[table_name] = pd.read_csv(file_path)

            # Infer schema if not loaded
            if not self.schema:
                self._infer_schema()

            return True

        except Exception as e:
            logging.error(f"Error loading CSV files: {e}")
            return False

    def _infer_schema(self):
        """Infer schema from loaded data"""
        logging.info("Inferring schema from data")

        schema = {}

        for table_name, df in self.original_data.items():
            table_schema = {'columns': {}}

            for column in df.columns:
                column_info = {}

                # Check if it's a name column
                if 'name' in column.lower() and not any(
                        substr in column.lower() for substr in ['filename', 'pathname']):
                    column_info['type'] = 'categorical'
                    column_info['is_name'] = True
                elif pd.api.types.is_numeric_dtype(df[column]):
                    column_info['type'] = 'numeric'
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    column_info['type'] = 'datetime'
                else:
                    column_info['type'] = 'categorical'

                # Check if potentially an ID column
                if column.lower().endswith('id') and df[column].nunique() == len(df):
                    column_info['is_id'] = True

                table_schema['columns'][column] = column_info

            schema[table_name] = table_schema

        self.schema = schema

    def preprocess_data(self):
        """Preprocess the loaded data"""
        logging.info("Preprocessing data")

        # Initialize transformer if not already done
        if self.transformer is None:
            self.transformer = DataCleaningPipeline(self.schema)

        # Preprocess each table
        for table_name, df in self.original_data.items():
            logging.info(f"Preprocessing table: {table_name}")

            # Identify name columns in schema if not already identified
            for column in df.columns:
                if 'name' in column.lower() and not any(substr in column.lower()
                                                        for substr in ['filename', 'pathname']):
                    if table_name in self.schema and column in self.schema[table_name]['columns']:
                        self.schema[table_name]['columns'][column]['is_name'] = True

            # Check if DataCleaningPipeline needs to be fit first
            if hasattr(self.transformer, 'fit'):
                self.transformer.fit(df)

            # Now transform the data
            self.original_data[table_name] = self.transformer.transform(df)

    def post_process_names(self, df, table_name, n_samples):
        """Replace name columns with realistic synthetic names"""
        import random

        # Check if this function should run at all
        if not hasattr(self, 'apply_name_abstraction') or not self.apply_name_abstraction:
            logging.info("Name abstraction disabled, returning original dataframe")
            return df

        if table_name not in self.schema:
            return df

        # Check for gender column to inform name generation
        gender_column = None
        gender_values = None

        for column in df.columns:
            if column.lower() in ['gender', 'sex']:
                gender_column = column
                gender_values = df[column].apply(lambda x: 'M' if str(x).lower() in ['m', 'male']
                else 'F' if str(x).lower() in ['f', 'female'] else None)
                break

        # Define name lists
        male_first_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
            "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua"
        ]

        female_first_names = [
            "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
            "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle"
        ]

        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
        ]

        # Process name columns
        if 'columns' in self.schema.get(table_name, {}):
            for column, info in self.schema[table_name]['columns'].items():
                if info.get('is_name') and column in df.columns:
                    logging.info(f"Generating synthetic names for column: {column}")

                    if gender_column:
                        # Generate names based on gender
                        names = []
                        for gender in gender_values:
                            if gender == 'M':
                                first_name = random.choice(male_first_names)
                            elif gender == 'F':
                                first_name = random.choice(female_first_names)
                            else:
                                first_name = random.choice(
                                    male_first_names if random.random() < 0.5 else female_first_names)

                            last_name = random.choice(last_names)
                            names.append(f"{first_name} {last_name}")
                        df[column] = names
                    else:
                        # Generate random names without gender info
                        df[column] = [
                            f"{random.choice(male_first_names if random.random() < 0.5 else female_first_names)} {random.choice(last_names)}"
                            for _ in range(len(df))]

        return df

    def identify_name_columns(self):
        """Interactive identification of name columns"""
        logging.info("Identifying name columns")

        for table_name, df in self.original_data.items():
            # Automatically detect potential name columns
            potential_name_columns = []
            for column in df.columns:
                if ('name' in column.lower() and
                        not any(substr in column.lower() for substr in ['filename', 'pathname'])):
                    potential_name_columns.append(column)

            if not potential_name_columns:
                continue

            # Ask user to confirm which columns are actually name columns
            print(f"\nFor table '{table_name}', the following columns might contain names:")
            for i, column in enumerate(potential_name_columns):
                # Show a sample of values from this column
                sample_values = df[column].dropna().sample(min(3, len(df))).tolist()
                print(f"{i + 1}. {column} (example values: {', '.join(str(v) for v in sample_values)})")

            print("\nWhich columns should be treated as person names for synthetic data generation?")
            print("Enter the numbers separated by commas, or 'all' to select all, or 'none' to select none:")

            user_input = input("> ").strip().lower()

            selected_columns = []
            if user_input == 'all':
                selected_columns = potential_name_columns
            elif user_input != 'none':
                try:
                    # Parse user input like "1,3,4"
                    selected_indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
                    selected_columns = [potential_name_columns[idx] for idx in selected_indices
                                        if 0 <= idx < len(potential_name_columns)]
                except:
                    print("Invalid input. No columns selected.")

            # Update schema with user selections
            for column in selected_columns:
                if table_name in self.schema and 'columns' in self.schema[table_name]:
                    if column in self.schema[table_name]['columns']:
                        self.schema[table_name]['columns'][column]['is_name'] = True
                        print(f"Marked '{column}' as a name column.")

    def identify_abstraction_columns(self):
        """Interactive identification of columns to be abstracted with dummy data"""
        logging.info("Identifying columns for abstraction/dummy data")

        for table_name, df in self.original_data.items():
            print(f"\n=== Table: {table_name} ===")
            print("Which columns would you like to fill with dummy/synthetic data?")

            # Show all columns with examples of their current data
            columns = list(df.columns)
            for i, column in enumerate(columns):
                # Get column type from schema
                col_type = "unknown"
                if (table_name in self.schema and
                        'columns' in self.schema[table_name] and
                        column in self.schema[table_name]['columns']):
                    col_type = self.schema[table_name]['columns'][column].get('type', 'unknown')

                # Get sample values
                sample_values = df[column].dropna().sample(min(3, len(df))).tolist()
                print(f"{i + 1}. {column} ({col_type}) - Examples: {', '.join(str(v) for v in sample_values)}")

            print("\nEnter the numbers of columns to abstract, separated by commas:")
            print("Or enter 'all' for all columns, 'none' for no columns, or 'sensitive' for likely sensitive columns")

            user_input = input("> ").strip().lower()

            selected_columns = []
            if user_input == 'all':
                selected_columns = columns
            elif user_input == 'sensitive':
                # Select likely sensitive columns based on name
                sensitive_keywords = ['name', 'email', 'phone', 'address', 'ssn', 'birth', 'age',
                                      'gender', 'income', 'salary', 'credit', 'password', 'account']
                selected_columns = [col for col in columns if
                                    any(keyword in col.lower() for keyword in sensitive_keywords)]
                print(f"Selected {len(selected_columns)} likely sensitive columns")
            elif user_input != 'none':
                try:
                    # Parse user input like "1,3,4"
                    selected_indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
                    selected_columns = [columns[idx] for idx in selected_indices if 0 <= idx < len(columns)]
                except:
                    print("Invalid input. No columns selected.")

            # For each selected column, ask user what type of abstraction they want
            for column in selected_columns:
                col_type = "unknown"
                if (table_name in self.schema and
                        'columns' in self.schema[table_name] and
                        column in self.schema[table_name]['columns']):
                    col_type = self.schema[table_name]['columns'][column].get('type', 'unknown')

                print(f"\nFor column '{column}' ({col_type}), what type of abstraction do you want?")

                abstraction_options = []

                # Suggest appropriate abstraction methods based on column type
                if col_type == 'numeric':
                    abstraction_options = ['random_numeric', 'scaled_numeric', 'random_in_range']
                elif 'name' in column.lower():
                    abstraction_options = ['random_names', 'initials_only']
                elif 'email' in column.lower():
                    abstraction_options = ['random_emails', 'domain_preserve']
                elif 'address' in column.lower():
                    abstraction_options = ['random_addresses', 'city_only']
                elif 'phone' in column.lower():
                    abstraction_options = ['random_phones', 'format_preserve']
                elif 'date' in column.lower() or col_type == 'datetime':
                    abstraction_options = ['random_dates', 'month_year_only']
                else:
                    abstraction_options = ['random_categorical', 'random_text', 'format_preserve']

                # Show options
                for i, option in enumerate(abstraction_options):
                    print(f"{i + 1}. {option}")

                print(f"{len(abstraction_options) + 1}. Skip this column")

                # Get user choice
                try:
                    choice = int(input("Enter number: "))
                    if 1 <= choice <= len(abstraction_options):
                        # Mark column for abstraction in schema
                        if table_name not in self.schema:
                            self.schema[table_name] = {'columns': {}}
                        if 'columns' not in self.schema[table_name]:
                            self.schema[table_name]['columns'] = {}
                        if column not in self.schema[table_name]['columns']:
                            self.schema[table_name]['columns'][column] = {}

                        self.schema[table_name]['columns'][column]['abstract'] = True
                        self.schema[table_name]['columns'][column]['abstract_method'] = abstraction_options[choice - 1]
                        print(f"Column '{column}' will be abstracted using {abstraction_options[choice - 1]}")
                except:
                    print(f"Invalid choice. Skipping column '{column}'.")

    def apply_abstractions(self, df, table_name):
        """Apply abstractions to columns marked for abstraction in the schema"""
        import random
        import string
        from datetime import datetime, timedelta

        if table_name not in self.schema or 'columns' not in self.schema[table_name]:
            return df

        result_df = df.copy()

        for column, info in self.schema[table_name]['columns'].items():
            if column not in df.columns:
                continue

            if info.get('abstract'):
                method = info.get('abstract_method', 'random_categorical')
                logging.info(f"Applying abstraction '{method}' to column '{column}'")

                n_samples = len(df)

                # Apply the appropriate abstraction method
                if method == 'random_numeric':
                    # Generate random numbers
                    min_val = df[column].min() if pd.api.types.is_numeric_dtype(df[column]) else 0
                    max_val = df[column].max() if pd.api.types.is_numeric_dtype(df[column]) else 100
                    result_df[column] = np.random.uniform(min_val, max_val, n_samples)

                elif method == 'scaled_numeric':
                    # Generate numbers with same distribution but different values
                    if pd.api.types.is_numeric_dtype(df[column]):
                        mean = df[column].mean()
                        std = df[column].std()
                        result_df[column] = np.random.normal(mean, std, n_samples)
                    else:
                        result_df[column] = np.random.uniform(0, 100, n_samples)

                elif method == 'random_in_range':
                    # Random values within min/max range
                    if pd.api.types.is_numeric_dtype(df[column]):
                        min_val = df[column].min()
                        max_val = df[column].max()
                        result_df[column] = np.random.uniform(min_val, max_val, n_samples)
                    else:
                        result_df[column] = np.random.uniform(0, 100, n_samples)

                elif method == 'random_names':
                    # Generate random names
                    result_df[column] = self.generate_random_names(n_samples)

                elif method == 'initials_only':
                    # Replace names with initials
                    result_df[column] = df[column].apply(
                        lambda x: ''.join([word[0].upper() + '.' for word in str(x).split()])
                    )

                elif method == 'random_emails':
                    # Random email addresses
                    domains = ['example.com', 'sample.org', 'test.net', 'dummy.co']
                    result_df[column] = [
                        f"user{i}@{random.choice(domains)}" for i in range(n_samples)
                    ]

                elif method == 'domain_preserve':
                    # Preserve domain but randomize username
                    def abstract_email(email):
                        if '@' not in str(email):
                            return f"user{random.randint(1000, 9999)}@example.com"
                        username, domain = str(email).split('@', 1)
                        return f"user{random.randint(1000, 9999)}@{domain}"

                    result_df[column] = df[column].apply(abstract_email)

                elif method == 'random_addresses':
                    # Random addresses
                    streets = ['Main St', 'Oak Ave', 'Maple Rd', 'Park Blvd', 'Cedar Ln']
                    cities = ['Springfield', 'Rivertown', 'Lakeside', 'Hilltop', 'Westview']
                    states = ['CA', 'NY', 'TX', 'FL', 'IL']

                    result_df[column] = [
                        f"{random.randint(100, 999)} {random.choice(streets)}, "
                        f"{random.choice(cities)}, {random.choice(states)} "
                        f"{random.randint(10000, 99999)}"
                        for _ in range(n_samples)
                    ]

                elif method == 'city_only':
                    # Keep only city part or generate random city
                    cities = ['Springfield', 'Rivertown', 'Lakeside', 'Hilltop', 'Westview']
                    result_df[column] = [random.choice(cities) for _ in range(n_samples)]

                elif method == 'random_phones':
                    # Random phone numbers
                    result_df[column] = [
                        f"({random.randint(100, 999)})-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                        for _ in range(n_samples)
                    ]

                elif method == 'format_preserve':
                    # Try to preserve format but change the actual values
                    def randomize_keeping_format(value):
                        value = str(value)
                        result = ""
                        for char in value:
                            if char.isdigit():
                                result += str(random.randint(0, 9))
                            elif char.isalpha():
                                if char.isupper():
                                    result += random.choice(string.ascii_uppercase)
                                else:
                                    result += random.choice(string.ascii_lowercase)
                            else:
                                result += char
                        return result

                    result_df[column] = df[column].apply(randomize_keeping_format)

                elif method == 'random_dates':
                    # Random dates
                    start_date = datetime(2000, 1, 1)
                    end_date = datetime(2023, 12, 31)
                    days_range = (end_date - start_date).days

                    result_df[column] = [
                        start_date + timedelta(days=random.randint(0, days_range))
                        for _ in range(n_samples)
                    ]

                elif method == 'month_year_only':
                    # Only keep month and year
                    def get_month_year(date_val):
                        try:
                            date = pd.to_datetime(date_val)
                            return f"{date.month}/{date.year}"
                        except:
                            return "01/2000"

                    result_df[column] = df[column].apply(get_month_year)

                elif method == 'random_categorical':
                    # Random values from existing set
                    unique_values = df[column].dropna().unique()
                    if len(unique_values) > 0:
                        result_df[column] = [random.choice(unique_values) for _ in range(n_samples)]
                    else:
                        result_df[column] = [f"Category{i % 5}" for i in range(n_samples)]

                elif method == 'random_text':
                    # Generate random text strings
                    def random_text(length=10):
                        return ''.join(random.choice(string.ascii_letters) for _ in range(length))

                    result_df[column] = [random_text(random.randint(5, 15)) for _ in range(n_samples)]

        return result_df

    def detect_temporal_relationships(self):
        """Detect and store temporal relationships between date columns"""
        logging.info("Detecting temporal relationships between columns")

        # Skip if no data or schema
        if not self.original_data or not self.schema:
            logging.warning("No data or schema available for relationship detection")
            return

        try:
            for table_name, df in self.original_data.items():
                if table_name not in self.schema:
                    continue

                logging.info(f"Detecting temporal relationships for table: {table_name}")

                # Find date columns
                date_columns = []
                temp_df = df.copy()  # Work with a copy

                for column in df.columns:
                    # Try to detect date columns by name or content
                    if 'date' in column.lower() or 'time' in column.lower():
                        try:
                            # Try to convert to datetime and store in temporary dataframe
                            temp_df[column] = pd.to_datetime(df[column], errors='coerce')

                            # Only include if conversion worked for some values
                            if not temp_df[column].isna().all():
                                date_columns.append(column)
                                # Also update type in schema
                                if (table_name in self.schema and
                                        'columns' in self.schema[table_name] and
                                        column in self.schema[table_name]['columns']):
                                    self.schema[table_name]['columns'][column]['type'] = 'datetime'
                        except:
                            pass

                logging.info(f"Found {len(date_columns)} date columns: {date_columns}")

                if len(date_columns) < 2:
                    continue

                # Initialize relationships structure if needed
                if 'relationships' not in self.schema[table_name]:
                    self.schema[table_name]['relationships'] = {}

                if 'temporal' not in self.schema[table_name]['relationships']:
                    self.schema[table_name]['relationships']['temporal'] = []

                # Check for sequential date pairs (like admission -> discharge)
                for i, date_col1 in enumerate(date_columns):
                    for date_col2 in date_columns[i + 1:]:
                        # Skip if dates are unrelated based on name
                        if not any(x in date_col1.lower() or x in date_col2.lower()
                                   for x in ['admit', 'discharge', 'start', 'end', 'in', 'out']):
                            continue

                        logging.info(f"Analyzing potential relationship: {date_col1} -> {date_col2}")

                        # Convert columns to datetime for comparison
                        try:
                            date1 = pd.to_datetime(df[date_col1], errors='coerce')
                            date2 = pd.to_datetime(df[date_col2], errors='coerce')

                            # Drop rows where either date is missing
                            valid_mask = ~(date1.isna() | date2.isna())
                            valid_count = valid_mask.sum()

                            if valid_count < 10:  # Need enough samples
                                logging.info(f"Not enough valid date pairs for {date_col1} -> {date_col2}")
                                continue

                            # Count how many times col1 is before col2
                            count_col1_before_col2 = (date1[valid_mask] <= date2[valid_mask]).mean()

                            # If one direction is dominant (>75%), record it as a relationship
                            if count_col1_before_col2 > 0.75:
                                logging.info(f"Detected temporal relationship: {date_col1} -> {date_col2}")

                                # Calculate typical duration between dates in days
                                durations = (date2[valid_mask] - date1[valid_mask]).dt.total_seconds() / (24 * 3600)

                                # Handle potential outliers by capping at reasonable values
                                durations = durations.clip(lower=0, upper=365)  # Cap at 1 year

                                # Calculate quartiles safely
                                try:
                                    quartile_25 = durations.quantile(0.25)
                                    quartile_50 = durations.quantile(0.5)
                                    quartile_75 = durations.quantile(0.75)
                                    quartiles = [quartile_25, quartile_50, quartile_75]
                                except:
                                    # Fallback to simple statistics if quantiles fail
                                    quartiles = [1, 3, 7]  # Default values

                                self.schema[table_name]['relationships']['temporal'].append({
                                    'from': date_col1,
                                    'to': date_col2,
                                    'type': 'sequential',
                                    'min_days': max(0, durations.min()),
                                    'median_days': durations.median(),
                                    'mean_days': durations.mean(),
                                    'max_days': min(365, durations.max()),  # Cap at 1 year
                                    'quartiles': quartiles
                                })
                            elif count_col1_before_col2 < 0.25:
                                logging.info(f"Detected temporal relationship: {date_col2} -> {date_col1}")

                                # Calculate durations in the opposite direction
                                durations = (date1[valid_mask] - date2[valid_mask]).dt.total_seconds() / (24 * 3600)
                                durations = durations.clip(lower=0, upper=365)

                                try:
                                    quartiles = [
                                        durations.quantile(0.25),
                                        durations.quantile(0.5),
                                        durations.quantile(0.75)
                                    ]
                                except:
                                    quartiles = [1, 3, 7]

                                self.schema[table_name]['relationships']['temporal'].append({
                                    'from': date_col2,
                                    'to': date_col1,
                                    'type': 'sequential',
                                    'min_days': max(0, durations.min()),
                                    'median_days': durations.median(),
                                    'mean_days': durations.mean(),
                                    'max_days': min(365, durations.max()),
                                    'quartiles': quartiles
                                })
                        except Exception as e:
                            logging.warning(f"Error analyzing relationship {date_col1} -> {date_col2}: {str(e)}")
        except Exception as e:
            logging.error(f"Error in detect_temporal_relationships: {str(e)}")
            logging.error(traceback.format_exc())

    def detect_conditional_dependencies(self):
        """Detect dependencies between categorical values and numeric metrics"""
        logging.info("Detecting conditional dependencies between columns")

        # Skip if no data or schema
        if not self.original_data or not self.schema:
            logging.warning("No data or schema available for dependency detection")
            return

        try:
            for table_name, df in self.original_data.items():
                if table_name not in self.schema:
                    continue

                logging.info(f"Detecting conditional dependencies for table: {table_name}")

                # Find date columns for duration calculations
                date_columns = []
                for column, info in self.schema[table_name].get('columns', {}).items():
                    if column not in df.columns:
                        continue

                    if info.get('type') == 'datetime' or 'date' in column.lower():
                        try:
                            # Test conversion without modifying original data
                            test_series = pd.to_datetime(df[column], errors='coerce')
                            if not test_series.isna().all():
                                date_columns.append(column)
                        except:
                            pass

                logging.info(f"Found {len(date_columns)} date columns for conditional dependencies")

                # Initialize relationships structure
                if 'relationships' not in self.schema[table_name]:
                    self.schema[table_name]['relationships'] = {}

                if 'conditional' not in self.schema[table_name]['relationships']:
                    self.schema[table_name]['relationships']['conditional'] = []

                # Calculate stay durations if we have temporal relationships
                durations = {}
                temporal_rels = self.schema[table_name].get('relationships', {}).get('temporal', [])

                for rel in temporal_rels:
                    from_col = rel.get('from')
                    to_col = rel.get('to')

                    if not from_col or not to_col:
                        continue

                    if from_col not in df.columns or to_col not in df.columns:
                        continue

                    if 'admit' in from_col.lower() or 'start' in from_col.lower():
                        if 'discharge' in to_col.lower() or 'end' in to_col.lower():
                            try:
                                # Calculate stay duration without modifying original data
                                date1 = pd.to_datetime(df[from_col], errors='coerce')
                                date2 = pd.to_datetime(df[to_col], errors='coerce')

                                valid_mask = ~(date1.isna() | date2.isna())
                                if valid_mask.sum() < 10:
                                    continue

                                # Calculate duration in days
                                duration_series = (date2 - date1).dt.total_seconds() / (24 * 3600)

                                # Only keep valid (positive) durations
                                valid_durations = duration_series[valid_mask & (duration_series > 0)]

                                if len(valid_durations) >= 10:
                                    duration_name = f"duration_{from_col}_{to_col}"
                                    durations[duration_name] = valid_durations
                                    logging.info(
                                        f"Calculated {len(valid_durations)} valid durations for {duration_name}")
                            except Exception as e:
                                logging.warning(f"Error calculating durations: {str(e)}")

                # Find categorical columns that might affect outcomes
                cat_columns = []
                for column in df.columns:
                    if column not in df.columns:
                        continue

                    # Check if it's a likely categorical column
                    if df[column].nunique() <= 10 and df[column].nunique() > 1:
                        if any(term in column.lower() for term in ['test', 'result', 'status', 'type', 'category']):
                            cat_columns.append(column)

                logging.info(f"Found {len(cat_columns)} potential categorical columns: {cat_columns}")

                # Check if categorical values are related to durations
                for cat_col in cat_columns:
                    for duration_name, duration_series in durations.items():
                        try:
                            # Get categories from the column
                            cat_values = df[cat_col].dropna().unique()

                            if len(cat_values) <= 1:
                                continue

                            logging.info(f"Analyzing if {cat_col} affects {duration_name}")

                            # Calculate median duration for each category
                            category_durations = {}
                            valid_categories = 0

                            for cat_val in cat_values:
                                # Find rows with this category that also have duration values
                                cat_mask = df[cat_col] == cat_val
                                cat_indices = duration_series.index[cat_mask]

                                if len(cat_indices) >= 5:  # Need enough samples
                                    cat_durations = duration_series.loc[cat_indices]

                                    category_durations[str(cat_val)] = {
                                        'mean': float(cat_durations.mean()),
                                        'median': float(cat_durations.median()),
                                        'min': float(cat_durations.min()),
                                        'max': float(cat_durations.max()),
                                        'count': int(len(cat_durations))
                                    }
                                    valid_categories += 1

                            if valid_categories <= 1:
                                continue

                            # Check if there's a significant difference between categories
                            medians = [info['median'] for info in category_durations.values()]
                            if not medians:
                                continue

                            min_median = min(medians)
                            max_median = max(medians)

                            # If max is at least 20% higher than min, consider it a relationship
                            if max_median > min_median * 1.2:
                                logging.info(f"Detected conditional dependency: {cat_col} affects {duration_name}")

                                # Extract date columns from duration name
                                parts = duration_name.split('_')
                                if len(parts) < 3:
                                    continue

                                date_col1 = parts[1]
                                date_col2 = parts[2]

                                self.schema[table_name]['relationships']['conditional'].append({
                                    'type': 'categorical_affects_duration',
                                    'categorical_column': cat_col,
                                    'from_date': date_col1,
                                    'to_date': date_col2,
                                    'category_durations': category_durations
                                })
                        except Exception as e:
                            logging.warning(f"Error analyzing {cat_col} -> {duration_name}: {str(e)}")
        except Exception as e:
            logging.error(f"Error in detect_conditional_dependencies: {str(e)}")
            logging.error(traceback.format_exc())

    def identify_columns_to_synthesize(self):
        """Interactive identification of columns to be synthesized"""
        logging.info("Identifying columns for synthesis")

        for table_name, df in self.original_data.items():

            # At the end, ask specifically about name abstraction
            print("\nDo you want to apply name abstraction to hide personal names? (yes/no)")
            name_abstraction_input = input("> ").strip().lower()
            self.apply_name_abstraction = name_abstraction_input.startswith('y')

            print(f"\n=== Table: {table_name} ===")
            print("Which columns would you like to synthesize (generate synthetic data for)?")
            if self.apply_name_abstraction:
                print("Name abstraction will be applied")
            else:
                print("Name abstraction will NOT be applied")

            # Show all columns with examples of their current data
            columns = list(df.columns)
            for i, column in enumerate(columns):
                # Get column type from schema
                col_type = "unknown"
                if (table_name in self.schema and
                        'columns' in self.schema[table_name] and
                        column in self.schema[table_name]['columns']):
                    col_type = self.schema[table_name]['columns'][column].get('type', 'unknown')

                # Get sample values
                sample_values = df[column].dropna().sample(min(3, len(df))).tolist()
                print(f"{i + 1}. {column} ({col_type}) - Examples: {', '.join(str(v) for v in sample_values)}")

            print("\nEnter the numbers of columns to synthesize, separated by commas:")
            print("Or enter 'all' for all columns, 'none' for no columns, or 'sensitive' for likely sensitive columns")

            user_input = input("> ").strip().lower()

            selected_columns = []
            if user_input == 'all':
                selected_columns = columns
            elif user_input == 'sensitive':
                # Select likely sensitive columns based on name
                sensitive_keywords = ['name', 'email', 'phone', 'address', 'ssn', 'birth', 'age',
                                      'gender', 'income', 'salary', 'credit', 'password', 'account']
                selected_columns = [col for col in columns if
                                    any(keyword in col.lower() for keyword in sensitive_keywords)]
                print(f"Selected {len(selected_columns)} likely sensitive columns")
            elif user_input != 'none':
                try:
                    # Parse user input like "1,3,4"
                    selected_indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
                    selected_columns = [columns[idx] for idx in selected_indices if 0 <= idx < len(columns)]
                except:
                    print("Invalid input. No columns selected.")



            # Mark the selected columns for synthesis in the schema
            for column in columns:
                if table_name not in self.schema:
                    self.schema[table_name] = {'columns': {}}
                if 'columns' not in self.schema[table_name]:
                    self.schema[table_name]['columns'] = {}
                if column not in self.schema[table_name]['columns']:
                    self.schema[table_name]['columns'][column] = {}

                # Set synthesize flag based on selection
                self.schema[table_name]['columns'][column]['synthesize'] = column in selected_columns

                if column in selected_columns:
                    print(f"Column '{column}' will be synthesized")
                else:
                    print(f"Column '{column}' will be copied from original data")

    def post_process_synthetic_data(self, df, table_name):
        """Apply detected relationships and constraints to synthetic data"""
        if table_name not in self.schema or df.empty:
            logging.warning(f"Skipping post-processing for {table_name} - empty DataFrame or missing schema")
            return df

        # Always make a copy to avoid modifying the original
        try:
            result_df = df.copy()
            original_rows = len(result_df)
            logging.info(f"Post-processing {original_rows} rows for table {table_name}")

            # Add debugging output
            logging.info(f"Original DataFrame columns: {list(result_df.columns)}")

            # Store original date formats for restoration later
            date_columns_original_format = {}

            # Check for temporal relationships
            temporal_rels = self.schema[table_name].get('relationships', {}).get('temporal', [])
            if temporal_rels:
                logging.info(f"Found {len(temporal_rels)} temporal relationships to enforce")
                for rel in temporal_rels:
                    from_col = rel.get('from')
                    to_col = rel.get('to')

                    if not from_col or not to_col:
                        logging.warning(f"Incomplete relationship specification: {rel}")
                        continue

                    if from_col not in result_df.columns or to_col not in result_df.columns:
                        logging.warning(f"Columns for relationship not found: {from_col} -> {to_col}")
                        continue

                    try:
                        logging.info(f"Enforcing temporal relationship: {from_col} -> {to_col}")

                        # Store original string formats before conversion
                        date_columns_original_format[from_col] = result_df[from_col].copy()
                        date_columns_original_format[to_col] = result_df[to_col].copy()

                        # Try to convert to datetime, but don't fail if not possible
                        try:
                            result_df[from_col] = pd.to_datetime(result_df[from_col], errors='coerce')
                            result_df[to_col] = pd.to_datetime(result_df[to_col], errors='coerce')
                        except Exception as e:
                            logging.warning(f"Could not convert columns to datetime: {str(e)}")
                            continue

                        # Only process rows where both dates are valid
                        valid_rows = result_df.dropna(subset=[from_col, to_col])
                        logging.info(f"Processing {len(valid_rows)} valid date pairs")

                        # Fix inconsistencies only for valid rows
                        inconsistent_mask = valid_rows[to_col] < valid_rows[from_col]
                        inconsistent_count = inconsistent_mask.sum()

                        if inconsistent_count > 0:
                            logging.info(f"Fixing {inconsistent_count} inconsistent date pairs")

                            # Get quartiles with fallback values
                            quartiles = rel.get('quartiles', [1, 3, 7])
                            min_days = rel.get('min_days', 0.5)
                            max_days = rel.get('max_days', 30)

                            # For each inconsistent row, generate a new end date
                            for idx in valid_rows.index[inconsistent_mask]:
                                random_val = np.random.random()

                                if random_val < 0.25:
                                    days = np.random.uniform(min_days, quartiles[0])
                                elif random_val < 0.5:
                                    days = np.random.uniform(quartiles[0], quartiles[1])
                                elif random_val < 0.75:
                                    days = np.random.uniform(quartiles[1], quartiles[2])
                                else:
                                    days = np.random.uniform(quartiles[2], max_days)

                                # Set the new end date
                                result_df.at[idx, to_col] = result_df.at[idx, from_col] + pd.Timedelta(days=days)

                        # IMPORTANT FIX: Convert datetime objects back to string format
                        # Try to determine the original date format from the original data
                        original_format = self._detect_date_format(date_columns_original_format[from_col])

                        if original_format:
                            logging.info(f"Converting {from_col} back to string format: {original_format}")
                            result_df[from_col] = result_df[from_col].dt.strftime(original_format)

                            logging.info(f"Converting {to_col} back to string format: {original_format}")
                            result_df[to_col] = result_df[to_col].dt.strftime(original_format)
                        else:
                            # Fallback to a standard format
                            logging.info(f"Using fallback date format for {from_col} and {to_col}")
                            result_df[from_col] = result_df[from_col].dt.strftime('%m/%d/%Y')
                            result_df[to_col] = result_df[to_col].dt.strftime('%m/%d/%Y')

                    except Exception as e:
                        logging.error(f"Error processing temporal relationship: {str(e)}")
                        # Restore original values if processing failed
                        if from_col in date_columns_original_format:
                            result_df[from_col] = date_columns_original_format[from_col]
                        if to_col in date_columns_original_format:
                            result_df[to_col] = date_columns_original_format[to_col]
                        # Continue with next relationship, don't disrupt the process

            # Check if we still have data
            if result_df.empty:
                logging.error("DataFrame became empty after temporal processing - returning original")
                return df

            # Simplified handling of conditional dependencies to avoid potential issues
            conditional_deps = self.schema[table_name].get('relationships', {}).get('conditional', [])
            if conditional_deps:
                logging.info(f"Found {len(conditional_deps)} conditional dependencies")

                for dep in conditional_deps:
                    if dep.get('type') != 'categorical_affects_duration':
                        continue

                    cat_col = dep.get('categorical_column')
                    from_date = dep.get('from_date')
                    to_date = dep.get('to_date')

                    if not all(col in result_df.columns for col in [cat_col, from_date, to_date]):
                        logging.warning(f"Columns for dependency not found: {cat_col}, {from_date}, {to_date}")
                        continue

                    try:
                        logging.info(f"Applying conditional dependency: {cat_col} affects {from_date} -> {to_date}")

                        # Store original formats
                        from_date_original = result_df[from_date].copy()
                        to_date_original = result_df[to_date].copy()

                        # Convert date columns safely
                        result_df[from_date] = pd.to_datetime(result_df[from_date], errors='coerce')
                        result_df[to_date] = pd.to_datetime(result_df[to_date], errors='coerce')

                        # Get valid rows
                        valid_mask = result_df[cat_col].notna() & result_df[from_date].notna()
                        valid_count = valid_mask.sum()

                        logging.info(f"Processing {valid_count} rows with valid category and start date")

                        # Get durations for categories
                        cat_durations = dep.get('category_durations', {})

                        # Only apply if we have duration info
                        if not cat_durations:
                            logging.warning("No category durations defined - skipping")
                            continue

                        # Process each category
                        for cat_val, duration_info in cat_durations.items():
                            cat_mask = (result_df[cat_col].astype(str) == cat_val) & valid_mask
                            cat_count = cat_mask.sum()

                            if cat_count == 0:
                                continue

                            logging.info(f"Setting durations for {cat_count} rows with category '{cat_val}'")

                            # Get duration parameters with fallbacks
                            median = duration_info.get('median', 5)
                            min_val = duration_info.get('min', max(0.5, median * 0.5))
                            max_val = duration_info.get('max', median * 2)

                            # Set end dates based on start dates
                            for idx in result_df.index[cat_mask]:
                                try:
                                    # Get start date
                                    start_date = result_df.at[idx, from_date]
                                    if pd.isna(start_date):
                                        continue

                                    # Generate duration - use simple uniform distribution for stability
                                    duration = np.random.uniform(min_val, max_val)

                                    # Set end date
                                    result_df.at[idx, to_date] = start_date + pd.Timedelta(days=duration)
                                except Exception as e:
                                    logging.warning(f"Error setting duration for row {idx}: {str(e)}")

                        # Convert back to string format
                        original_format = self._detect_date_format(from_date_original)
                        if original_format:
                            result_df[from_date] = result_df[from_date].dt.strftime(original_format)
                            result_df[to_date] = result_df[to_date].dt.strftime(original_format)
                        else:
                            result_df[from_date] = result_df[from_date].dt.strftime('%m/%d/%Y')
                            result_df[to_date] = result_df[to_date].dt.strftime('%m/%d/%Y')

                    except Exception as e:
                        logging.error(f"Error processing conditional dependency: {str(e)}")

            # Final checks
            final_rows = len(result_df)
            if final_rows < original_rows:
                logging.warning(f"Lost {original_rows - final_rows} rows during post-processing - investigation needed")
                if final_rows == 0:
                    logging.error("All data was lost - returning original data")
                    return df

            logging.info(f"Post-processing complete: {final_rows} rows remaining")
            return result_df

        except Exception as e:
            logging.error(f"Unexpected error in post-processing: {str(e)}")
            logging.error(traceback.format_exc())
            # Return the original data as a fallback
            return df

    def _detect_date_format(self, date_series):
        """Detect the date format from a series of date strings"""
        try:
            # Get a few non-null sample values
            sample_values = date_series.dropna().head(10).tolist()

            if not sample_values:
                return None

            # Common date formats to test
            formats_to_test = [
                '%m/%d/%Y',  # 12/31/2023
                '%d/%m/%Y',  # 31/12/2023
                '%Y-%m-%d',  # 2023-12-31
                '%m-%d-%Y',  # 12-31-2023
                '%d-%m-%Y',  # 31-12-2023
                '%Y/%m/%d',  # 2023/12/31
                '%m/%d/%y',  # 12/31/23
                '%d/%m/%y',  # 31/12/23
            ]

            # Test each format against the sample values
            for fmt in formats_to_test:
                try:
                    # Try to parse all sample values with this format
                    parsed_count = 0
                    for val in sample_values[:5]:  # Test first 5 values
                        try:
                            pd.to_datetime(str(val), format=fmt)
                            parsed_count += 1
                        except:
                            break

                    # If all sample values parsed successfully, use this format
                    if parsed_count == len(sample_values[:5]):
                        logging.info(f"Detected date format: {fmt}")
                        return fmt
                except:
                    continue

            # If no format worked, try to infer from the first value
            first_val = str(sample_values[0])
            if '/' in first_val:
                if len(first_val.split('/')[-1]) == 4:  # Year is 4 digits
                    return '%m/%d/%Y'
                else:
                    return '%m/%d/%y'
            elif '-' in first_val:
                if len(first_val.split('-')[0]) == 4:  # Year first
                    return '%Y-%m-%d'
                else:
                    return '%m-%d-%Y'

            logging.warning("Could not detect date format, using default")
            return '%m/%d/%Y'  # Default format

        except Exception as e:
            logging.error(f"Error detecting date format: {str(e)}")
            return '%m/%d/%Y'  # Default format



    def analyze_data(self):
        """Analyze the preprocessed data"""
        logging.info("Analyzing data")

        # Initialize analyzer if not already done
        if self.analyzer is None:
            self.analyzer = RelationshipDiscoverer(self.schema)

        # Analyze each table
        for table_name, df in self.original_data.items():
            logging.info(f"Analyzing table: {table_name}")
            self.analyzer.discover_relationships(df)

    # Automated Preprocessing Fix
    def clean_numeric_columns(self):
        """Clean all numeric columns across all tables"""
        print("\n===== CLEANING NUMERIC COLUMNS =====")
        for table_name, df in self.original_data.items():
            print(f"\nTable: {table_name}")
            for col in df.columns:
                if self.schema.get(col, {}).get('type', '') == 'numeric':
                    # Check for problematic values
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        print(f"Column '{col}': No issues found")
                    except ValueError:
                        # Count problematic values
                        mask = df[col].apply(lambda x: isinstance(x, str))
                        problem_count = mask.sum()
                        sample_values = df.loc[mask, col].head(3).tolist() if problem_count > 0 else []
                        print(f"Column '{col}': Found {problem_count} problematic values. Examples: {sample_values}")

                        # Convert and log results
                        before_count = df[col].isna().sum()
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        after_count = df[col].isna().sum()
                        print(f"  - Converted {after_count - before_count} values to NaN")

                        # Fill NaN values
                        mean_val = df[col]. mean()
                        df[col] = df[col].fillna(mean_val)
                        print(f"  - Filled NaN values with mean: {mean_val:.2f}")

            # Update the data
            self.original_data[table_name] = df
        print("\n===== NUMERIC CLEANING COMPLETE =====")

    def verify_data_is_clean(self):
        """Verify all numeric columns are actually numeric"""
        print("\n===== VERIFYING CLEAN DATA =====")
        all_clean = True

        for table_name, df in self.original_data.items():
            for col in df.columns:
                if self.schema.get(col, {}).get('type', '') == 'numeric':
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        print(f"✓ Table '{table_name}', Column '{col}': Clean")
                    except ValueError as e:
                        all_clean = False
                        print(f"✗ Table '{table_name}', Column '{col}': Still has issues - {e}")

        if all_clean:
            print("All data is clean! ✓")
        else:
            print("WARNING: Some data issues remain! ✗")
        print("===== VERIFICATION COMPLETE =====\n")

    def clean_problematic_rows(self):
        """Remove rows with invalid data in numeric columns"""
        for table_name, df in self.original_data.items():
            print(f"\nChecking table: {table_name}")

            # Create a mask to track rows that should be dropped
            rows_to_drop = pd.Series(False, index=df.index)
            drop_reasons = {}

            # Check each numeric column
            for col in df.columns:
                if self.schema.get(col, {}).get('type', '') == 'numeric':
                    # Identify rows with non-numeric values
                    numeric_mask = pd.to_numeric(df[col], errors='coerce').isna()
                    # Only consider actual strings, not NaN values that were already there
                    string_mask = df[col].apply(lambda x: isinstance(x, str))
                    problem_mask = numeric_mask & string_mask

                    if problem_mask.any():
                        # Get examples of problematic values
                        problematic_values = df.loc[problem_mask, col].unique()
                        value_examples = problematic_values[:3]

                        # Add to the drop mask
                        row_count_before = rows_to_drop.sum()
                        rows_to_drop = rows_to_drop | problem_mask
                        new_rows = rows_to_drop.sum() - row_count_before

                        if new_rows > 0:
                            drop_reasons[col] = {
                                'count': int(problem_mask.sum()),
                                'examples': value_examples.tolist()
                            }

            # Drop the problematic rows
            if rows_to_drop.any():
                original_count = len(df)
                df = df[~rows_to_drop]
                self.original_data[table_name] = df

                # Print summary
                print(
                    f"Dropped {rows_to_drop.sum()} rows ({rows_to_drop.mean() * 100:.2f}% of data) with invalid values")
                for col, info in drop_reasons.items():
                    print(f"  - Column '{col}': {info['count']} invalid values, examples: {info['examples']}")
                print(f"  Table size: {original_count} → {len(df)} rows")
            else:
                print(f"No problematic rows found in table {table_name}")

    def generate_synthetic_data(self, method='auto', parameters=None):
        """Generate synthetic data with guaranteed output"""
        logging.info(f"Generating synthetic data using method: {method}")

        # Check if perturbation mode is enabled
        if hasattr(self, 'apply_perturbation') and self.apply_perturbation:
            logging.info("Using perturbation mode instead of full synthesis")
            return self.generate_perturbed_data()

        # Check if we have original data
        if not self.original_data:
            logging.error("No original data available for synthetic generation")
            return False

        # Initialize generator if not already done
        if self.generator is None:
            try:
                self.generator = SyntheticGenerationEngine(self.schema)
            except Exception as e:
                logging.error(f"Error initializing generator: {str(e)}")
                logging.error(traceback.format_exc())
                self.generator = None

        # Generate synthetic data for each table
        success = False
        for table_name, df in self.original_data.items():
            logging.info(f"Generating synthetic data for table: {table_name} with shape {df.shape}")

            # Determine number of samples
            n_samples = len(df)
            if n_samples == 0:
                logging.warning(f"Original table {table_name} is empty, skipping")
                continue

            # Identify columns to synthesize versus columns to copy
            columns_to_synthesize = []
            columns_to_copy = []

            for column in df.columns:
                if (table_name in self.schema and
                        'columns' in self.schema[table_name] and
                        column in self.schema[table_name]['columns'] and
                        self.schema[table_name]['columns'][column].get('synthesize', True)):
                    columns_to_synthesize.append(column)
                else:
                    columns_to_copy.append(column)

            logging.info(
                f"Columns to synthesize: {len(columns_to_synthesize)}, Columns to copy: {len(columns_to_copy)}")

            # Initialize synthetic_df as None - will be set later
            synthetic_df = None
            full_synthetic_df = None

            # Important: Train on ALL columns to preserve relationships
            try:
                # Train models on ALL columns to capture relationships
                if method == 'auto':
                    logging.info("Training models with auto method on all columns")
                    self.generator.train_models(df)  # Train on the entire dataframe

                    # Generate synthetic data for the entire structure
                    full_synthetic_df = self.generator.generate_synthetic_data(n_samples)

                    if full_synthetic_df is None or full_synthetic_df.empty:
                        logging.error(f"Generation produced empty DataFrame for {table_name}")
                        full_synthetic_df = self._create_fallback_synthetic_data(df, n_samples)

                    # Extract only the columns that need to be synthesized
                    if columns_to_synthesize:
                        # Filter for columns that exist in full_synthetic_df
                        available_columns = [col for col in columns_to_synthesize if col in full_synthetic_df.columns]
                        if available_columns:
                            synthetic_df = full_synthetic_df[available_columns]
                        else:
                            logging.warning("No columns to synthesize found in generated data")
                            synthetic_df = pd.DataFrame(index=range(n_samples))
                    else:
                        synthetic_df = pd.DataFrame(index=range(n_samples))

                elif method == 'ctgan':
                    # Other methods...
                    pass
                else:
                    raise ValueError(f"Unsupported generation method: {method}")

            except Exception as e:
                logging.error(f"Error generating synthetic data for {table_name}: {str(e)}")
                logging.error(traceback.format_exc())
                synthetic_df = None  # Ensure it's defined for later checks

            # Check if generation was successful
            if synthetic_df is None or synthetic_df.empty:
                logging.warning(f"Primary generation failed for {table_name}, using fallback")
                if columns_to_synthesize:
                    synthetic_df = self._create_fallback_synthetic_data(df[columns_to_synthesize], n_samples)
                else:
                    synthetic_df = pd.DataFrame(index=range(n_samples))

            # Final verification of synthetic data
            if synthetic_df is None or synthetic_df.empty:
                logging.error(f"Both primary and fallback generation failed for {table_name}")
                continue

            # Create a final DataFrame with all columns
            final_df = pd.DataFrame(index=range(n_samples))

            # Add synthesized columns
            for column in columns_to_synthesize:
                if synthetic_df is not None and column in synthetic_df.columns:
                    final_df[column] = synthetic_df[column]
                else:
                    logging.warning(f"Synthesized data missing column {column}, using fallback")
                    final_df[column] = self._generate_fallback_column(df[column], n_samples)

            # Add copied columns - sample randomly from original data
            if columns_to_copy:
                # Create a random index mapping for sampling
                original_indices = np.random.choice(len(df), size=n_samples, replace=len(df) < n_samples)
                for column in columns_to_copy:
                    final_df[column] = df[column].iloc[original_indices].reset_index(drop=True)

            # Ensure all original columns are present
            for column in df.columns:
                if column not in final_df.columns:
                    logging.warning(f"Column {column} missing from final data, adding it")
                    final_df[column] = self._generate_fallback_column(df[column], n_samples)

            # Apply name abstraction only if requested by user
            if hasattr(self, 'apply_name_abstraction') and self.apply_name_abstraction:
                logging.info(f"Applying name abstraction for table: {table_name}")
                final_df = self.post_process_names(final_df, table_name, n_samples)
            else:
                logging.info(f"Skipping name abstraction as per user preference")

            # Apply general column abstractions if applicable
            if table_name in self.schema and 'columns' in self.schema[table_name]:
                logging.info(f"Applying general column abstractions for table: {table_name}")
                final_df = self.apply_abstractions(final_df, table_name)

            # Store the final synthetic data
            self.synthetic_data[table_name] = final_df

            # Log the completed generation
            logging.info(
                f"Generated synthetic data for {table_name}: {len(final_df)} rows, {len(final_df.columns)} columns")
            success = True

        # FIXED: Use the correct boolean flag name and call the method properly
        if hasattr(self, 'should_apply_age_grouping') and self.should_apply_age_grouping:
            logging.info("Applying age grouping to all synthetic data tables")
            for table_name in self.synthetic_data.keys():
                logging.info(f"Applying age grouping to synthetic data for table: {table_name}")
                self.synthetic_data[table_name] = self.apply_age_grouping(
                    self.synthetic_data[table_name], table_name
                )

        # Apply address synthesis if configured
        if hasattr(self, 'should_apply_address_synthesis') and self.should_apply_address_synthesis:
            logging.info("Applying address synthesis to all synthetic data tables")
            for table_name in self.synthetic_data.keys():
                logging.info(f"Applying address synthesis to synthetic data for table: {table_name}")
                self.synthetic_data[table_name] = self.apply_address_synthesis(
                    self.synthetic_data[table_name], table_name
                )

        # Check overall success
        if not success:
            logging.error("Failed to generate any synthetic data")
            return False

        # Log synthetic data sizes
        for table_name, synthetic_df in self.synthetic_data.items():
            logging.info(
                f"Table {table_name} synthetic data: {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns")

        return True

    def _generate_fallback_column(self, original_series, n_samples):
        """Generate fallback values for a single column"""
        try:
            # Sample with replacement from original values
            if len(original_series) > 0:
                return original_series.sample(n_samples, replace=True).reset_index(drop=True)
            else:
                # If original is empty, return empty/NA values
                return pd.Series([None] * n_samples)
        except Exception as e:
            logging.error(f"Error generating fallback column: {str(e)}")
            # Return empty/NA values
            return pd.Series([None] * n_samples)

    def _create_fallback_synthetic_data(self, original_df, n_samples):
        """Create fallback synthetic data when the generator fails"""
        logging.warning("Using fallback synthetic data generation")

        try:
            # Create a copy of the dataframe structure
            synthetic_df = pd.DataFrame(columns=original_df.columns)

            # Determine how to populate the data
            if n_samples <= len(original_df):
                # If we need fewer rows than original, sample from original
                synthetic_df = original_df.sample(n_samples, replace=False).copy().reset_index(drop=True)
            else:
                # If we need more rows, sample with replacement
                synthetic_df = original_df.sample(n_samples, replace=True).copy().reset_index(drop=True)

            logging.info(f"Created basic synthetic dataframe with {len(synthetic_df)} rows")

            # Now add some variation to make it actually "synthetic"
            for column in synthetic_df.columns:
                try:
                    # Identify column type
                    if pd.api.types.is_numeric_dtype(synthetic_df[column]):
                        # For numeric columns, add small random variation
                        column_std = synthetic_df[column].std()
                        if column_std > 0:
                            noise = np.random.normal(0, column_std * 0.1, len(synthetic_df))
                            synthetic_df[column] = synthetic_df[column] + noise

                    elif pd.api.types.is_datetime64_any_dtype(synthetic_df[column]):
                        # For datetime columns, add small random offsets
                        dates = pd.to_datetime(synthetic_df[column])
                        offsets = pd.to_timedelta(np.random.randint(-5, 6, len(synthetic_df)), unit='d')
                        synthetic_df[column] = dates + offsets

                    elif synthetic_df[column].nunique() < 10:
                        # For categorical columns with few values, shuffle some values
                        shuffle_mask = np.random.random(len(synthetic_df)) < 0.2
                        if shuffle_mask.any():
                            shuffle_indices = np.where(shuffle_mask)[0]
                            shuffle_values = synthetic_df.loc[shuffle_indices, column].values
                            np.random.shuffle(shuffle_values)
                            synthetic_df.loc[shuffle_indices, column] = shuffle_values
                except Exception as e:
                    logging.warning(f"Error adding variation to column {column}: {str(e)}")

            logging.info(f"Fallback synthetic data created with shape {synthetic_df.shape}")
            return synthetic_df

        except Exception as e:
            logging.error(f"Fallback synthetic data generation failed: {str(e)}")
            logging.error(traceback.format_exc())

            # Ultimate fallback - create minimal synthetic data
            try:
                # Create a simple dataframe with the right columns
                minimal_df = pd.DataFrame(columns=original_df.columns)

                # Add at least one row with default values
                row_data = {}
                for column in original_df.columns:
                    try:
                        # Try to get a valid value from the original data
                        valid_values = original_df[column].dropna()
                        if len(valid_values) > 0:
                            row_data[column] = valid_values.iloc[0]
                        else:
                            # Use an appropriate default value
                            if pd.api.types.is_numeric_dtype(original_df[column]):
                                row_data[column] = 0
                            elif pd.api.types.is_datetime64_any_dtype(original_df[column]):
                                row_data[column] = pd.Timestamp('2023-01-01')
                            else:
                                row_data[column] = "Unknown"
                    except:
                        row_data[column] = None

                # Add the row to the dataframe
                minimal_df = minimal_df.append(row_data, ignore_index=True)

                # Replicate to reach n_samples
                while len(minimal_df) < n_samples:
                    minimal_df = minimal_df.append(row_data, ignore_index=True)

                logging.warning(f"Created minimal synthetic data with {len(minimal_df)} rows")
                return minimal_df
            except Exception as e:
                logging.error(f"Even minimal fallback generation failed: {str(e)}")
                return pd.DataFrame(columns=original_df.columns)

    def evaluate_synthetic_data(self):
        """Evaluate the quality of synthetic data"""
        logging.info("Evaluating synthetic data")

        # Initialize evaluator if not already done
        if self.evaluator is None:
            self.evaluator = SyntheticDataEvaluator()
            self.evaluator.set_schema(self.schema)

        # Evaluate each table
        results = {}

        for table_name in self.original_data.keys():
            if table_name not in self.synthetic_data:
                continue

            logging.info(f"Evaluating table: {table_name}")

            original_df = self.original_data[table_name]
            synthetic_df = self.synthetic_data[table_name]

            # Ensure consistent types before evaluation
            for column in original_df.columns:
                if column not in synthetic_df.columns:
                    continue

                # Check if it's a datetime column in schema
                is_datetime = False
                if (table_name in self.schema and
                        'columns' in self.schema[table_name] and
                        column in self.schema[table_name]['columns']):
                    is_datetime = self.schema[table_name]['columns'][column].get('type') == 'datetime'

                if is_datetime or 'date' in column.lower():
                    try:
                        # For date columns, convert both to strings in a consistent format
                        original_df[column] = pd.to_datetime(original_df[column], errors='coerce')
                        synthetic_df[column] = pd.to_datetime(synthetic_df[column], errors='coerce')

                        # Convert to string to avoid comparison issues
                        original_df[column] = original_df[column].dt.strftime('%Y-%m-%d')
                        synthetic_df[column] = synthetic_df[column].dt.strftime('%Y-%m-%d')
                    except Exception as e:
                        logging.warning(f"Could not process date column {column}: {str(e)}")
                        # Convert both to string to be safe
                        original_df[column] = original_df[column].astype(str)
                        synthetic_df[column] = synthetic_df[column].astype(str)

            # Evaluate statistical similarity
            stats_results = self.evaluator.evaluate_statistical_similarity(
                original_df, synthetic_df
            )

            # Evaluate privacy risk
            privacy_results = self.evaluator.evaluate_privacy_risk(
                original_df, synthetic_df
            )

            # Evaluate ML utility
            ml_results = self.evaluator.evaluate_ml_utility(
                original_df, synthetic_df
            )

            # Combine results
            results[table_name] = {
                'statistical_similarity': stats_results,
                'privacy_risk': privacy_results,
                'ml_utility': ml_results,
                'overall_score': (
                        0.4 * stats_results['overall_score'] +
                        0.4 * ml_results['overall_score'] +
                        0.2 * (1.0 - privacy_results['overall_score'])  # Lower privacy risk is better
                )
            }

        self.evaluation_results = results

        # Log overall scores
        for table_name, table_results in results.items():
            logging.info(f"Table {table_name} - Overall score: {table_results['overall_score']:.4f}")

        return results

    def validate_synthetic_data(self):
        """Validate synthetic data against constraints"""
        logging.info("Validating synthetic data")

        # Initialize validator if not already done
        if self.validator is None:
            self.validator = DataValidator(self.schema)

        # Validate each table
        validation_results = {}

        for table_name, synthetic_df in self.synthetic_data.items():
            logging.info(f"Validating table: {table_name}")

            # Validate against schema and constraints
            issues = self.validator.validate(synthetic_df)

            # Store validation results
            validation_results[table_name] = {
                'errors': issues,  # This assumes validate() returns a list of issues
                'has_errors': len(issues) > 0
            }

            # Log validation results
            if issues:
                logging.warning(f"Table {table_name} has {len(issues)} validation errors")
                for issue in issues:
                    logging.warning(f"  - {issue}")

        return validation_results

    def export_synthetic_data(self, format_type: str, output_path: str) -> bool:
        """Export synthetic data to specified format with guaranteed output"""
        logging.info(f"Exporting synthetic data to {format_type} format: {output_path}")

        try:
            # Check if we have data to export
            if not self.synthetic_data:
                logging.error("No synthetic data to export")
                return False

            # Check if we have non-empty tables
            non_empty_tables = {k: v for k, v in self.synthetic_data.items() if not v.empty}
            if not non_empty_tables:
                logging.error("All synthetic data tables are empty")
                return False

            logging.info(f"Exporting {len(non_empty_tables)} non-empty tables")

            # Make sure output path is valid
            if not output_path:
                output_path = "synthetic_data"

            # If it's a directory, make sure it exists
            if os.path.isdir(output_path) or output_path.endswith('/') or output_path.endswith('\\'):
                os.makedirs(output_path, exist_ok=True)
            else:
                # Make sure the parent directory exists
                parent_dir = os.path.dirname(output_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

            # Handle different export formats
            if format_type.lower() == 'csv':
                # Direct CSV export
                success = False
                for table_name, df in non_empty_tables.items():
                    try:
                        # Create the file path
                        if os.path.isdir(output_path):
                            file_path = os.path.join(output_path, f"{table_name}.csv")
                        else:
                            file_path = f"{output_path}_{table_name}.csv"

                        # Export to CSV
                        df.to_csv(file_path, index=False)
                        logging.info(f"Exported {len(df)} rows to {file_path}")
                        success = True
                    except Exception as e:
                        logging.error(f"Error exporting {table_name} to CSV: {str(e)}")

                return success

            # Handle other export formats...

            # Use export adapter as last resort
            try:
                # Create export adapter
                exporter = ExportAdapter.create_adapter(format_type)

                # Export synthetic data
                result = exporter.export(non_empty_tables, output_path)
                logging.info(f"Export adapter result: {result}")
                return result
            except Exception as e:
                logging.error(f"Export adapter error: {str(e)}")
                # Fall back to CSV export
                logging.warning("Falling back to CSV export")
                return self._emergency_csv_export(non_empty_tables, output_path)

        except Exception as e:
            logging.error(f"Error in export_synthetic_data: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def _emergency_csv_export(self, tables, output_path):
        """Emergency CSV export when all else fails"""
        try:
            logging.warning("Using emergency CSV export")

            success = False
            for table_name, df in tables.items():
                try:
                    # Create a simple filename
                    file_path = f"{output_path}_{table_name}_emergency.csv"

                    # Export to CSV
                    df.to_csv(file_path, index=False)
                    logging.info(f"Emergency export: {len(df)} rows to {file_path}")
                    success = True
                except Exception as inner_e:
                    logging.error(f"Emergency export failed for {table_name}: {str(inner_e)}")

            return success
        except Exception as e:
            logging.error(f"Emergency export mechanism failed: {str(e)}")
            return False

    def load_data(self, input_path):
        """Load data from various sources based on the input path"""
        if os.path.isdir(input_path):
            # If input_path is a directory, load CSV files
            return self.load_csv_directory(input_path)
        elif input_path.endswith('.csv'):
            # If input_path is a single CSV file
            table_name = os.path.splitext(os.path.basename(input_path))[0]
            self.original_data[table_name] = pd.read_csv(input_path)
            return True
        else:
            # Assume it's a database connection string
            return self.connect_database(input_path)

    def perturb_data(self, df, table_name):
        """Apply controlled perturbation to data rows while preserving structure and relationships"""
        logging.info(f"Applying data perturbation to maintain similarity with controlled deviations")

        result_df = df.copy()
        schema_info = self.schema.get(table_name, {}).get('columns', {})

        # Get column types for intelligent perturbation
        column_types = {}
        for column in result_df.columns:
            # Try to determine column type
            col_type = "unknown"
            if column in schema_info:
                col_type = schema_info[column].get('type', 'unknown')
            elif pd.api.types.is_numeric_dtype(result_df[column]):
                col_type = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(result_df[column]):
                col_type = 'datetime'
            elif result_df[column].nunique() < 10:
                col_type = 'categorical'
            elif any(term in column.lower() for term in ['name', 'first', 'last']):
                col_type = 'name'
            elif any(term in column.lower() for term in ['address', 'street', 'city']):
                col_type = 'address'
            elif any(term in column.lower() for term in ['email']):
                col_type = 'email'

            column_types[column] = col_type

        logging.info(f"Identified column types: {column_types}")

        # Apply appropriate perturbation based on column type
        for column, col_type in column_types.items():
            perturbation_factor = schema_info.get(column, {}).get('perturbation_factor', 0.2)  # Default 20% change

            try:
                # Different perturbation approaches for different column types
                if col_type == 'numeric':
                    # Check if it looks like an age column
                    if 'age' in column.lower():
                        # Age-specific perturbation (±5 years)
                        noise = np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], size=len(result_df))
                        result_df[column] = result_df[column] + noise
                        # Enforce reasonable age limits
                        result_df[column] = result_df[column].clip(0, 110)
                    else:
                        # General numeric perturbation
                        column_std = result_df[column].std()
                        if column_std > 0:
                            # Add random noise proportional to the standard deviation
                            noise = np.random.normal(0, column_std * perturbation_factor, len(result_df))
                            result_df[column] = result_df[column] + noise

                            # Round to original precision
                            original_decimals = self._get_decimal_precision(df[column])
                            result_df[column] = result_df[column].round(original_decimals)

                elif col_type == 'datetime':
                    # Add small random offsets (±7 days)
                    try:
                        dates = pd.to_datetime(result_df[column])
                        # Random offsets between -7 and +7 days
                        offsets = pd.to_timedelta(np.random.randint(-7, 8, len(result_df)), unit='d')
                        result_df[column] = dates + offsets
                    except:
                        logging.warning(f"Could not perturb datetime column {column}")

                elif col_type == 'categorical':
                    # Only swap categories for a small percentage of rows
                    swap_probability = perturbation_factor
                    swap_mask = np.random.random(len(result_df)) < swap_probability

                    if swap_mask.any():
                        # Get unique category values
                        categories = result_df[column].unique()

                        # For each row marked for swap, assign a different category
                        for idx in result_df.index[swap_mask]:
                            current_val = result_df.at[idx, column]
                            other_categories = [c for c in categories if c != current_val]

                            if other_categories:
                                result_df.at[idx, column] = np.random.choice(other_categories)

                elif col_type == 'name':
                    # Preserve first names but swap last names for a percentage of rows
                    if not self.apply_name_abstraction:  # Only if we're not already doing full name anonymization
                        swap_probability = perturbation_factor
                        swap_mask = np.random.random(len(result_df)) < swap_probability

                        if swap_mask.any() and len(result_df) > 1:
                            # Shuffle the last names among the marked rows
                            swap_indices = np.where(swap_mask)[0]
                            last_names = result_df.loc[swap_indices, column].values.copy()
                            np.random.shuffle(last_names)
                            result_df.loc[swap_indices, column] = last_names

                # Add more type-specific perturbation methods as needed

            except Exception as e:
                logging.warning(f"Error perturbing column {column}: {str(e)}")

        logging.info(f"Perturbation complete for {table_name}")
        return result_df

    def _get_decimal_precision(self, series):
        """Determine the decimal precision of a numeric series"""
        try:
            # Extract decimal parts of the numbers
            decimal_parts = series.apply(lambda x: str(x).split('.')[-1] if '.' in str(x) else '')

            # Find the maximum decimal places
            if decimal_parts.str.len().max() > 0:
                return decimal_parts.str.len().max()
            return 0
        except:
            return 0

    def identify_perturbation_options(self):
        """Interactive configuration of perturbation settings"""
        print("\n=== Data Perturbation Settings ===")
        print("Do you want to apply data perturbation (small controlled changes) instead of full synthesis? (yes/no)")
        user_input = input("> ").strip().lower()

        self.apply_perturbation = user_input.startswith('y')

        if self.apply_perturbation:
            print("\nSelect perturbation level:")
            print("1. Minimal (±5% deviation from original values)")
            print("2. Moderate (±10% deviation from original values)")
            print("3. Standard (±20% deviation from original values)")
            print("4. High (±30% deviation from original values)")
            print("5. Custom (specify your own percentage)")

            level_input = input("> ").strip()

            try:
                level = int(level_input)
                if level == 1:
                    self.perturbation_factor = 0.05
                elif level == 2:
                    self.perturbation_factor = 0.1
                elif level == 3:
                    self.perturbation_factor = 0.2
                elif level == 4:
                    self.perturbation_factor = 0.3
                elif level == 5:
                    print("Enter custom perturbation percentage (e.g., 15 for 15%):")
                    custom = float(input("> ").strip())
                    self.perturbation_factor = custom / 100.0
                else:
                    self.perturbation_factor = 0.2  # Default
            except:
                print("Invalid input, using standard perturbation level (20%)")
                self.perturbation_factor = 0.2

            print(f"Data will be perturbed with approximately {self.perturbation_factor * 100:.1f}% deviation")

            # Ask about per-column settings
            print("\nDo you want to configure perturbation settings for specific columns? (yes/no)")
            column_config = input("> ").strip().lower()

            if column_config.startswith('y'):
                self.configure_column_perturbation()
        else:
            print("Full synthetic data generation will be used instead of perturbation")

    def configure_column_perturbation(self):
        """Configure perturbation settings for specific columns"""
        for table_name, df in self.original_data.items():
            print(f"\n=== Table: {table_name} ===")

            # Show all columns with types
            columns = list(df.columns)
            for i, column in enumerate(columns):
                # Determine column type
                col_type = "unknown"
                if (table_name in self.schema and
                        'columns' in self.schema[table_name] and
                        column in self.schema[table_name]['columns']):
                    col_type = self.schema[table_name]['columns'][column].get('type', 'unknown')
                elif pd.api.types.is_numeric_dtype(df[column]):
                    col_type = 'numeric'

                print(f"{i + 1}. {column} ({col_type})")

            print("\nEnter the numbers of columns to configure, separated by commas (or 'none'):")
            user_input = input("> ").strip().lower()

            if user_input == 'none':
                continue

            selected_indices = []
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
            except:
                print("Invalid input, no columns selected")
                continue

            # Configure each selected column
            for idx in selected_indices:
                if 0 <= idx < len(columns):
                    column = columns[idx]

                    print(f"\nConfiguration for column: {column}")
                    print("Enter perturbation factor (percentage, e.g., 15 for 15%):")

                    try:
                        factor = float(input("> ").strip())
                        factor = factor / 100.0  # Convert percentage to decimal

                        # Store in schema
                        if table_name not in self.schema:
                            self.schema[table_name] = {'columns': {}}
                        if 'columns' not in self.schema[table_name]:
                            self.schema[table_name]['columns'] = {}
                        if column not in self.schema[table_name]['columns']:
                            self.schema[table_name]['columns'][column] = {}

                        self.schema[table_name]['columns'][column]['perturbation_factor'] = factor
                        print(f"Set perturbation factor for {column} to {factor * 100:.1f}%")
                    except:
                        print("Invalid input, using default perturbation factor")

    def generate_perturbed_data(self):
        """Generate data by applying perturbation to the original data"""
        logging.info("Generating perturbed data with controlled deviations")

        if not self.original_data:
            logging.error("No original data available for perturbation")
            return False

        success = False
        for table_name, df in self.original_data.items():
            logging.info(f"Perturbing data for table: {table_name}")

            if len(df) == 0:
                logging.warning(f"Original table {table_name} is empty, skipping")
                continue

            try:
                # Create a copy of the original data
                perturbed_df = df.copy()

                # FIXED: Apply age grouping if configured
                if hasattr(self, 'should_apply_age_grouping') and self.should_apply_age_grouping:
                    logging.info(f"Applying age grouping for table: {table_name}")
                    perturbed_df = self.apply_age_grouping(perturbed_df, table_name)

                # Apply perturbation
                perturbed_df = self.perturb_data(perturbed_df, table_name)

                # Apply name abstraction if requested
                if hasattr(self, 'apply_name_abstraction') and self.apply_name_abstraction:
                    logging.info(f"Applying name abstraction for table: {table_name}")
                    perturbed_df = self.post_process_names(perturbed_df, table_name, len(perturbed_df))

                # Apply other abstractions if configured
                if table_name in self.schema and 'columns' in self.schema[table_name]:
                    logging.info(f"Applying column abstractions for table: {table_name}")
                    perturbed_df = self.apply_abstractions(perturbed_df, table_name)

                # Store the perturbed data
                self.synthetic_data[table_name] = perturbed_df
                logging.info(f"Successfully generated perturbed data for {table_name} with {len(perturbed_df)} rows")
                success = True

            except Exception as e:
                logging.error(f"Error perturbing data for {table_name}: {str(e)}")
                logging.error(traceback.format_exc())

        return success

    def evaluate_original_data(self):
        """Comprehensive evaluation of the original data"""
        logging.info("Evaluating original data")

        if not self.original_data:
            logging.error("No original data available for evaluation")
            return None

        evaluation_results = {}

        for table_name, df in self.original_data.items():
            logging.info(f"Evaluating original data for table: {table_name}")

            # Basic statistics
            basic_stats = self._calculate_basic_statistics(df, "Original")

            # Data quality metrics
            quality_metrics = self._calculate_data_quality(df, "Original")

            # Distribution analysis
            distribution_analysis = self._analyze_distributions(df, table_name, "Original")

            # Correlation analysis
            correlation_analysis = self._analyze_correlations_detailed(df, "Original")

            # Store results
            evaluation_results[table_name] = {
                'basic_statistics': basic_stats,
                'data_quality': quality_metrics,
                'distributions': distribution_analysis,
                'correlations': correlation_analysis,
                'data_type': 'Original',
                'table_name': table_name
            }

        return evaluation_results

    def evaluate_synthetic_data(self):
        """Comprehensive evaluation of the synthetic data"""
        logging.info("Evaluating synthetic data")

        if not self.synthetic_data:
            logging.error("No synthetic data available for evaluation")
            return None

        evaluation_results = {}

        for table_name, df in self.synthetic_data.items():
            logging.info(f"Evaluating synthetic data for table: {table_name}")

            # Basic statistics
            basic_stats = self._calculate_basic_statistics(df, "Synthetic")

            # Data quality metrics
            quality_metrics = self._calculate_data_quality(df, "Synthetic")

            # Distribution analysis
            distribution_analysis = self._analyze_distributions(df, table_name, "Synthetic")

            # Correlation analysis
            correlation_analysis = self._analyze_correlations_detailed(df, "Synthetic")

            # Store results
            evaluation_results[table_name] = {
                'basic_statistics': basic_stats,
                'data_quality': quality_metrics,
                'distributions': distribution_analysis,
                'correlations': correlation_analysis,
                'data_type': 'Synthetic',
                'table_name': table_name
            }

        return evaluation_results

    def _calculate_basic_statistics(self, df, data_type):
        """Calculate basic statistics for a dataframe"""
        stats = {
            'data_type': data_type,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': {}
        }

        for column in df.columns:
            col_stats = {
                'dtype': str(df[column].dtype),
                'non_null_count': df[column].count(),
                'null_count': df[column].isnull().sum(),
                'null_percentage': (df[column].isnull().sum() / len(df)) * 100,
                'unique_count': df[column].nunique(),
                'unique_percentage': (df[column].nunique() / len(df)) * 100
            }

            # Add statistics based on data type
            if pd.api.types.is_numeric_dtype(df[column]):
                col_stats.update({
                    'mean': df[column].mean(),
                    'median': df[column].median(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'q25': df[column].quantile(0.25),
                    'q75': df[column].quantile(0.75),
                    'skewness': df[column].skew(),
                    'kurtosis': df[column].kurtosis()
                })
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                try:
                    col_stats.update({
                        'min_date': df[column].min(),
                        'max_date': df[column].max(),
                        'date_range_days': (df[column].max() - df[column].min()).days
                    })
                except:
                    pass
            else:
                # Categorical data
                value_counts = df[column].value_counts()
                col_stats.update({
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'mode_percentage': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0
                })

            stats['columns'][column] = col_stats

        return stats

    def _calculate_data_quality(self, df, data_type):
        """Calculate data quality metrics"""
        quality = {
            'data_type': data_type,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'complete_rows': df.dropna().shape[0],
            'complete_percentage': (df.dropna().shape[0] / len(df)) * 100,
            'columns_with_nulls': df.isnull().any().sum(),
            'total_null_values': df.isnull().sum().sum(),
            'column_quality': {}
        }

        for column in df.columns:
            col_quality = {
                'completeness': (df[column].count() / len(df)) * 100,
                'uniqueness': (df[column].nunique() / len(df)) * 100 if len(df) > 0 else 0,
                'consistency': 100  # Placeholder - could add specific consistency checks
            }

            # Check for potential data issues
            issues = []
            if df[column].isnull().sum() > 0:
                issues.append(f"{df[column].isnull().sum()} null values")

            if pd.api.types.is_numeric_dtype(df[column]):
                # Check for outliers using IQR method
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                outliers = ((df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))).sum()
                if outliers > 0:
                    issues.append(f"{outliers} outliers")

            col_quality['issues'] = issues
            quality['column_quality'][column] = col_quality

        return quality

    def _analyze_distributions(self, df, table_name, data_type):
        """Analyze data distributions"""
        distributions = {
            'data_type': data_type,
            'table_name': table_name,
            'numeric_distributions': {},
            'categorical_distributions': {}
        }

        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                # Basic distribution stats
                dist_stats = {
                    'distribution_type': 'numeric',
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'normality_test': None
                }

                # Test for normality (if sample size is reasonable)
                if len(df) < 5000:  # Avoid memory issues with large datasets
                    from scipy import stats
                    try:
                        stat, p_value = stats.normaltest(df[col].dropna())
                        dist_stats['normality_test'] = {
                            'statistic': stat,
                            'p_value': p_value,
                            'is_normal': p_value > 0.05
                        }
                    except:
                        pass

                distributions['numeric_distributions'][col] = dist_stats
            except:
                pass

        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()
                dist_stats = {
                    'distribution_type': 'categorical',
                    'unique_values': len(value_counts),
                    'top_5_values': value_counts.head(5).to_dict(),
                    'entropy': self._calculate_entropy(value_counts),
                    'concentration_ratio': value_counts.iloc[0] / len(df) if len(value_counts) > 0 else 0
                }

                distributions['categorical_distributions'][col] = dist_stats
            except:
                pass

        return distributions

    def _analyze_correlations_detailed(self, df, data_type):
        """Detailed correlation analysis"""
        correlations = {
            'data_type': data_type,
            'numeric_correlations': None,
            'strong_correlations': [],
            'correlation_summary': {}
        }

        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            correlations['numeric_correlations'] = corr_matrix.to_dict()

            # Find strong correlations (>0.7 or <-0.7)
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_corrs.append({
                            'column1': corr_matrix.columns[i],
                            'column2': corr_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })

            correlations['strong_correlations'] = strong_corrs

            # Correlation summary
            correlations['correlation_summary'] = {
                'max_correlation': corr_matrix.abs().max().max(),
                'mean_correlation': corr_matrix.abs().mean().mean(),
                'strong_pairs': len(strong_corrs)
            }

        return correlations

    def _calculate_entropy(self, value_counts):
        """Calculate Shannon entropy for categorical data"""
        try:
            from scipy import stats
            probabilities = value_counts / value_counts.sum()
            return stats.entropy(probabilities)
        except:
            return None

    def print_evaluation_report(self, original_eval=None, synthetic_eval=None):
        """Print comprehensive evaluation report comparing original and synthetic data"""

        print("\n" + "=" * 80)
        print("COMPREHENSIVE DATA EVALUATION REPORT")
        print("=" * 80)

        # If evaluations not provided, calculate them
        if original_eval is None:
            original_eval = self.evaluate_original_data()
        if synthetic_eval is None and self.synthetic_data:
            synthetic_eval = self.evaluate_synthetic_data()

        for table_name in original_eval.keys():
            print(f"\n{'=' * 60}")
            print(f"TABLE: {table_name.upper()}")
            print(f"{'=' * 60}")

            orig_data = original_eval[table_name]
            synth_data = synthetic_eval.get(table_name) if synthetic_eval else None

            # 1. BASIC STATISTICS COMPARISON
            print(f"\n{'-' * 40}")
            print("1. BASIC STATISTICS")
            print(f"{'-' * 40}")

            print(f"{'Metric':<25} {'Original':<15} {'Synthetic':<15} {'Difference':<15}")
            print("-" * 70)

            orig_stats = orig_data['basic_statistics']
            synth_stats = synth_data['basic_statistics'] if synth_data else None

            # Row and column comparison - Fixed formatting
            orig_rows = orig_stats['total_rows']
            synth_rows = synth_stats['total_rows'] if synth_stats else 'N/A'
            diff_rows = (synth_stats['total_rows'] - orig_stats['total_rows']) if synth_stats else 'N/A'

            print(f"{'Total Rows':<25} {orig_rows:<15} {synth_rows:<15} {diff_rows:<15}")

            orig_cols = orig_stats['total_columns']
            synth_cols = synth_stats['total_columns'] if synth_stats else 'N/A'
            diff_cols = (synth_stats['total_columns'] - orig_stats['total_columns']) if synth_stats else 'N/A'

            print(f"{'Total Columns':<25} {orig_cols:<15} {synth_cols:<15} {diff_cols:<15}")

            # Memory usage - Fixed formatting
            orig_memory = f"{orig_stats['memory_usage_mb']:.2f}"
            synth_memory = f"{synth_stats['memory_usage_mb']:.2f}" if synth_stats else 'N/A'
            diff_memory = f"{(synth_stats['memory_usage_mb'] - orig_stats['memory_usage_mb']):.2f}" if synth_stats else 'N/A'

            print(f"{'Memory (MB)':<25} {orig_memory:<15} {synth_memory:<15} {diff_memory:<15}")

            # 2. DATA QUALITY COMPARISON
            print(f"\n{'-' * 40}")
            print("2. DATA QUALITY")
            print(f"{'-' * 40}")

            orig_quality = orig_data['data_quality']
            synth_quality = synth_data['data_quality'] if synth_data else None

            print(f"{'Metric':<25} {'Original':<15} {'Synthetic':<15} {'Difference':<15}")
            print("-" * 70)

            # Duplicate rows
            orig_dup = orig_quality['duplicate_rows']
            synth_dup = synth_quality['duplicate_rows'] if synth_quality else 'N/A'
            diff_dup = (synth_quality['duplicate_rows'] - orig_quality['duplicate_rows']) if synth_quality else 'N/A'

            print(f"{'Duplicate Rows':<25} {orig_dup:<15} {synth_dup:<15} {diff_dup:<15}")

            # Complete rows percentage - Fixed formatting
            orig_complete = f"{orig_quality['complete_percentage']:.2f}"
            synth_complete = f"{synth_quality['complete_percentage']:.2f}" if synth_quality else 'N/A'
            diff_complete = f"{(synth_quality['complete_percentage'] - orig_quality['complete_percentage']):.2f}" if synth_quality else 'N/A'

            print(f"{'Complete Rows %':<25} {orig_complete:<15} {synth_complete:<15} {diff_complete:<15}")

            # Total null values
            orig_nulls = orig_quality['total_null_values']
            synth_nulls = synth_quality['total_null_values'] if synth_quality else 'N/A'
            diff_nulls = (synth_quality['total_null_values'] - orig_quality[
                'total_null_values']) if synth_quality else 'N/A'

            print(f"{'Total Null Values':<25} {orig_nulls:<15} {synth_nulls:<15} {diff_nulls:<15}")

            # 3. COLUMN-BY-COLUMN COMPARISON
            print(f"\n{'-' * 40}")
            print("3. COLUMN-BY-COLUMN ANALYSIS")
            print(f"{'-' * 40}")

            for column in orig_stats['columns'].keys():
                print(f"\nColumn: {column}")
                print("-" * 30)

                orig_col = orig_stats['columns'][column]
                synth_col = synth_stats['columns'].get(column) if synth_stats else None

                print(f"  Data Type: {orig_col['dtype']}")
                print(f"  Original - Unique: {orig_col['unique_count']}, Nulls: {orig_col['null_count']}")
                if synth_col:
                    print(f"  Synthetic - Unique: {synth_col['unique_count']}, Nulls: {synth_col['null_count']}")

                # Numeric column details - Fixed formatting
                if 'mean' in orig_col:
                    orig_mean = f"{orig_col['mean']:.2f}"
                    orig_std = f"{orig_col['std']:.2f}"
                    orig_min = f"{orig_col['min']:.2f}"
                    orig_max = f"{orig_col['max']:.2f}"

                    print(f"  Original Stats - Mean: {orig_mean}, Std: {orig_std}, Range: [{orig_min}, {orig_max}]")

                    if synth_col and 'mean' in synth_col:
                        synth_mean = f"{synth_col['mean']:.2f}"
                        synth_std = f"{synth_col['std']:.2f}"
                        synth_min = f"{synth_col['min']:.2f}"
                        synth_max = f"{synth_col['max']:.2f}"

                        print(
                            f"  Synthetic Stats - Mean: {synth_mean}, Std: {synth_std}, Range: [{synth_min}, {synth_max}]")

                        diff_mean = f"{synth_col['mean'] - orig_col['mean']:.2f}"
                        diff_std = f"{synth_col['std'] - orig_col['std']:.2f}"

                        print(f"  Difference - Mean: {diff_mean}, Std: {diff_std}")

            # 4. DISTRIBUTION ANALYSIS
            print(f"\n{'-' * 40}")
            print("4. DISTRIBUTION ANALYSIS")
            print(f"{'-' * 40}")

            orig_dist = orig_data['distributions']
            synth_dist = synth_data['distributions'] if synth_data else None

            print("Numeric Distributions:")
            for col, dist_info in orig_dist['numeric_distributions'].items():
                print(f"  {col}:")
                orig_skew = f"{dist_info['skewness']:.3f}"
                orig_kurt = f"{dist_info['kurtosis']:.3f}"
                print(f"    Original - Skewness: {orig_skew}, Kurtosis: {orig_kurt}")

                if synth_dist and col in synth_dist['numeric_distributions']:
                    synth_dist_info = synth_dist['numeric_distributions'][col]
                    synth_skew = f"{synth_dist_info['skewness']:.3f}"
                    synth_kurt = f"{synth_dist_info['kurtosis']:.3f}"
                    print(f"    Synthetic - Skewness: {synth_skew}, Kurtosis: {synth_kurt}")

                    diff_skew = f"{synth_dist_info['skewness'] - dist_info['skewness']:.3f}"
                    diff_kurt = f"{synth_dist_info['kurtosis'] - dist_info['kurtosis']:.3f}"
                    print(f"    Difference - Skewness: {diff_skew}, Kurtosis: {diff_kurt}")

            print("\nCategorical Distributions:")
            for col, dist_info in orig_dist['categorical_distributions'].items():
                print(f"  {col}:")
                orig_entropy = f"{dist_info['entropy']:.3f}" if dist_info['entropy'] else 'N/A'
                print(f"    Original - Unique Values: {dist_info['unique_values']}, Entropy: {orig_entropy}")

                if synth_dist and col in synth_dist['categorical_distributions']:
                    synth_dist_info = synth_dist['categorical_distributions'][col]
                    synth_entropy = f"{synth_dist_info['entropy']:.3f}" if synth_dist_info['entropy'] else 'N/A'
                    print(
                        f"    Synthetic - Unique Values: {synth_dist_info['unique_values']}, Entropy: {synth_entropy}")

            # 5. CORRELATION ANALYSIS
            print(f"\n{'-' * 40}")
            print("5. CORRELATION ANALYSIS")
            print(f"{'-' * 40}")

            orig_corr = orig_data['correlations']
            synth_corr = synth_data['correlations'] if synth_data else None

            if orig_corr['correlation_summary']:
                print("Original Data Correlations:")
                orig_max_corr = f"{orig_corr['correlation_summary']['max_correlation']:.3f}"
                orig_mean_corr = f"{orig_corr['correlation_summary']['mean_correlation']:.3f}"
                print(f"  Max Correlation: {orig_max_corr}")
                print(f"  Mean Correlation: {orig_mean_corr}")
                print(f"  Strong Correlation Pairs: {orig_corr['correlation_summary']['strong_pairs']}")

                if synth_corr and synth_corr['correlation_summary']:
                    print("Synthetic Data Correlations:")
                    synth_max_corr = f"{synth_corr['correlation_summary']['max_correlation']:.3f}"
                    synth_mean_corr = f"{synth_corr['correlation_summary']['mean_correlation']:.3f}"
                    print(f"  Max Correlation: {synth_max_corr}")
                    print(f"  Mean Correlation: {synth_mean_corr}")
                    print(f"  Strong Correlation Pairs: {synth_corr['correlation_summary']['strong_pairs']}")

            # 6. STATISTICAL SIMILARITY (if available)
            if hasattr(self, 'evaluation_results') and self.evaluation_results:
                print(f"\n{'-' * 40}")
                print("6. STATISTICAL SIMILARITY METRICS")
                print(f"{'-' * 40}")

                table_eval = self.evaluation_results.get(table_name, {})
                overall_score = f"{table_eval.get('overall_score', 'N/A'):.4f}" if isinstance(
                    table_eval.get('overall_score'), (int, float)) else 'N/A'
                print(f"Overall Similarity Score: {overall_score}")

                if 'statistical_similarity' in table_eval:
                    stats_sim = table_eval['statistical_similarity']
                    stats_score = f"{stats_sim.get('overall_score', 'N/A'):.4f}" if isinstance(
                        stats_sim.get('overall_score'), (int, float)) else 'N/A'
                    print(f"Statistical Similarity: {stats_score}")

                if 'ml_utility' in table_eval:
                    ml_util = table_eval['ml_utility']
                    ml_score = f"{ml_util.get('overall_score', 'N/A'):.4f}" if isinstance(ml_util.get('overall_score'),
                                                                                          (int, float)) else 'N/A'
                    print(f"ML Utility Score: {ml_score}")

                if 'privacy_risk' in table_eval:
                    privacy = table_eval['privacy_risk']
                    privacy_score = f"{privacy.get('overall_score', 'N/A'):.4f}" if isinstance(
                        privacy.get('overall_score'), (int, float)) else 'N/A'
                    print(f"Privacy Risk Score: {privacy_score}")

    def generate_evaluation_summary(self):
        """Generate and print evaluation summary"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)

        # Evaluate original data
        print("Evaluating original data...")
        original_eval = self.evaluate_original_data()

        # Evaluate synthetic data if available
        synthetic_eval = None
        if self.synthetic_data:
            print("Evaluating synthetic data...")
            synthetic_eval = self.evaluate_synthetic_data()

        # Print the comprehensive report
        self.print_evaluation_report(original_eval, synthetic_eval)

        # Return evaluations for further use
        return original_eval, synthetic_eval

    def identify_age_grouping_options(self):
        """Interactive configuration for age grouping settings"""
        print("\n=== Age Grouping Settings ===")
        print("Do you want to convert age columns into age groups/ranges? (yes/no)")
        user_input = input("> ").strip().lower()

        # FIXED: Renamed boolean flag to avoid conflict with method name
        self.should_apply_age_grouping = user_input.startswith('y')

        if self.should_apply_age_grouping:
            print("\nAge grouping will help protect privacy by converting specific ages to ranges")

            # Find potential age columns
            age_columns = []
            for table_name, df in self.original_data.items():
                for column in df.columns:
                    if ('age' in column.lower() or
                            (pd.api.types.is_numeric_dtype(df[column]) and
                             df[column].min() >= 0 and df[column].max() <= 120)):
                        age_columns.append((table_name, column))

            if not age_columns:
                print("No potential age columns found in your data.")
                self.should_apply_age_grouping = False
                return

            print(f"\nFound {len(age_columns)} potential age column(s):")
            for i, (table_name, column) in enumerate(age_columns):
                sample_values = self.original_data[table_name][column].dropna().sample(
                    min(5, len(self.original_data[table_name]))).tolist()
                print(f"{i + 1}. {table_name}.{column} - Examples: {sample_values}")

            print(
                "\nSelect which columns should be converted to age groups (comma-separated numbers, 'all', or 'none'):")
            selection = input("> ").strip().lower()

            selected_columns = []
            if selection == 'all':
                selected_columns = age_columns
            elif selection != 'none':
                try:
                    selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                    selected_columns = [age_columns[idx] for idx in selected_indices if 0 <= idx < len(age_columns)]
                except:
                    print("Invalid input, no columns selected")

            if not selected_columns:
                self.should_apply_age_grouping = False
                return

            # Configure age grouping method
            print(f"\nSelected {len(selected_columns)} column(s) for age grouping")
            print("Choose age grouping method:")
            print("1. Standard 5-year groups (0-4, 5-9, 10-14, ...)")
            print("2. Standard 10-year groups (0-9, 10-19, 20-29, ...)")
            print("3. Life stage groups (Child, Teen, Young Adult, Adult, Senior)")
            print("4. Custom equal-width groups (specify your own interval)")
            print("5. Custom boundary groups (specify your own boundaries)")

            method_choice = input("> ").strip()

            # Configure grouping parameters
            grouping_config = {}

            try:
                choice = int(method_choice)
                if choice == 1:
                    grouping_config = {
                        'method': 'equal_width',
                        'width': 5,
                        'start': 0,
                        'end': 100
                    }
                elif choice == 2:
                    grouping_config = {
                        'method': 'equal_width',
                        'width': 10,
                        'start': 0,
                        'end': 100
                    }
                elif choice == 3:
                    grouping_config = {
                        'method': 'life_stages',
                        'boundaries': [0, 13, 18, 25, 65, 120],
                        'labels': ['Child (0-12)', 'Teen (13-17)', 'Young Adult (18-24)', 'Adult (25-64)',
                                   'Senior (65+)']
                    }
                elif choice == 4:
                    print("Enter the width for each age group (e.g., 5 for 5-year groups):")
                    width = int(input("> ").strip())
                    print("Enter starting age (default 0):")
                    start = int(input("> ").strip() or "0")
                    print("Enter ending age (default 100):")
                    end = int(input("> ").strip() or "100")

                    grouping_config = {
                        'method': 'equal_width',
                        'width': width,
                        'start': start,
                        'end': end
                    }
                elif choice == 5:
                    print("Enter age boundaries separated by commas (e.g., 0,18,30,50,65,100):")
                    boundaries_input = input("> ").strip()
                    boundaries = [int(x.strip()) for x in boundaries_input.split(',')]
                    boundaries.sort()

                    print("Enter labels for each group separated by commas (optional, leave blank for auto-generated):")
                    labels_input = input("> ").strip()
                    labels = None
                    if labels_input:
                        labels = [x.strip() for x in labels_input.split(',')]

                    grouping_config = {
                        'method': 'custom_boundaries',
                        'boundaries': boundaries,
                        'labels': labels
                    }
                else:
                    print("Invalid choice, using default 10-year groups")
                    grouping_config = {
                        'method': 'equal_width',
                        'width': 10,
                        'start': 0,
                        'end': 100
                    }
            except:
                print("Invalid input, using default 10-year groups")
                grouping_config = {
                    'method': 'equal_width',
                    'width': 10,
                    'start': 0,
                    'end': 100
                }

            # Store configuration in schema
            for table_name, column in selected_columns:
                if table_name not in self.schema:
                    self.schema[table_name] = {'columns': {}}
                if 'columns' not in self.schema[table_name]:
                    self.schema[table_name]['columns'] = {}
                if column not in self.schema[table_name]['columns']:
                    self.schema[table_name]['columns'][column] = {}

                self.schema[table_name]['columns'][column]['age_grouping'] = grouping_config
                print(f"Age grouping configured for {table_name}.{column}")

            print(f"Age grouping will be applied using: {grouping_config}")

            # Show a preview of how the grouping will look
            for table_name, column in selected_columns:
                self.demonstrate_age_grouping(
                    self.original_data[table_name],
                    column,
                    grouping_config
                )
        else:
            print("Age grouping will NOT be applied")

    def apply_age_grouping(self, df, table_name):
        """Apply age grouping to specified columns"""
        # FIXED: Use the correct boolean flag name
        if not hasattr(self, 'should_apply_age_grouping') or not self.should_apply_age_grouping:
            logging.info("Age grouping not enabled, skipping")
            return df

        result_df = df.copy()
        schema_info = self.schema.get(table_name, {}).get('columns', {})

        applied_grouping = False
        for column, info in schema_info.items():
            if column not in df.columns:
                continue

            grouping_config = info.get('age_grouping')
            if not grouping_config:
                continue

            logging.info(f"Applying age grouping to column {column}")

            try:
                # Apply the appropriate grouping method
                method = grouping_config.get('method', 'equal_width')

                if method == 'equal_width':
                    result_df[column] = self._apply_equal_width_grouping(
                        result_df[column],
                        grouping_config.get('width', 10),
                        grouping_config.get('start', 0),
                        grouping_config.get('end', 100)
                    )
                elif method == 'life_stages':
                    result_df[column] = self._apply_life_stage_grouping(
                        result_df[column],
                        grouping_config.get('boundaries'),
                        grouping_config.get('labels')
                    )
                elif method == 'custom_boundaries':
                    result_df[column] = self._apply_custom_boundary_grouping(
                        result_df[column],
                        grouping_config.get('boundaries'),
                        grouping_config.get('labels')
                    )

                logging.info(f"Age grouping successfully applied to {column}")
                applied_grouping = True

            except Exception as e:
                logging.error(f"Error applying age grouping to column {column}: {str(e)}")

        if applied_grouping:
            logging.info(f"Age grouping completed for table {table_name}")
        else:
            logging.info(f"No age grouping configurations found for table {table_name}")

        return result_df

    def _apply_equal_width_grouping(self, series, width, start, end):
        """Apply equal-width age grouping"""
        import pandas as pd

        # Create bins
        bins = list(range(start, end + width, width))
        if bins[-1] < end:
            bins.append(end)

        # Create labels
        labels = []
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:  # Last group
                labels.append(f"{bins[i]}+")
            else:
                labels.append(f"{bins[i]}-{bins[i + 1] - 1}")

        # Apply grouping
        try:
            grouped = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
            return grouped.astype(str)
        except Exception as e:
            logging.error(f"Error in equal width grouping: {str(e)}")
            return series

    def _apply_life_stage_grouping(self, series, boundaries, labels):
        """Apply life stage grouping"""
        import pandas as pd

        if not boundaries or not labels:
            # Default life stages
            boundaries = [0, 13, 18, 25, 65, 120]
            labels = ['Child (0-12)', 'Teen (13-17)', 'Young Adult (18-24)', 'Adult (25-64)', 'Senior (65+)']

        try:
            grouped = pd.cut(series, bins=boundaries, labels=labels, right=False, include_lowest=True)
            return grouped.astype(str)
        except Exception as e:
            logging.error(f"Error in life stage grouping: {str(e)}")
            return series

    def _apply_custom_boundary_grouping(self, series, boundaries, labels):
        """Apply custom boundary grouping"""
        import pandas as pd

        if not boundaries:
            logging.error("No boundaries provided for custom grouping")
            return series

        # Generate labels if not provided
        if not labels:
            labels = []
            for i in range(len(boundaries) - 1):
                if i == len(boundaries) - 2:  # Last group
                    labels.append(f"{boundaries[i]}+")
                else:
                    labels.append(f"{boundaries[i]}-{boundaries[i + 1] - 1}")

        try:
            grouped = pd.cut(series, bins=boundaries, labels=labels, right=False, include_lowest=True)
            return grouped.astype(str)
        except Exception as e:
            logging.error(f"Error in custom boundary grouping: {str(e)}")
            return series

    def demonstrate_age_grouping(self, df, column, grouping_config):
        """Show a preview of how age grouping will look"""
        print(f"\nAge Grouping Preview for column '{column}':")
        print("-" * 40)

        # Show original values
        original_sample = df[column].dropna().sample(min(10, len(df))).sort_values()
        print("Original values (sample):")
        print(original_sample.tolist())

        # Apply grouping to the sample
        method = grouping_config.get('method', 'equal_width')

        if method == 'equal_width':
            grouped_sample = self._apply_equal_width_grouping(
                original_sample,
                grouping_config.get('width', 10),
                grouping_config.get('start', 0),
                grouping_config.get('end', 100)
            )
        elif method == 'life_stages':
            grouped_sample = self._apply_life_stage_grouping(
                original_sample,
                grouping_config.get('boundaries'),
                grouping_config.get('labels')
            )
        elif method == 'custom_boundaries':
            grouped_sample = self._apply_custom_boundary_grouping(
                original_sample,
                grouping_config.get('boundaries'),
                grouping_config.get('labels')
            )

        print("Grouped values:")
        for orig, grouped in zip(original_sample, grouped_sample):
            print(f"  {orig} → {grouped}")

        print(f"\nUnique groups created: {grouped_sample.nunique()}")
        print(f"Group distribution:")
        group_counts = grouped_sample.value_counts()
        for group, count in group_counts.items():
            print(f"  {group}: {count} values")

    def identify_address_synthesis_options(self):
        """Interactive configuration for address synthesis settings"""
        print("\n=== Address Synthesis Settings ===")
        print("Do you want to anonymize/synthesize address columns? (yes/no)")
        user_input = input("> ").strip().lower()

        self.should_apply_address_synthesis = user_input.startswith('y')

        if self.should_apply_address_synthesis:
            print("\nAddress synthesis will help protect privacy by anonymizing location data")

            # Initialize the address synthesizer
            print("Initializing address synthesizer...")
            enable_gis = True
            if not GIS_AVAILABLE:
                print("Note: GIS libraries not available. Install with: pip install geopy usaddress")
                print("Continuing with basic address synthesis...")
                enable_gis = False

            self.address_synthesizer = AddressSynthesizer(enable_gis=enable_gis)

            # Find potential address columns
            address_columns = []
            for table_name, df in self.original_data.items():
                identified_cols = self.address_synthesizer.identify_address_columns(df)
                for col in identified_cols:
                    address_columns.append((table_name, col))

            # Also check for manually identifiable address columns
            for table_name, df in self.original_data.items():
                for column in df.columns:
                    if any(addr_keyword in column.lower() for addr_keyword in
                           ['address', 'addr', 'street', 'location', 'residence']):
                        if (table_name, column) not in address_columns:
                            address_columns.append((table_name, column))

            if not address_columns:
                print("No potential address columns found in your data.")
                self.should_apply_address_synthesis = False
                return

            print(f"\nFound {len(address_columns)} potential address column(s):")
            for i, (table_name, column) in enumerate(address_columns):
                sample_values = self.original_data[table_name][column].dropna().sample(
                    min(3, len(self.original_data[table_name]))).tolist()
                print(f"{i + 1}. {table_name}.{column} - Examples: {sample_values}")

            print("\nSelect which columns should be anonymized (comma-separated numbers, 'all', or 'none'):")
            selection = input("> ").strip().lower()

            selected_columns = []
            if selection == 'all':
                selected_columns = address_columns
            elif selection != 'none':
                try:
                    selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                    selected_columns = [address_columns[idx] for idx in selected_indices if
                                        0 <= idx < len(address_columns)]
                except:
                    print("Invalid input, no columns selected")

            if not selected_columns:
                self.should_apply_address_synthesis = False
                return

            # Configure address anonymization method
            print(f"\nSelected {len(selected_columns)} column(s) for address anonymization")
            print("Choose address anonymization method:")
            print("1. Remove house numbers only (keep street, city, state, zip)")
            print("2. Keep street name only")
            print("3. Keep city and state only")
            print("4. Keep zip code only")
            print("5. Keep general area (city, state, zip)")
            print("6. Generate realistic synthetic addresses")

            method_choice = input("> ").strip()

            # Map choices to method names
            method_mapping = {
                '1': 'remove_house_number',
                '2': 'street_only',
                '3': 'city_state_only',
                '4': 'zip_only',
                '5': 'general_area',
                '6': 'synthesize_realistic'
            }

            method = method_mapping.get(method_choice, 'remove_house_number')
            method_descriptions = {
                'remove_house_number': 'Remove house numbers only',
                'street_only': 'Keep street name only',
                'city_state_only': 'Keep city and state only',
                'zip_only': 'Keep zip code only',
                'general_area': 'Keep general area',
                'synthesize_realistic': 'Generate realistic synthetic addresses'
            }

            print(f"Selected method: {method_descriptions[method]}")

            # Store configuration in schema
            for table_name, column in selected_columns:
                if table_name not in self.schema:
                    self.schema[table_name] = {'columns': {}}
                if 'columns' not in self.schema[table_name]:
                    self.schema[table_name]['columns'] = {}
                if column not in self.schema[table_name]['columns']:
                    self.schema[table_name]['columns'][column] = {}

                self.schema[table_name]['columns'][column]['address_synthesis'] = {
                    'method': method,
                    'enabled': True
                }
                print(f"Address synthesis configured for {table_name}.{column}")

            # Show preview
            print(f"\nAddress anonymization will be applied using: {method_descriptions[method]}")

            # Show a preview of how the anonymization will look
            for table_name, column in selected_columns[:1]:  # Show preview for first column only
                self.demonstrate_address_synthesis(
                    self.original_data[table_name],
                    column,
                    method
                )
        else:
            print("Address synthesis will NOT be applied")

    def demonstrate_address_synthesis(self, df, column, method):
        """Show a preview of how address synthesis will look"""
        print(f"\nAddress Synthesis Preview for column '{column}':")
        print("-" * 50)

        # Show original values
        original_sample = df[column].dropna().head(5)
        print("Original addresses (sample):")
        for addr in original_sample:
            print(f"  {addr}")

        print(f"\nAfter applying '{method}' method:")

        # Apply synthesis to the sample
        synthesizer = getattr(self, 'address_synthesizer', AddressSynthesizer())
        for addr in original_sample:
            anonymized = synthesizer.anonymize_address(str(addr), method)
            print(f"  {addr} → {anonymized}")

    def apply_address_synthesis(self, df, table_name):
        """Apply address synthesis to specified columns"""
        if not hasattr(self, 'should_apply_address_synthesis') or not self.should_apply_address_synthesis:
            logging.info("Address synthesis not enabled, skipping")
            return df

        result_df = df.copy()
        schema_info = self.schema.get(table_name, {}).get('columns', {})

        applied_synthesis = False
        for column, info in schema_info.items():
            if column not in df.columns:
                continue

            synthesis_config = info.get('address_synthesis')
            if not synthesis_config or not synthesis_config.get('enabled'):
                continue

            logging.info(f"Applying address synthesis to column {column}")

            try:
                method = synthesis_config.get('method', 'remove_house_number')

                # Use the address synthesizer
                synthesizer = getattr(self, 'address_synthesizer', AddressSynthesizer())
                result_df[column] = synthesizer.process_address_column(result_df[column], method)

                logging.info(f"Address synthesis successfully applied to {column}")
                applied_synthesis = True

            except Exception as e:
                logging.error(f"Error applying address synthesis to column {column}: {str(e)}")

        if applied_synthesis:
            logging.info(f"Address synthesis completed for table {table_name}")
        else:
            logging.info(f"No address synthesis configurations found for table {table_name}")

        return result_df

    def run_pipeline(self, input_path, output_path, format_type="csv", generation_method="auto",
                     interactive=False, apply_perturbation=False, perturbation_factor=0.2, print_evaluation=True):
        """Run the full pipeline with error handling"""
        try:
            # Start timing
            start_time = time.time()
            logging.info("Starting synthetic data generation pipeline")

            # Store perturbation settings
            self.apply_perturbation = apply_perturbation
            self.perturbation_factor = perturbation_factor


            # Step 1: Load data
            logging.info(f"Loading data from: {input_path}")
            if os.path.isdir(input_path):
                success = self.load_csv_directory(input_path)
            elif input_path.endswith('.csv'):
                table_name = os.path.splitext(os.path.basename(input_path))[0]
                self.original_data[table_name] = pd.read_csv(input_path)
                success = True
            else:
                success = self.connect_database(input_path)

            if not success:
                logging.error("Failed to load data. Aborting pipeline.")
                return False

            # Step 2: Infer schema if not provided
            if not self.schema:
                logging.info("Inferring schema from data")
                self._infer_schema()

            # Default perturbation settings
            self.apply_perturbation = apply_perturbation
            self.perturbation_factor = 0.2  # Default 20% change


            # Step 3: Interactive column identification (if enabled)
            if interactive:
                 self.identify_name_columns()
                 self.identify_abstraction_columns()  # New abstraction step
                 self.identify_columns_to_synthesize()  # New method for choosing columns to synthesize
                 # Add perturbation configuration
                 self.identify_perturbation_options()
                 self.identify_age_grouping_options()  # NEW: Age grouping configuration
                 self.identify_address_synthesis_options()  # NEW: Address synthesis configuration

            # Step 4: Detect relationships (add this before preprocessing)
            logging.info("Detecting data relationships and dependencies")
            self.detect_temporal_relationships()
            self.detect_conditional_dependencies()


            # Step 5: Preprocess data
            logging.info("Preprocessing data")
            self.preprocess_data()

            # Step 6: Analyze data relationships
            logging.info("Analyzing data relationships")
            self.analyze_data()

            # Step 7: Generate synthetic data
            logging.info(f"Generating synthetic data using method: {generation_method}")
            generation_success = self.generate_synthetic_data(method=generation_method)

            if not generation_success:
                logging.error("Synthetic data generation failed")
                return False

            # Make sure we have synthetic data
            if not self.synthetic_data:
                logging.error("No synthetic data was generated")
                return False

            # Check if any tables were generated with data
            has_data = False
            for table_name, df in self.synthetic_data.items():
                if not df.empty:
                    has_data = True
                    logging.info(f"Table {table_name} has {len(df)} rows of synthetic data")
                else:
                    logging.warning(f"Table {table_name} has empty synthetic data")

            if not has_data:
                logging.error("All synthetic data tables are empty")
                return False

            # Step 8: Post-process to enforce relationships
            # After generating synthetic data, add careful post-processing
            logging.info("Post-processing synthetic data to maintain relationships")

            # Keep original data safe
            original_synthetic_data = copy.deepcopy(self.synthetic_data)

            try:
                for table_name in list(self.synthetic_data.keys()):
                    if table_name not in self.synthetic_data or self.synthetic_data[table_name].empty:
                        logging.warning(f"No data to post-process for table {table_name}")
                        continue

                    logging.info(f"Post-processing table: {table_name}")
                    processed_df = self.post_process_synthetic_data(
                        self.synthetic_data[table_name], table_name
                    )

                    # Verify we still have data
                    if processed_df.empty:
                        logging.error(f"Post-processing produced empty DataFrame for {table_name} - using original")
                        self.synthetic_data[table_name] = original_synthetic_data[table_name]
                    else:
                        self.synthetic_data[table_name] = processed_df
            except Exception as e:
                logging.error(f"Post-processing failed: {str(e)}")
                logging.error(traceback.format_exc())
                logging.info("Reverting to original synthetic data")
                self.synthetic_data = original_synthetic_data

            # Add more debug info before export
            for table_name, df in self.synthetic_data.items():
                logging.info(f"Table {table_name} ready for export: {len(df)} rows, {len(df.columns)} columns")
                if df.empty:
                    logging.error(f"Table {table_name} is empty before export!")

            # Step 9: Evaluate synthetic data
            logging.info("Evaluating synthetic data")
            evaluation_results = self.evaluate_synthetic_data()

            # Step 10: Validate synthetic data
            logging.info("Validating synthetic data")
            validation_results = self.validate_synthetic_data()

            # After generating synthetic data and before export
            if print_evaluation:
                logging.info("Generating evaluation report")
                self.generate_evaluation_summary()

            # Validate output path
            if not output_path:
                output_path = "synthetic_data"
                logging.warning(f"No output path specified, using default: {output_path}")

            # Make sure it's an absolute path or relative to current directory
            if not os.path.isabs(output_path):
                output_path = os.path.join(os.getcwd(), output_path)
                logging.info(f"Using absolute output path: {output_path}")

            # Try to create the directory if it's a directory path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    logging.info(f"Created output directory: {output_dir}")
                except Exception as e:
                    logging.warning(f"Could not create output directory: {str(e)}")

            # Step 11: Export synthetic data
            logging.info(f"Exporting synthetic data to {format_type} format: {output_path}")
            export_success = self.export_synthetic_data(format_type, output_path)

            if not export_success:
                logging.error("Failed to export synthetic data.")
                return False

            # Calculate elapsed time and log completion
            elapsed_time = time.time() - start_time
            logging.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")

            return True

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = SyntheticDataPipeline()

    # Run pipeline
    pipeline.run_pipeline(
        input_path="data/input",
        output_path="data/output",
        format_type="csv",
        generation_method="auto"
    )