import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from models.ctgan_model import CTGAN
from models.tabular_vae import TabularVAEModel
from models.copula import GaussianCopula
from evaluation.statistical_evaluator import SyntheticDataEvaluator


class SyntheticGenerationEngine:
    """
    Orchestrates the training and generation of synthetic data models
    """

    def __init__(self, schema):
        self.schema = schema
        self.models = {}
        self.best_model = None
        self.evaluation_results = None

    def select_best_model(self, df):
        """Select appropriate model based on data characteristics"""
        num_rows = len(df)
        num_cols = len(df.columns)
        has_dates = any(info.get('type') == 'datetime' for col, info in self.schema.items())
        categorical_ratio = sum(1 for col, info in self.schema.items() if info.get('type') == 'categorical') / len(
            self.schema)

        if num_rows < 1000:
            # For small datasets
            return GaussianCopula(self.schema)
        elif categorical_ratio > 0.7:
            # For highly categorical data
            return CTGAN(self.schema)
        elif has_dates:
            # For data with temporal components
            # Note: You would need to implement TimeGAN
            return CTGAN(self.schema)  # Fallback until TimeGAN is implemented
        else:
            # Default for balanced datasets
            return TabularVAEModel(self.schema)

    def train_models(self, df: pd.DataFrame, test_size=0.2):
        """Train multiple synthetic data generation models"""
        # Split data for evaluation
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        # Initialize models
        models = {
            'ctgan': CTGAN(self.schema),
            'tvae': TabularVAEModel(self.schema),
            'copula': GaussianCopula(self.schema)
        }

        # Train each model
        for name, model in models.items():
            print(f"Training {name} model...")
            model.fit(train_data)
            self.models[name] = model

            # Generate samples for evaluation
            synthetic_data = model.sample(len(test_data))

            # Add to model dictionary
            self.models[name] = {
                'model': model,
                'synthetic_data': synthetic_data
            }

        # Evaluate models
        self.evaluation_results = self._evaluate_models(test_data)

        # Select best model
        self.best_model = self._select_best_model()

        return self

    def _evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate the quality of synthetic data from different models"""
        # Initialize evaluator
        evaluator = SyntheticDataEvaluator()

        results = {}
        for name, model_info in self.models.items():
            synthetic_data = model_info['synthetic_data']

            # Calculate evaluation metrics
            stats_similarity = evaluator.evaluate_statistical_similarity(test_data, synthetic_data)
            ml_utility = evaluator.evaluate_ml_utility(test_data, synthetic_data)
            privacy_risk = evaluator.evaluate_privacy_risk(test_data, synthetic_data)

            # Combine metrics
            results[name] = {
                'statistical_similarity': stats_similarity['overall_score'],
                'ml_utility': ml_utility['overall_score'],
                'privacy_risk': privacy_risk['overall_score'],
                'total_score': (
                        0.4 * stats_similarity['overall_score'] +
                        0.4 * ml_utility['overall_score'] +
                        0.2 * (1.0 - privacy_risk['overall_score'])  # Lower privacy risk is better
                ),
                'detailed': {
                    'statistical_similarity': stats_similarity,
                    'ml_utility': ml_utility,
                    'privacy_risk': privacy_risk
                }
            }

        return results

    def _select_best_model(self) -> str:
        """Select the best model based on evaluation results"""
        best_model = None
        best_score = -float('inf')

        for name, results in self.evaluation_results.items():
            if results['total_score'] > best_score:
                best_score = results['total_score']
                best_model = name

        return best_model

    def generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train_models first.")

        # Get best model
        model = self.models[self.best_model]['model']

        # Generate synthetic data
        return model.sample(n_samples)

    def generate_conditional_samples(self, conditions: Dict[str, Any], n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples conditioned on specific column values
        conditions: Dictionary of {column_name: desired_value}
        """
        # This is a simplified implementation of conditional generation
        # For more sophisticated approaches, look into conditional GANs or VAEs

        # Generate more samples than needed to allow for filtering
        oversampling_factor = 10
        samples = self.generate_synthetic_data(n_samples * oversampling_factor)

        # Filter samples based on conditions
        for column, value in conditions.items():
            if column not in samples.columns:
                continue

            # For categorical columns, exact matching
            if self.schema.get(column, {}).get('type') == 'categorical':
                samples = samples[samples[column] == value]

            # For continuous columns, find closest values
            else:
                samples = samples.iloc[(samples[column] - value).abs().argsort()]

        # Return requested number of samples (or fewer if not enough match conditions)
        return samples.head(n_samples)

    def generate_random_names(n_samples, gender=None):
        """
        Generate random names.

        Parameters:
        n_samples: Number of names to generate
        gender: Optional gender specification ('M', 'F', or None for both)

        Returns:
        List of randomly generated names
        """
        # Lists of common first names by gender
        male_first_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
            "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
            "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
            "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon"
        ]

        female_first_names = [
            "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
            "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle",
            "Carol", "Amanda", "Dorothy", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia",
            "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma", "Nicole", "Helen"
        ]

        # List of common last names
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
            "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"
        ]

        # Generate random names
        names = []
        for _ in range(n_samples):
            if gender == 'M':
                first_name = np.random.choice(male_first_names)
            elif gender == 'F':
                first_name = np.random.choice(female_first_names)
            else:
                # Randomly select gender if not specified
                first_name = np.random.choice(male_first_names if np.random.random() < 0.5 else female_first_names)

            last_name = np.random.choice(last_names)
            names.append(f"{first_name} {last_name}")

        return names

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