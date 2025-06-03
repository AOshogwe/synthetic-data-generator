import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class PrivacyRiskEvaluator:
    """
    Evaluates privacy risks in synthetic data
    """

    def __init__(self, schema):
        self.schema = schema

    def evaluate(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate privacy risks in synthetic data
        Returns a dictionary of metrics
        """
        # Initialize results dictionary with default values
        results = {
            'distance_to_closest': {'risk_score': 0.0, 'details': {}},
            'membership_disclosure_risk': {'risk_score': 0.0, 'details': {}},
            'attribute_disclosure_risk': {'risk_score': 0.0, 'details': {}},
            'k_anonymity': {'risk_score': 0.0, 'details': {}},
            'overall_score': 0.0
        }

        try:
            # Calculate distance to closest record
            distance_result = self._evaluate_distance_to_closest(original_df, synthetic_df)
            if distance_result is not None:
                results['distance_to_closest'] = distance_result
        except Exception as e:
            print(f"Error in distance evaluation: {str(e)}")

        try:
            # Calculate membership disclosure risk
            membership_result = self._evaluate_membership_disclosure(original_df, synthetic_df)
            if membership_result is not None:
                results['membership_disclosure_risk'] = membership_result
        except Exception as e:
            print(f"Error in membership disclosure evaluation: {str(e)}")

        try:
            # Calculate attribute disclosure risk for sensitive attributes
            attribute_result = self._evaluate_attribute_disclosure(original_df, synthetic_df)
            if attribute_result is not None:
                results['attribute_disclosure_risk'] = attribute_result
        except Exception as e:
            print(f"Error in attribute disclosure evaluation: {str(e)}")

        try:
            # Calculate k-anonymity
            k_anonymity_result = self._evaluate_k_anonymity(synthetic_df)
            if k_anonymity_result is not None:
                results['k_anonymity'] = k_anonymity_result
        except Exception as e:
            print(f"Error in k-anonymity evaluation: {str(e)}")

        # Calculate overall privacy risk score using available metrics
        valid_scores = []
        for metric in ['distance_to_closest', 'membership_disclosure_risk', 'attribute_disclosure_risk', 'k_anonymity']:
            if results[metric] is not None and 'risk_score' in results[metric]:
                valid_scores.append(results[metric]['risk_score'])

        if valid_scores:
            results['overall_score'] = np.mean(valid_scores)
        else:
            results['overall_score'] = 0.0  # Default when no valid scores are available

        return results

    # Robust evaluation function
    def evaluate_privacy_risk(original_df, synthetic_df, schema):
        """Evaluate privacy risks with error handling"""
        try:
            # Ensure proper type handling
            processed_original = original_df.copy()
            processed_synthetic = synthetic_df.copy()

            # Process each column according to its type
            for column, info in schema.items():
                if column not in processed_original.columns or column not in processed_synthetic.columns:
                    continue

                col_type = info.get('type')
                if col_type == 'numeric':
                    processed_original[column] = pd.to_numeric(processed_original[column], errors='coerce')
                    processed_synthetic[column] = pd.to_numeric(processed_synthetic[column], errors='coerce')
                elif col_type == 'categorical':
                    processed_original[column] = processed_original[column].astype(str)
                    processed_synthetic[column] = processed_synthetic[column].astype(str)

            # Calculate attribute disclosure risk
            risk_score = calculate_attribute_disclosure(processed_original, processed_synthetic)

            # Calculate k-anonymity
            k_anonymity = calculate_k_anonymity(processed_synthetic)

            return {
                'attribute_disclosure': risk_score,
                'k_anonymity': k_anonymity,
                'overall_risk': (risk_score + k_anonymity) / 2
            }
        except Exception as e:
            print(f"Error in privacy evaluation: {str(e)}")
            return {
                'attribute_disclosure': None,
                'k_anonymity': None,
                'overall_risk': None,
                'error': str(e)
            }

    def _evaluate_distance_to_closest(self, original_df: pd.DataFrame,
                                      synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate risk based on distance to closest record in original data
        Small distances indicate potential privacy leakage
        """
        # Select common numeric columns for distance calculation
        numeric_cols = [col for col in original_df.columns
                        if pd.api.types.is_numeric_dtype(original_df[col]) and
                        col in synthetic_df.columns]

        if not numeric_cols:
            return {
                'mean_min_distance': None,
                'risk_score': 0.0  # No numeric columns, can't calculate
            }

        # Normalize data for fair distance comparison
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        orig_scaled = scaler.fit_transform(original_df[numeric_cols].fillna(0))
        synth_scaled = scaler.transform(synthetic_df[numeric_cols].fillna(0))

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(orig_scaled)

        # Calculate distances to nearest neighbors
        distances, _ = nn.kneighbors(synth_scaled)
        min_distances = distances.flatten()

        # Calculate mean minimum distance
        mean_min_distance = np.mean(min_distances)

        # Convert to risk score (higher distance = lower risk)
        # Assuming distances follow chi-squared distribution with df=len(numeric_cols)
        df = len(numeric_cols)
        expected_distance = np.sqrt(df)  # Expected distance for independent variables

        # Normalize risk score to [0, 1]
        # Risk is high when distance is much smaller than expected
        normalized_distance = mean_min_distance / expected_distance
        risk_score = np.exp(-normalized_distance)  # Exponential decay

        return {
            'mean_min_distance': mean_min_distance,
            'expected_distance': expected_distance,
            'normalized_distance': normalized_distance,
            'risk_score': risk_score
        }

    def _evaluate_membership_disclosure(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate risk of membership disclosure using a machine learning classifier
        """
        try:
            # Create combined dataset with label indicating original (1) or synthetic (0)
            original_labeled = original_df.copy()
            original_labeled['is_original'] = 1

            synthetic_labeled = synthetic_df.copy()
            synthetic_labeled['is_original'] = 0

            combined_data = pd.concat([original_labeled, synthetic_labeled], axis=0)

            # Identify features to use for the classifier
            # Exclude the target column and any problematic columns
            features = [col for col in combined_data.columns if col != 'is_original']

            # Check if we have any features to work with
            if not features:
                print("Warning: No features available for membership disclosure evaluation")
                return {
                    'risk_score': 0.0,
                    'details': {
                        'accuracy': None,
                        'auc': None,
                        'precision': None,
                        'recall': None
                    }
                }

            # Check if we have any non-empty dummy variables to create
            # For each column, check if it's suitable for encoding
            valid_features = []
            for col in features:
                # Skip columns with all missing values
                if combined_data[col].isna().all():
                    continue

                # If categorical/object column, check if it has multiple values
                if pd.api.types.is_object_dtype(combined_data[col]) or combined_data[col].dtype.name == 'category':
                    if combined_data[col].nunique() > 1:
                        valid_features.append(col)
                else:
                    # Numeric columns are fine
                    valid_features.append(col)

            if not valid_features:
                print("Warning: No valid features for membership disclosure evaluation")
                return {
                    'risk_score': 0.0,
                    'details': {
                        'accuracy': None,
                        'auc': None,
                        'precision': None,
                        'recall': None
                    }
                }

            # Process features for model
            # For categorical features, create dummy variables
            # For numeric features, standardize
            X_processed = pd.DataFrame(index=combined_data.index)

            for col in valid_features:
                if pd.api.types.is_object_dtype(combined_data[col]) or combined_data[col].dtype.name == 'category':
                    # Get dummies for this column individually
                    try:
                        dummies = pd.get_dummies(combined_data[col], prefix=col, drop_first=True)
                        if not dummies.empty:
                            X_processed = pd.concat([X_processed, dummies], axis=1)
                    except Exception as e:
                        print(f"Warning: Could not create dummies for column {col}: {str(e)}")
                else:
                    # For numeric columns, just copy
                    X_processed[col] = combined_data[col]

            # If we still have no features, return safe default
            if X_processed.empty or X_processed.shape[1] == 0:
                print("Warning: No features available after processing")
                return {
                    'risk_score': 0.0,
                    'details': {
                        'accuracy': None,
                        'auc': None,
                        'precision': None,
                        'recall': None
                    }
                }

            # Continue with your existing membership disclosure evaluation
            # ...

        except Exception as e:
            print(f"Error in membership disclosure evaluation: {str(e)}")
            return {
                'risk_score': 0.0,
                'details': {
                    'error': str(e)
                }
            }

    def _evaluate_attribute_disclosure(self, original_df: pd.DataFrame,
                                       synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate attribute disclosure risk by checking how well sensitive attributes
        can be predicted from non-sensitive ones
        """
        # For simplicity, consider all categorical columns as potentially sensitive
        categorical_cols = [col for col, info in self.schema.items()
                            if info['type'] == 'categorical' and
                            col in original_df.columns and
                            col in synthetic_df.columns]

        if not categorical_cols:
            return {
                'average_prediction_accuracy': None,
                'risk_score': 0.0  # No categorical columns, can't calculate
            }

        # Evaluate risk for each sensitive attribute
        prediction_scores = []

        for sensitive_col in categorical_cols:
            # Skip if too many unique values (computationally expensive)
            if original_df[sensitive_col].nunique() > 10:
                continue

            # Use other columns to predict sensitive column
            other_cols = [col for col in original_df.columns
                          if col != sensitive_col and col in synthetic_df.columns]

            # Skip if too few features
            if len(other_cols) < 2:
                continue

            # Prepare data
            try:
                X_orig = pd.get_dummies(original_df[other_cols], drop_first=True)
                y_orig = original_df[sensitive_col]

                X_synth = pd.get_dummies(synthetic_df[other_cols], drop_first=True)
                y_synth = synthetic_df[sensitive_col]

                # Align feature columns
                common_cols = X_orig.columns.intersection(X_synth.columns)
                X_orig = X_orig[common_cols]
                X_synth = X_synth[common_cols]

                # Skip if too few features after alignment
                if X_orig.shape[1] < 2:
                    continue

                # Train classifier on synthetic data
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X_synth, y_synth)

                # Test on original data
                accuracy = clf.score(X_orig, y_orig)

                # Adjust for baseline accuracy (random guessing)
                baseline = 1.0 / original_df[sensitive_col].nunique()
                adjusted_accuracy = (accuracy - baseline) / (1 - baseline)
                adjusted_accuracy = max(0, adjusted_accuracy)  # Ensure non-negative

                prediction_scores.append(adjusted_accuracy)
            except:
                continue

        if not prediction_scores:
            return {
                'average_prediction_accuracy': None,
                'risk_score': 0.0  # Couldn't calculate scores
            }

        # Calculate average prediction accuracy
        avg_accuracy = np.mean(prediction_scores)

        # Convert to risk score (higher accuracy = higher risk)
        risk_score = avg_accuracy

        return {
            'average_prediction_accuracy': avg_accuracy,
            'risk_score': risk_score
        }

    def _evaluate_k_anonymity(self, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate k-anonymity of synthetic data
        k-anonymity measures how many records share the same combination of attributes
        """
        # Use quasi-identifiers (typically demographic information)
        # For simplicity, consider all categorical columns as quasi-identifiers
        quasi_identifiers = [col for col, info in self.schema.items()
                             if info['type'] == 'categorical' and
                             col in synthetic_df.columns]

        if not quasi_identifiers:
            return {
                'k': None,
                'risk_score': 0.0  # No quasi-identifiers, can't calculate
            }

        # Count records with same combination of quasi-identifiers
        try:
            counts = synthetic_df.groupby(quasi_identifiers).size()

            # k is the minimum count
            k = counts.min()

            # Convert to risk score (higher k = lower risk)
            # k=1 means unique combination (high risk), k>=5 is typically considered acceptable
            risk_score = np.exp(-0.5 * k) if k < 5 else 0.0
        except:
            k = None
            risk_score = 0.5  # Default moderate risk if calculation fails

        return {
            'k': k,
            'risk_score': risk_score
        }