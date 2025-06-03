import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, f1_score, mean_squared_error


class MLUtilityEvaluator:
    """
    Evaluates machine learning utility of synthetic data
    """

    def __init__(self, schema):
        self.schema = schema

    def evaluate(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate ML utility of synthetic data
        Tests how well models trained on synthetic data perform on original data
        """
        # Initialize results dictionary
        results = {
            'classification_tasks': {},
            'regression_tasks': {},
            'overall_score': None
        }

        # Identify potential target columns
        classification_targets = []
        regression_targets = []

        for column, info in self.schema.items():
            if column not in original_df.columns or column not in synthetic_df.columns:
                continue

            if info['type'] == 'categorical' and original_df[column].nunique() <= 10:
                classification_targets.append(column)
            elif info['type'] == 'numeric':
                regression_targets.append(column)

        # Evaluate classification tasks
        classification_scores = []
        for target in classification_targets:
            task_result = self._evaluate_classification_task(
                original_df, synthetic_df, target
            )

            if task_result is not None:
                results['classification_tasks'][target] = task_result
                classification_scores.append(task_result['utility_score'])

        # Evaluate regression tasks
        regression_scores = []
        for target in regression_targets:
            task_result = self._evaluate_regression_task(
                original_df, synthetic_df, target
            )

            if task_result is not None:
                results['regression_tasks'][target] = task_result
                regression_scores.append(task_result['utility_score'])

        # Calculate overall score
        all_scores = classification_scores + regression_scores
        if all_scores:
            results['overall_score'] = np.mean(all_scores)
        else:
            results['overall_score'] = 0.0

        return results

    def _evaluate_classification_task(self, original_df: pd.DataFrame,
                                      synthetic_df: pd.DataFrame,
                                      target_column: str) -> Dict[str, Any]:
        """Evaluate a classification task"""
        try:
            # Get feature columns (excluding target)
            feature_cols = [col for col in original_df.columns
                            if col != target_column and col in synthetic_df.columns]

            # Prepare original data
            X_orig = pd.get_dummies(original_df[feature_cols], drop_first=True)
            y_orig = original_df[target_column]

            # Split original data
            X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=42
            )

            # Prepare synthetic data
            X_synth = pd.get_dummies(synthetic_df[feature_cols], drop_first=True)
            y_synth = synthetic_df[target_column]

            # Align feature columns between original and synthetic
            common_cols = X_orig.columns.intersection(X_synth.columns)
            X_orig_train = X_orig_train[common_cols]
            X_orig_test = X_orig_test[common_cols]
            X_synth = X_synth[common_cols]

            # Train classifier on original data
            orig_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            orig_clf.fit(X_orig_train, y_orig_train)
            y_orig_pred = orig_clf.predict(X_orig_test)
            orig_accuracy = accuracy_score(y_orig_test, y_orig_pred)

            # Train classifier on synthetic data
            synth_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            synth_clf.fit(X_synth, y_synth)
            y_synth_pred = synth_clf.predict(X_orig_test)
            synth_accuracy = accuracy_score(y_orig_test, y_synth_pred)

            # Calculate F1 scores
            try:
                orig_f1 = f1_score(y_orig_test, y_orig_pred, average='weighted')
                synth_f1 = f1_score(y_orig_test, y_synth_pred, average='weighted')
            except:
                orig_f1 = None
                synth_f1 = None

            # Calculate relative performance (utility score)
            if orig_accuracy > 0:
                relative_performance = synth_accuracy / orig_accuracy
                # Cap at 1.0 (synthetic shouldn't be better than original)
                relative_performance = min(1.0, relative_performance)
            else:
                relative_performance = 0.0

            return {
                'original_accuracy': orig_accuracy,
                'synthetic_accuracy': synth_accuracy,
                'original_f1': orig_f1,
                'synthetic_f1': synth_f1,
                'relative_performance': relative_performance,
                'utility_score': relative_performance
            }

        except:
            # Return None if evaluation fails
            return None

    def _evaluate_regression_task(self, original_df: pd.DataFrame,
                                  synthetic_df: pd.DataFrame,
                                  target_column: str) -> Dict[str, Any]:
        """Evaluate a regression task"""
        try:
            # Get feature columns (excluding target)
            feature_cols = [col for col in original_df.columns
                            if col != target_column and col in synthetic_df.columns]

            # Prepare original data
            X_orig = pd.get_dummies(original_df[feature_cols], drop_first=True)
            y_orig = original_df[target_column]

            # Split original data
            X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=42
            )

            # Prepare synthetic data
            X_synth = pd.get_dummies(synthetic_df[feature_cols], drop_first=True)
            y_synth = synthetic_df[target_column]

            # Align feature columns between original and synthetic
            common_cols = X_orig.columns.intersection(X_synth.columns)
            X_orig_train = X_orig_train[common_cols]
            X_orig_test = X_orig_test[common_cols]
            X_synth = X_synth[common_cols]

            # Train regressor on original data
            orig_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            orig_reg.fit(X_orig_train, y_orig_train)
            y_orig_pred = orig_reg.predict(X_orig_test)
            orig_r2 = r2_score(y_orig_test, y_orig_pred)
            orig_rmse = np.sqrt(mean_squared_error(y_orig_test, y_orig_pred))

            # Train regressor on synthetic data
            synth_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            synth_reg.fit(X_synth, y_synth)
            y_synth_pred = synth_reg.predict(X_orig_test)
            synth_r2 = r2_score(y_orig_test, y_synth_pred)
            synth_rmse = np.sqrt(mean_squared_error(y_orig_test, y_synth_pred))

            # Calculate relative performance (utility score)
            if orig_r2 > 0:
                r2_relative = synth_r2 / orig_r2
                # Cap at 1.0 (synthetic shouldn't be better than original)
                r2_relative = min(1.0, max(0.0, r2_relative))
            else:
                r2_relative = 0.0

            if orig_rmse > 0:
                rmse_relative = orig_rmse / max(synth_rmse, 1e-10)
                # Cap at 1.0
                rmse_relative = min(1.0, max(0.0, rmse_relative))
            else:
                rmse_relative = 0.0

            # Combined utility score
            utility_score = 0.5 * r2_relative + 0.5 * rmse_relative

            return {
                'original_r2': orig_r2,
                'synthetic_r2': synth_r2,
                'original_rmse': orig_rmse,
                'synthetic_rmse': synth_rmse,
                'r2_relative': r2_relative,
                'rmse_relative': rmse_relative,
                'utility_score': utility_score
            }

        except:
            # Return None if evaluation fails
            return None