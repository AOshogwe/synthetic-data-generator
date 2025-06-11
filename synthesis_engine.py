import pandas as pd
import numpy as np
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Complete synthetic data generation engine that integrates multiple approaches
    """

    def __init__(self):
        self.supported_models = [
            "CTGAN", "TVAE", "CopulaGAN", "GaussianCopula",
            "BasicGenerator", "StatisticalGenerator"
        ]

    def generate_synthetic_data(
            self,
            original_data: pd.DataFrame,
            model_type: str = "StatisticalGenerator",
            num_samples: int = 1000,
            privacy_level: str = "medium",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Generate synthetic data using the specified model
        """

        try:
            logger.info(f"Starting synthesis with {model_type}, {num_samples} samples")

            if model_type == "CTGAN":
                return self._generate_with_ctgan(original_data, num_samples, **kwargs)
            elif model_type == "TVAE":
                return self._generate_with_tvae(original_data, num_samples, **kwargs)
            elif model_type == "CopulaGAN":
                return self._generate_with_copula_gan(original_data, num_samples, **kwargs)
            elif model_type == "GaussianCopula":
                return self._generate_with_gaussian_copula(original_data, num_samples, **kwargs)
            elif model_type == "StatisticalGenerator":
                return self._generate_statistical(original_data, num_samples, **kwargs)
            else:
                # Fallback to basic generator
                return self._generate_basic(original_data, num_samples, **kwargs)

        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            logger.error(traceback.format_exc())
            # Always fallback to statistical generator
            logger.info("Falling back to statistical generator")
            return self._generate_statistical(original_data, num_samples, **kwargs)

    def _generate_with_ctgan(self, data: pd.DataFrame, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate using CTGAN model"""
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)

            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=kwargs.get('epochs', 100),  # Reduced for faster generation
                batch_size=kwargs.get('batch_size', 500),
                verbose=False
            )

            logger.info("Fitting CTGAN model...")
            synthesizer.fit(data)

            logger.info(f"Generating {num_samples} synthetic samples...")
            synthetic_data = synthesizer.sample(num_samples)

            return {
                "synthetic_data": synthetic_data,
                "model_type": "CTGAN",
                "num_samples": len(synthetic_data),
                "metadata": {
                    "original_shape": data.shape,
                    "synthetic_shape": synthetic_data.shape,
                    "model_params": {
                        "epochs": kwargs.get('epochs', 100),
                        "batch_size": kwargs.get('batch_size', 500)
                    }
                }
            }

        except ImportError:
            logger.warning("SDV not available, falling back to statistical generator")
            return self._generate_statistical(data, num_samples, **kwargs)
        except Exception as e:
            logger.error(f"CTGAN generation failed: {e}")
            return self._generate_statistical(data, num_samples, **kwargs)

    def _generate_with_tvae(self, data: pd.DataFrame, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate using TVAE model"""
        try:
            from sdv.single_table import TVAESynthesizer
            from sdv.metadata import SingleTableMetadata

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)

            synthesizer = TVAESynthesizer(
                metadata,
                epochs=kwargs.get('epochs', 100),
                batch_size=kwargs.get('batch_size', 500)
            )

            logger.info("Fitting TVAE model...")
            synthesizer.fit(data)

            logger.info(f"Generating {num_samples} synthetic samples...")
            synthetic_data = synthesizer.sample(num_samples)

            return {
                "synthetic_data": synthetic_data,
                "model_type": "TVAE",
                "num_samples": len(synthetic_data),
                "metadata": {
                    "original_shape": data.shape,
                    "synthetic_shape": synthetic_data.shape,
                    "model_params": kwargs
                }
            }

        except ImportError:
            return self._generate_statistical(data, num_samples, **kwargs)
        except Exception as e:
            logger.error(f"TVAE generation failed: {e}")
            return self._generate_statistical(data, num_samples, **kwargs)

    def _generate_with_copula_gan(self, data: pd.DataFrame, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate using CopulaGAN model"""
        try:
            from sdv.single_table import CopulaGANSynthesizer
            from sdv.metadata import SingleTableMetadata

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)

            synthesizer = CopulaGANSynthesizer(metadata)

            logger.info("Fitting CopulaGAN model...")
            synthesizer.fit(data)

            logger.info(f"Generating {num_samples} synthetic samples...")
            synthetic_data = synthesizer.sample(num_samples)

            return {
                "synthetic_data": synthetic_data,
                "model_type": "CopulaGAN",
                "num_samples": len(synthetic_data),
                "metadata": {
                    "original_shape": data.shape,
                    "synthetic_shape": synthetic_data.shape,
                    "model_params": kwargs
                }
            }

        except ImportError:
            return self._generate_statistical(data, num_samples, **kwargs)
        except Exception as e:
            logger.error(f"CopulaGAN generation failed: {e}")
            return self._generate_statistical(data, num_samples, **kwargs)

    def _generate_with_gaussian_copula(self, data: pd.DataFrame, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate using Gaussian Copula model"""
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)

            synthesizer = GaussianCopulaSynthesizer(metadata)

            logger.info("Fitting Gaussian Copula model...")
            synthesizer.fit(data)

            logger.info(f"Generating {num_samples} synthetic samples...")
            synthetic_data = synthesizer.sample(num_samples)

            return {
                "synthetic_data": synthetic_data,
                "model_type": "GaussianCopula",
                "num_samples": len(synthetic_data),
                "metadata": {
                    "original_shape": data.shape,
                    "synthetic_shape": synthetic_data.shape,
                    "model_params": kwargs
                }
            }

        except ImportError:
            return self._generate_statistical(data, num_samples, **kwargs)
        except Exception as e:
            logger.error(f"Gaussian Copula generation failed: {e}")
            return self._generate_statistical(data, num_samples, **kwargs)

    def _generate_statistical(self, data: pd.DataFrame, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate using statistical methods (always works as fallback)"""
        logger.info("Using statistical generation method")

        synthetic_data = pd.DataFrame()

        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Numerical columns - use normal distribution with original mean/std
                mean = data[column].mean()
                std = data[column].std()

                if pd.isna(mean) or pd.isna(std) or std == 0:
                    # Handle edge cases
                    synthetic_values = np.full(num_samples, data[column].iloc[0] if len(data) > 0 else 0)
                else:
                    synthetic_values = np.random.normal(mean, std, num_samples)

                # Ensure integer columns remain integers
                if data[column].dtype == 'int64':
                    synthetic_values = np.round(synthetic_values).astype(int)

                synthetic_data[column] = synthetic_values

            elif data[column].dtype == 'bool':
                # Boolean columns - maintain original probability
                true_prob = data[column].mean()
                synthetic_data[column] = np.random.choice([True, False], num_samples, p=[true_prob, 1 - true_prob])

            else:
                # Categorical columns - sample from original distribution
                value_counts = data[column].value_counts(normalize=True)
                if len(value_counts) > 0:
                    synthetic_data[column] = np.random.choice(
                        value_counts.index,
                        num_samples,
                        p=value_counts.values
                    )
                else:
                    synthetic_data[column] = ['Unknown'] * num_samples

        return {
            "synthetic_data": synthetic_data,
            "model_type": "StatisticalGenerator",
            "num_samples": len(synthetic_data),
            "metadata": {
                "original_shape": data.shape,
                "synthetic_shape": synthetic_data.shape,
                "generation_method": "statistical_sampling",
                "model_params": kwargs
            }
        }

    def _generate_basic(self, data: pd.DataFrame, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Basic generator - simple random sampling with noise"""
        logger.info("Using basic generation method")

        if len(data) == 0:
            raise ValueError("Cannot generate synthetic data from empty dataset")

        # Simple approach: sample with replacement and add noise
        indices = np.random.choice(len(data), num_samples, replace=True)
        synthetic_data = data.iloc[indices].reset_index(drop=True)

        # Add noise to numerical columns
        for column in synthetic_data.columns:
            if synthetic_data[column].dtype in ['int64', 'float64']:
                noise_std = synthetic_data[column].std() * 0.1  # 10% noise
                if not pd.isna(noise_std) and noise_std > 0:
                    noise = np.random.normal(0, noise_std, num_samples)
                    synthetic_data[column] = synthetic_data[column] + noise

                    # Keep integers as integers
                    if data[column].dtype == 'int64':
                        synthetic_data[column] = np.round(synthetic_data[column]).astype(int)

        return {
            "synthetic_data": synthetic_data,
            "model_type": "BasicGenerator",
            "num_samples": len(synthetic_data),
            "metadata": {
                "original_shape": data.shape,
                "synthetic_shape": synthetic_data.shape,
                "generation_method": "sampling_with_noise",
                "model_params": kwargs
            }
        }

    def validate_data_quality(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of generated synthetic data"""

        quality_metrics = {}

        try:
            # Basic shape comparison
            quality_metrics["shape_match"] = original_data.shape[1] == synthetic_data.shape[1]

            # Column-wise quality metrics
            column_metrics = {}

            for column in original_data.columns:
                if column in synthetic_data.columns:
                    col_metrics = {}

                    if original_data[column].dtype in ['int64', 'float64']:
                        # Numerical column metrics
                        orig_mean = original_data[column].mean()
                        synth_mean = synthetic_data[column].mean()
                        orig_std = original_data[column].std()
                        synth_std = synthetic_data[column].std()

                        col_metrics["mean_difference"] = abs(orig_mean - synth_mean) if not pd.isna(
                            orig_mean) and not pd.isna(synth_mean) else float('inf')
                        col_metrics["std_difference"] = abs(orig_std - synth_std) if not pd.isna(
                            orig_std) and not pd.isna(synth_std) else float('inf')
                        col_metrics["data_type"] = "numerical"

                    else:
                        # Categorical column metrics
                        orig_unique = set(original_data[column].unique())
                        synth_unique = set(synthetic_data[column].unique())

                        col_metrics["unique_overlap"] = len(orig_unique.intersection(synth_unique)) / len(
                            orig_unique) if len(orig_unique) > 0 else 0
                        col_metrics["unique_count_original"] = len(orig_unique)
                        col_metrics["unique_count_synthetic"] = len(synth_unique)
                        col_metrics["data_type"] = "categorical"

                    column_metrics[column] = col_metrics

            quality_metrics["column_metrics"] = column_metrics

            # Overall quality score
            scores = []
            for col, metrics in column_metrics.items():
                if metrics["data_type"] == "numerical":
                    mean_score = max(0, 1 - metrics["mean_difference"] / (original_data[col].std() + 1e-6))
                    std_score = max(0, 1 - metrics["std_difference"] / (original_data[col].std() + 1e-6))
                    scores.append((mean_score + std_score) / 2)
                else:
                    scores.append(metrics["unique_overlap"])

            quality_metrics["overall_quality_score"] = np.mean(scores) if scores else 0

        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            quality_metrics["error"] = str(e)
            quality_metrics["overall_quality_score"] = 0

        return quality_metrics