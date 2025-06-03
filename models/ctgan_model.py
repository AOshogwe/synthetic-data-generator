import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import OneHotEncoder


class CTGANGenerator(nn.Module):
    """Generator network for CTGAN"""

    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super(CTGANGenerator, self).__init__()

        # Build network architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Create sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CTGANDiscriminator(nn.Module):
    """Discriminator network for CTGAN"""

    def __init__(self, input_dim, hidden_dims=[256, 256]):
        super(CTGANDiscriminator, self).__init__()

        # Build network architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        # Create sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CTGAN:
    """
    Conditional Tabular GAN implementation for generating synthetic tabular data
    """

    def __init__(self, schema, embedding_dim=128, generator_dims=[256, 256],
                 discriminator_dims=[256, 256], batch_size=500, epochs=30):
        self.schema = schema
        self.embedding_dim = embedding_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.batch_size = batch_size
        self.epochs = epochs

        # Components to be initialized during fit
        self.generator = None
        self.discriminator = None
        self.transformer = None
        self.output_info = None
        self.categorical_columns = None
        self.categorical_dims = None

    def fit(self, df: pd.DataFrame):
        """Fit CTGAN to the data"""
        # Preprocess data
        data, categorical_columns, transformer, output_info = self._preprocess_data(df)

        # Store preprocessing info
        self.transformer = transformer
        self.output_info = output_info
        self.categorical_columns = categorical_columns

        # Define model dimensions
        data_dim = data.shape[1]
        self.categorical_dims = [info['dim'] for info in output_info]

        # Initialize generator and discriminator
        self.generator = CTGANGenerator(
            input_dim=self.embedding_dim,
            output_dim=data_dim,
            hidden_dims=self.generator_dims
        )

        self.discriminator = CTGANDiscriminator(
            input_dim=data_dim,
            hidden_dims=self.discriminator_dims
        )

        # Set up optimizers
        optimizer_g = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        # Convert to torch tensor
        data_tensor = torch.from_numpy(data.astype('float32'))

        # Training loop
        for epoch in range(self.epochs):
            # Train discriminator
            mean_d_loss = self._train_discriminator(
                data_tensor, optimizer_d, self.batch_size
            )

            # Train generator
            mean_g_loss = self._train_generator(optimizer_g, self.batch_size)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}: "
                      f"D loss: {mean_d_loss:.4f}, G loss: {mean_g_loss:.4f}")

        return self

    def _preprocess_data(self, df: pd.DataFrame):
        """Preprocess data for CTGAN"""
        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Print all column types for debugging
        for col in df_copy.columns:
            print(f"Column '{col}': dtype={df_copy[col].dtype}, sample values={df_copy[col].head(2).tolist()}")

        # Identify categorical and continuous columns based on schema
        categorical_columns = [col for col, info in self.schema.items()
                               if info.get('type', '') == 'categorical' and col in df.columns]

        continuous_columns = [col for col, info in self.schema.items()
                              if info.get('type', '') == 'numeric' and col in df.columns]

        print(f"Initial categorical columns: {categorical_columns}")
        print(f"Initial continuous columns: {continuous_columns}")

        # Force all columns with object dtype to be categorical
        for col in df_copy.columns:
            if col in continuous_columns and (
                    df_copy[col].dtype == 'object' or df_copy[col].apply(lambda x: isinstance(x, str)).any()):
                print(f"Forcing column '{col}' to categorical due to string values")
                categorical_columns.append(col)
                continuous_columns.remove(col)

        print(f"Final categorical columns: {categorical_columns}")
        print(f"Final continuous columns: {continuous_columns}")

        # Handle categorical columns with one-hot encoding
        transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if categorical_columns:
            transformer.fit(df_copy[categorical_columns])

        # Continue with your processing...

        # Prepare output info for reconstruction
        output_info = []
        col_names = []

        # Process categorical columns
        if categorical_columns:
            categorical_data = transformer.transform(df[categorical_columns])
            categorical_dims = [len(categories) for categories in transformer.categories_]

            idx = 0
            for i, dim in enumerate(categorical_dims):
                output_info.append({
                    'type': 'categorical',
                    'dim': dim,
                    'column': categorical_columns[i]
                })
                col_names.extend([f'{categorical_columns[i]}_{j}' for j in range(dim)])
                idx += dim

        # Process continuous columns
        continuous_data = df[continuous_columns].values

        if continuous_columns:
            for i, col in enumerate(continuous_columns):
                output_info.append({
                    'type': 'continuous',
                    'dim': 1,
                    'column': col
                })
                col_names.append(col)

        # Combine data
        if categorical_columns and continuous_columns:
            data = np.column_stack([categorical_data, continuous_data])
        elif categorical_columns:
            data = categorical_data
        else:
            data = continuous_data

        for column in df.columns:
            if self.schema.get(column, {}).get('type', '') == 'numeric':
                df[column] = pd.to_numeric(df[column], errors='coerce')
                df[column] = df[column].fillna(df[column].mean())

        return data, categorical_columns, transformer, output_info

    def _train_discriminator(self, data_tensor, optimizer, batch_size):
        """Train the discriminator for one epoch"""
        self.discriminator.train()
        self.generator.eval()

        data_size = len(data_tensor)
        total_loss = 0

        # Train in batches
        for i in range(0, data_size, batch_size):
            # Get real batch
            batch_end = min(i + batch_size, data_size)
            real_data = data_tensor[i:batch_end]

            # Generate noise
            noise = torch.randn(len(real_data), self.embedding_dim)

            # Generate fake data
            with torch.no_grad():
                fake_data = self.generator(noise)

            # Reset gradients
            optimizer.zero_grad()

            # Real data loss
            real_pred = self.discriminator(real_data)
            real_loss = torch.mean((real_pred - 1) ** 2)

            # Fake data loss
            fake_pred = self.discriminator(fake_data)
            fake_loss = torch.mean(fake_pred ** 2)

            # Combined loss
            loss = (real_loss + fake_loss) / 2
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        return total_loss / (data_size // batch_size + 1)

    def _train_generator(self, optimizer, batch_size):
        """Train the generator for one epoch"""
        self.generator.train()
        self.discriminator.eval()

        total_loss = 0

        # Train in batches
        for _ in range(0, batch_size):
            # Generate noise
            noise = torch.randn(batch_size, self.embedding_dim)

            # Reset gradients
            optimizer.zero_grad()

            # Generate fake data
            fake_data = self.generator(noise)

            # Compute loss
            fake_pred = self.discriminator(fake_data)
            loss = torch.mean((fake_pred - 1) ** 2)
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        return total_loss / batch_size

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples"""
        self.generator.eval()

        # Generate in batches to avoid memory issues
        data = []
        remaining = n_samples

        while remaining > 0:
            current_batch = min(self.batch_size, remaining)

            # Generate noise
            noise = torch.randn(current_batch, self.embedding_dim)

            # Generate fake data
            with torch.no_grad():
                fake_data = self.generator(noise).numpy()

            data.append(fake_data)
            remaining -= current_batch

        # Combine all batches
        data = np.vstack(data)

        # Post-process data back to original format
        return self._inverse_transform(data)

    def _inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Convert generated data back to original format"""
        # Initialize output dataframe
        output = {}

        # Start index for slicing the data array
        st = 0

        # Process each column according to its output info
        for info in self.output_info:
            dim = info['dim']
            column = info['column']

            # Extract relevant slice of data
            column_data = data[:, st:st + dim]

            # Handle categorical columns
            if info['type'] == 'categorical':
                # Convert one-hot back to categories
                if self.categorical_columns and column in self.categorical_columns:
                    # Get the index of the column in categorical_columns
                    col_idx = self.categorical_columns.index(column)

                    # Get the categories for this column
                    categories = self.transformer.categories_[col_idx]

                    # Get indices of maximum values (most probable category)
                    category_indices = np.argmax(column_data, axis=1)

                    # Map indices to category values
                    output[column] = [categories[idx] for idx in category_indices]

            # Handle continuous columns
            elif info['type'] == 'continuous':
                output[column] = column_data.flatten()

            # Move to next column in data array
            st += dim

            # Detect and replace name columns with random names
            for column in output.keys():
                if 'name' in column.lower() and not any(
                        substr in column.lower() for substr in ['filename', 'pathname']):
                    # Generate random names
                    n_samples = len(data)

                    # Check if gender information might be available
                    gender = None
                    if 'gender' in output and 'male' in str(output['gender']).lower():
                        gender = 'M'
                    elif 'gender' in output and 'female' in str(output['gender']).lower():
                        gender = 'F'

                    # Generate random names
                    output[column] = generate_random_names(n_samples, gender)

        return pd.DataFrame(output)