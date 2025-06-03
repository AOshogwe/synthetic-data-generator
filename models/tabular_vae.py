import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader, TensorDataset
from models.ctgan_model import CTGAN
from models.copula import GaussianCopula


class TabularVAE(nn.Module):
    """
    Variational Autoencoder for tabular data generation
    """

    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=64,
                 categorical_dims=None, beta=1.0):
        super(TabularVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight

        # Store categorical dimensions for reconstruction
        self.categorical_dims = categorical_dims or []

        # Calculate continuous dimensions
        if categorical_dims:
            self.categorical_start_idx = input_dim - sum(categorical_dims)
        else:
            self.categorical_start_idx = input_dim

        # ENCODER
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Mean and variance for latent distribution
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # DECODER
        # Build decoder layers
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer
        self.final_layer = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x):
        """Encode input to latent representation"""
        # Pass through encoder
        h = self.encoder(x)

        # Get mean and variance
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode latent representation to reconstructed input"""
        # Pass through decoder
        h = self.decoder(z)

        # Get reconstruction
        reconstruction = self.final_layer(h)

        # Split into continuous and categorical parts
        if self.categorical_dims:
            # Continuous part (use as is)
            continuous = reconstruction[:, :self.categorical_start_idx]

            # Categorical part (apply softmax to each categorical variable)
            categorical = []
            start_idx = self.categorical_start_idx

            for dim in self.categorical_dims:
                end_idx = start_idx + dim
                categorical.append(F.softmax(reconstruction[:, start_idx:end_idx], dim=1))
                start_idx = end_idx

            # Combine continuous and categorical parts
            return torch.cat([continuous] + categorical, dim=1)
        else:
            return reconstruction

    def forward(self, x):
        """Forward pass"""
        # Encode
        mu, log_var = self.encode(x)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decode(z)

        return reconstruction, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        """VAE loss function"""
        # Reconstruction loss
        # For continuous variables: MSE
        continuous_loss = F.mse_loss(
            recon_x[:, :self.categorical_start_idx],
            x[:, :self.categorical_start_idx],
            reduction='sum'
        )

        # For categorical variables: Cross-entropy
        categorical_loss = 0
        if self.categorical_dims:
            start_idx = self.categorical_start_idx

            for dim in self.categorical_dims:
                end_idx = start_idx + dim

                # Get target as one-hot indices
                target = x[:, start_idx:end_idx]

                # Get prediction logits
                pred = recon_x[:, start_idx:end_idx]

                # Calculate cross-entropy loss
                categorical_loss += F.binary_cross_entropy(
                    pred, target, reduction='sum'
                )

                start_idx = end_idx

        # Total reconstruction loss
        recon_loss = continuous_loss + categorical_loss

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class TabularVAEModel:
    """
    Wrapper for TabularVAE model with training and sampling functionality
    """

    def __init__(self, schema, hidden_dims=[256, 128], latent_dim=64,
                 batch_size=256, epochs=40, beta=1.0, learning_rate=1e-3):
        self.schema = schema
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.learning_rate = learning_rate

        self.model = None
        self.transformer = None
        self.output_info = None

    def fit(self, df: pd.DataFrame):
        """Fit VAE to the data"""
        # Preprocess data (using the same preprocessing as CTGAN)
        data, categorical_columns, transformer, output_info = self._preprocess_data(df)

        # Store preprocessing info
        self.transformer = transformer
        self.output_info = output_info

        # Calculate categorical dimensions for each categorical variable
        categorical_dims = []
        for info in output_info:
            if info['type'] == 'categorical':
                categorical_dims.append(info['dim'])

        # Initialize model
        self.model = TabularVAE(
            input_dim=data.shape[1],
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            categorical_dims=categorical_dims,
            beta=self.beta
        )

        # Set up optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Convert to torch tensor and create dataloader
        data_tensor = torch.from_numpy(data.astype('float32'))
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            for batch in dataloader:
                x = batch[0]

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                recon_x, mu, log_var = self.model(x)

                # Calculate loss
                loss, recon_loss, kl_loss = self.model.loss_function(recon_x, x, mu, log_var)

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}: "
                      f"Loss: {total_loss / len(dataloader):.4f}, "
                      f"Recon: {total_recon / len(dataloader):.4f}, "
                      f"KL: {total_kl / len(dataloader):.4f}")

        return self

    def _preprocess_data(self, df: pd.DataFrame):
        """Preprocess data for VAE (same as CTGAN preprocessing)"""
        # Use the same preprocessing logic as CTGAN
        ctgan = CTGAN(self.schema)

        for column in df.columns:
            if self.schema.get(column, {}).get('type', '') == 'numeric':
                df[column] = pd.to_numeric(df[column], errors='coerce')
                df[column] = df[column].fillna(df[column].mean())

        return ctgan._preprocess_data(df)

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples"""
        self.model.eval()

        # Generate latent vectors
        z = torch.randn(n_samples, self.latent_dim)

        # Decode latent vectors
        with torch.no_grad():
            samples = self.model.decode(z).numpy()

        # Convert to dataframe using the same logic as CTGAN
        return self._inverse_transform(samples)

    def _inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Convert generated data back to original format"""
        # Use the same inverse transform logic as CTGAN
        ctgan = CTGAN(self.schema)
        ctgan.transformer = self.transformer
        ctgan.output_info = self.output_info

        # Get categorical columns
        ctgan.categorical_columns = [info['column'] for info in self.output_info
                                     if info['type'] == 'categorical']

        return ctgan._inverse_transform(data)