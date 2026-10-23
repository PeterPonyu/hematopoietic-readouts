
import numpy as np
from sklearn.cluster import KMeans
from .model import LiVAE
from .mixin import envMixin


class Env(LiVAE, envMixin):
    """
    Training environment for Lorentzian Variational Autoencoder.
    
    This class provides a training environment wrapper that combines the LiVAE model
    with environment-specific functionality for data management and evaluation.
    It handles data sampling, training steps, and score computation during the
    training process.
    
    Inherits from:
        LiVAE: The lorentzian interpretable variational autoencoder model
        envMixin: Environment-specific utility methods for scoring and evaluation
    """
    
    def __init__(
        self,
        adata,
        layer,
        percent,
        irecon,
        lorentz,
        beta,
        dip,
        tc,
        info,
        hidden_dim,
        latent_dim,
        i_dim,
        lr,
        device,
        *args,
        **kwargs
    ):
        """
        Initialize the training environment.
        
        Args:
            adata: AnnData object containing the dataset
            layer (str): Name of the data layer to use from adata
            percent (float): Fraction of data to use in each batch (0.0 to 1.0)
            irecon (float): Weight for intermediate reconstruction loss
            lorentz (float): Weight for Lorentz distance regularization
            beta (float): Weight for KL divergence (β-VAE parameter)
            dip (float): Weight for DIP regularization
            tc (float): Weight for total correlation penalty
            info (float): Weight for InfoVAE MMD regularization
            hidden_dim (int): Dimension of hidden layers in the network
            latent_dim (int): Dimension of primary latent space
            i_dim (int): Dimension of intermediate latent representation
            lr (float): Learning rate for the optimizer
            device (str/torch.device): Device to run computations on
            *args: Additional positional arguments passed to parent classes
            **kwargs: Additional keyword arguments passed to parent classes
        """
        # Register and preprocess the AnnData object
        self._register_anndata(adata, layer, latent_dim)
        
        # Calculate batch size based on percentage of total observations
        self.batch_size = int(percent * self.n_obs)
        
        # Initialize parent LiVAE model with specified parameters
        super().__init__(
            irecon     = irecon,
            lorentz    = lorentz,
            beta       = beta,
            dip        = dip,
            tc         = tc,
            info       = info,
            state_dim  = self.n_var,      # Input dimension from data
            hidden_dim = hidden_dim, 
            latent_dim = latent_dim,
            i_dim      = i_dim,
            lr         = lr, 
            device     = device
        )
        
        # Initialize score tracking list
        self.score = []
    
    def load_data(self):
        """
        Sample and load a batch of training data.
        
        Randomly samples a subset of the dataset according to the specified
        batch size and stores the indices for potential later use.
        
        Returns:
            np.ndarray: Sampled data batch of shape (batch_size, n_features)
        """
        data, idx = self._sample_data()
        self.idx = idx  # Store indices of sampled data points
        return data
        
    def step(self, data):
        """
        Perform one training step and evaluate performance.
        
        Executes a complete training iteration including:
        1. Model parameter update using the provided data
        2. Latent representation extraction
        3. Performance score computation and storage
        
        Args:
            data (np.ndarray): Training data batch
        """
        # Update model parameters with current batch
        self.update(data)
        
        # Extract latent representations from the updated model
        latent = self.take_latent(data)
        
        # Compute evaluation scores using the latent representations
        score = self._calc_score(latent)
        
        # Store the computed scores for tracking training progress
        self.score.append(score)
    
    def _sample_data(self):
        """
        Randomly sample data points for training batch.
        
        Creates a random permutation of all observation indices and selects
        a subset according to the specified batch size.
        
        Returns:
            tuple: (data, indices) where:
                - data (np.ndarray): Selected data points
                - indices (np.ndarray): Indices of selected data points
        """
        # Create random permutation of all observation indices
        idx = np.random.permutation(self.n_obs)
        
        # Randomly select batch_size indices from the permutation
        idx_ = np.random.choice(idx, self.batch_size)
        
        # Extract corresponding data points
        data = self.X[idx_, :]
        
        return data, idx_

    def _register_anndata(self, adata, layer: str, latent_dim):
        """
        Register and preprocess AnnData object for training.
        
        Extracts the specified data layer, applies log transformation,
        stores data dimensions, and generates initial cluster labels
        using K-means clustering.
        
        Args:
            adata: AnnData object containing the dataset
            layer (str): Name of the data layer to extract
            latent_dim (int): Number of clusters for K-means initialization
            
        Returns:
            None: Modifies instance attributes in-place
        """
        # Extract and log-transform the specified data layer
        # Note: .A converts sparse matrix to dense if necessary

        data = adata.layers[layer]
        if hasattr(data, 'toarray'):
            # Sparse matrix (scipy.sparse)
            data_dense = data.toarray()
        elif hasattr(data, 'A'):
            # Numpy matrix (less common, but handle it)
            data_dense = data.A
        else:
            # Already dense (numpy array or similar)
            data_dense = np.asarray(data)
        
        self.X = np.log1p(data_dense)
        
        # Store dataset dimensions
        self.n_obs = adata.shape[0]  # Number of observations (cells)
        self.n_var = adata.shape[1]  # Number of variables (genes)
        
        # Generate initial cluster labels using K-means
        # This provides ground truth labels for evaluation
        self.labels = KMeans(latent_dim).fit_predict(self.X)
        
        return
