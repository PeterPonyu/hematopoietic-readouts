
import torch
import tqdm
from anndata import AnnData
from .environment import Env


class agent(Env):
    """
    An agent class for LiVAE (Lorentzian interpretable Variational Autoencoder) architecture.
    
    This class provides a high-level interface for training and using the LiVAE model.
    It extends the Env class with convenient methods for model fitting and embedding
    extraction, making it easy to train the model and obtain latent representations.
    
    The agent manages the complete training workflow including progress monitoring,
    loss tracking, and performance evaluation through multiple clustering metrics.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell expression data.
    layer : str, optional
        The layer of the AnnData object to use for training, by default 'counts'.
    percent : float, optional
        Fraction of data to sample in each training batch (0.0 to 1.0), by default 0.01.
    irecon : float, optional
        Weight for interpretable reconstruction loss regularization, by default 0.0.
    lorentz : float, optional
        Weight for Lorentz distance regularization in hyperbolic space, by default 0.0.
    beta : float, optional
        Beta parameter for KL divergence weighting (β-VAE), by default 1.0.
    dip : float, optional
        Weight for DIP (Disentangled Inferred Prior) regularization, by default 0.0.
    tc : float, optional
        Weight for total correlation penalty (β-TC-VAE), by default 0.0.
    info : float, optional
        Weight for InfoVAE MMD regularization, by default 0.0.
    hidden_dim : int, optional
        Dimension of hidden layers in encoder/decoder networks, by default 128.
    latent_dim : int, optional
        Dimension of the primary latent space, by default 10.
    i_dim : int, optional
        Dimension of interpretable latent representation, by default 2.
    lr : float, optional
        Learning rate for the Adam optimizer, by default 1e-4.
    device : torch.device, optional
        Device to run computations on. Defaults to GPU if available, otherwise CPU.

    Methods
    -------
    fit(epochs=1000)
        Train the model for a specified number of epochs with progress monitoring.
    get_iembed()
        Extract interpretable embedding representations from the trained model.
    get_latent()
        Extract primary latent representations from the trained model.
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        percent: float = .01,
        irecon: float = .0,
        lorentz: float = .0,
        beta: float = 1.,
        dip: float = .0,
        tc: float = .0,
        info: float = .0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        lr: float = 1e-4,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        """
        Initialize the agent with specified model parameters.
        
        All parameters are passed directly to the parent Env class, which handles
        the model initialization and data preprocessing.
        """
        # Initialize parent environment with all specified parameters
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
            irecon=irecon,
            lorentz=lorentz,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            lr=lr,
            device=device
        )
        
    def fit(self, epochs: int = 1000):
        """
        Train the model for a specified number of epochs with progress monitoring.
        
        Performs iterative training with real-time progress visualization showing
        loss values and clustering performance metrics. Updates are displayed every
        10 epochs to balance informativeness with performance.

        Parameters
        ----------
        epochs : int, optional
            Number of training epochs to perform, by default 1000.

        Returns
        -------
        agent
            Returns self to enable method chaining.
            
        Notes
        -----
        Progress bar displays:
        - Loss: Total combined loss value
        - ARI: Adjusted Rand Index for clustering quality
        - NMI: Normalized Mutual Information
        - ASW: Average Silhouette Width
        - C_H: Calinski-Harabasz Index
        - D_B: Davies-Bouldin Index  
        - P_C: Graph connectivity score
        """
        # Initialize progress bar with custom formatting
        with tqdm.tqdm(total=int(epochs), desc='Fitting', ncols=150) as pbar:
            for i in range(int(epochs)):
                # Load a batch of training data
                data = self.load_data()
                
                # Perform one training step (update + evaluation)
                self.step(data)
                
                # Update progress bar with metrics every 10 epochs
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({
                        'Loss': f'{self.loss[-1][0]:.2f}',        # Total loss
                        'ARI': f'{(self.score[-1][0]):.2f}',      # Adjusted Rand Index
                        'NMI': f'{(self.score[-1][1]):.2f}',      # Normalized Mutual Information
                        'ASW': f'{(self.score[-1][2]):.2f}',      # Average Silhouette Width
                        'C_H': f'{(self.score[-1][3]):.2f}',      # Calinski-Harabasz Index
                        'D_B': f'{(self.score[-1][4]):.2f}',      # Davies-Bouldin Index
                        'P_C': f'{(self.score[-1][5]):.2f}'       # Graph connectivity score
                    })
                
                # Advance progress bar by one step
                pbar.update(1)
                
        return self
    
    def get_iembed(self):
        """
        Extract interpretable embedding representations from the trained model.
        
        Computes the interpretable latent representation (i_dim dimensional) by
        performing a forward pass through the complete network and extracting
        the compressed latent encoding component.

        Returns
        -------
        numpy.ndarray
            interpretable embedding matrix of shape (n_observations, i_dim).
            Each row represents the compressed latent representation of one cell.
            
        Notes
        -----
        This method accesses the third-to-last output of the VAE forward pass,
        which corresponds to the latent encoder output (le) in the model architecture.
        The result is detached from the computation graph and converted to NumPy.
        """
        # Convert full dataset to tensor and move to device
        input_tensor = torch.tensor(self.X, dtype=torch.float).to(self.device)
        
        # Forward pass through VAE and extract interpretable embedding (le)
        # Index [-3] corresponds to 'le' in the return tuple:
        # (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_lorentz, ld_lorentz)
        interpretable_embedding = self.nn(input_tensor)[-5]
        
        # Detach from computation graph and convert to NumPy
        return interpretable_embedding.detach().cpu().numpy()
        
    def get_latent(self):
        """
        Extract primary latent representations from the trained model.
        
        Computes the main latent representation (latent_dim dimensional) using
        the take_latent method, which performs encoding through the VAE and
        returns the sampled latent variables.

        Returns
        -------
        numpy.ndarray
            Primary latent representation matrix of shape (n_observations, latent_dim).
            Each row represents the main latent encoding of one cell, corresponding
            to the sampled latent variable q_z from the encoder distribution.
            
        Notes
        -----
        This method uses the take_latent utility which handles the forward pass
        and gradient detachment automatically, making it suitable for inference.
        """
        # Extract latent representations using the inherited take_latent method
        q_z = self.take_latent(self.X)
        return q_z
