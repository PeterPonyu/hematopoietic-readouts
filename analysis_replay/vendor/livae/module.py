
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .utils import exp_map_at_origin


def weight_init(m):
    """
    Initialize weights for linear layers using Xavier normal initialization.
    
    Args:
        m: PyTorch module to initialize
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, .01)


class Encoder(nn.Module):
    """
    Variational encoder that maps states to latent distributions.
    
    Encodes input states into mean and standard deviation parameters for a 
    normal distribution in the latent space.
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        Initialize the encoder network.
        
        Args:
            state_dim (int): Dimension of input state
            hidden_dim (int): Dimension of hidden layers
            action_dim (int): Dimension of output action/latent space
        """
        super(Encoder, self).__init__()
        
        # Neural network architecture: state_dim -> hidden_dim -> hidden_dim -> action_dim*2
        self.nn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Output both mean and std parameters
        )
        
        # Initialize network weights
        self.apply(weight_init)

    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            tuple: (q_z, q_m, q_s) where:
                - q_z: Sampled latent variable
                - q_m: Mean of latent distribution  
                - q_s: Standard deviation of latent distribution
        """
        output = self.nn(x)
        
        # Split output into mean and std components
        q_m = output[:, :int(output.shape[-1] / 2)]  # First half: means
        q_s = output[:, int(output.shape[-1] / 2):]  # Second half: std deviations
        
        # Apply softplus to ensure positive standard deviations
        s = F.softplus(q_s) + 1e-6  # Add small epsilon for numerical stability
        
        # Create normal distribution and sample from it
        n = Normal(q_m, s)
        q_z = n.rsample()  # Reparameterized sampling
        
        return q_z, q_m, q_s


class Decoder(nn.Module):
    """
    Decoder that reconstructs states from latent representations.
    
    Maps latent variables back to the original state space with softmax output.
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        Initialize the decoder network.
        
        Args:
            state_dim (int): Dimension of output state
            hidden_dim (int): Dimension of hidden layers
            action_dim (int): Dimension of input action/latent space
        """
        super(Decoder, self).__init__()
        
        # Neural network architecture: action_dim -> hidden_dim -> hidden_dim -> state_dim
        self.nn = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softmax(dim=-1)  # Softmax for probability distribution over states
        )
        
        # Learnable dispersion parameter
        self.disp = nn.Parameter(torch.randn(state_dim))
        
        # Initialize network weights
        self.apply(weight_init)

    def forward(self, x):
        """
        Forward pass through decoder.
        
        Args:
            x (torch.Tensor): Input latent variable tensor
            
        Returns:
            torch.Tensor: Reconstructed state probabilities
        """
        output = self.nn(x)
        return output


class VAE(nn.Module):
    """
    Variational Autoencoder with hyperbolic latent space.
    
    Combines encoder and decoder with additional latent space transformations
    involving hyperbolic geometry through exponential maps.
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim, i_dim):
        """
        Initialize the VAE model.
        
        Args:
            state_dim (int): Dimension of input/output state space
            hidden_dim (int): Dimension of hidden layers in encoder/decoder
            action_dim (int): Dimension of primary latent space
            i_dim (int): Dimension of intermediate latent representation
        """
        super(VAE, self).__init__()
        
        # Main encoder-decoder components
        self.encoder = Encoder(state_dim, hidden_dim, action_dim)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim)
        
        # Additional latent space transformations
        self.latent_encoder = nn.Linear(action_dim, i_dim)    # Compress latent representation
        self.latent_decoder = nn.Linear(i_dim, action_dim)    # Expand back to original size
        
    def forward(self, x):
        """
        Complete forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            tuple: (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_lorentz, ld_lorentz) where:
                - q_z: Sampled latent variable from encoder
                - q_m: Mean of encoder latent distribution
                - q_s: Std of encoder latent distribution
                - pred_x: Reconstructed state from original latent
                - le: Compressed latent encoding
                - ld: Decompressed latent decoding
                - pred_xl: Reconstructed state from compressed-decompressed latent
                - z_lorentz: Hyperbolic representation of original latent
                - ld_lorentz: Hyperbolic representation of compressed-decompressed latent
        """
        # Encode input to latent distribution and sample
        q_z, q_m, q_s = self.encoder(x)
        
        # Transform to hyperbolic space using exponential map
        z_tangent = F.pad(q_z, (1, 0), "constant", 0)  # Add zero padding for hyperbolic embedding
        z_lorentz = exp_map_at_origin(z_tangent)
        
        # Compress and decompress latent representation
        le = self.latent_encoder(q_z)  # Compressed latent encoding
        ld = self.latent_decoder(le)   # Decompressed latent decoding
        
        # Transform decompressed latent to hyperbolic space
        ld_tangent = F.pad(ld, (1, 0), "constant", 0)  # Add zero padding
        ld_lorentz = exp_map_at_origin(ld_tangent)
        
        # Decode both original and compressed-decompressed latents
        pred_x = self.decoder(q_z)  # Reconstruction from original latent
        pred_xl = self.decoder(ld)  # Reconstruction from compressed-decompressed latent
        
        return q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_lorentz, ld_lorentz
