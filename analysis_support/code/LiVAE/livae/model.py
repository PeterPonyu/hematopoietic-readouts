
import torch
import torch.optim as optim
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE
from .utils import lorentz_distance


class LiVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Lorentzian-Iterpretable Variational Autoencoder with multiple regularization techniques.
    
    This class combines several VAE variants and regularization methods:
    - scVI-style negative binomial reconstruction
    - DIP (Disentangled Inferred Prior) regularization
    - β-TC (Total Correlation) decomposition
    - InfoVAE-style MMD regularization
    - Hyperbolic geometry constraints via Lorentz distance
    - Intermediate reconstruction loss
    
    Inherits from multiple mixin classes that provide specific loss computations.
    """
    
    def __init__(
        self,
        irecon,
        lorentz,
        beta,
        dip,
        tc,
        info,
        state_dim, 
        hidden_dim, 
        latent_dim,
        i_dim,
        lr,
        device,
        *args, 
        **kwargs
    ):
        """
        Initialize the iVAE model with specified regularization weights.
        
        Args:
            irecon (float): Weight for intermediate reconstruction loss
            lorentz (float): Weight for Lorentz distance regularization
            beta (float): Weight for KL divergence (β-VAE parameter)
            dip (float): Weight for DIP regularization
            tc (float): Weight for total correlation penalty
            info (float): Weight for InfoVAE MMD regularization
            state_dim (int): Dimension of input state space
            hidden_dim (int): Dimension of hidden layers
            latent_dim (int): Dimension of primary latent space
            i_dim (int): Dimension of intermediate latent representation
            lr (float): Learning rate for optimizer
            device (str/torch.device): Device to run computations on
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Store regularization weights
        self.irecon = irecon    # Intermediate reconstruction loss weight
        self.lorentz = lorentz  # Hyperbolic geometry regularization weight
        self.beta = beta        # KL divergence weight (β-VAE)
        self.dip = dip         # Disentangled Inferred Prior weight
        self.tc = tc           # Total Correlation weight
        self.info = info       # InfoVAE MMD weight
        
        # Initialize the VAE network and optimizer
        self.nn = VAE(state_dim, hidden_dim, latent_dim, i_dim).to(device)
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        
        # Store device and initialize loss tracking
        self.device = device
        self.loss = []  # List to store loss components for each training step
    
    def take_latent(self, state):
        """
        Extract latent representation from input states.
        
        Performs a forward pass through the encoder to obtain the latent
        representation without gradient computation.
        
        Args:
            state (array-like): Input state data
            
        Returns:
            np.ndarray: Latent representation as numpy array
        """
        # Convert input to tensor and move to device
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        # Forward pass through VAE (only need latent sample q_z)
        q_z, _, _, _, _, _, _, _, _ = self.nn(state)
        
        # Return detached latent representation as numpy array
        return q_z.detach().cpu().numpy()
        
    def update(self, states):
        """
        Perform one training step with multi-component loss.
        
        Computes and optimizes a composite loss function consisting of:
        - Reconstruction loss (negative binomial)
        - Intermediate reconstruction loss
        - Hyperbolic geometry regularization
        - KL divergence
        - DIP regularization
        - Total correlation penalty
        - InfoVAE MMD regularization
        
        Args:
            states (array-like): Batch of input states for training
        """
        # Convert input states to tensor
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        
        # Forward pass through the complete VAE
        q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_lorentz, ld_lorentz = self.nn(states)
       
        # === Primary Reconstruction Loss ===
        # Scale predictions by total counts per cell
        l = states.sum(-1).view(-1, 1)  # Total counts per sample
        pred_x = pred_x * l             # Scale reconstruction by total counts
        
        # Negative binomial reconstruction loss
        disp = torch.exp(self.nn.decoder.disp)  # Dispersion parameter
        recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()
        
        # === Hyperbolic Geometry Regularization ===
        if self.lorentz:
            # Penalize distance between original and compressed representations in hyperbolic space
            lorentz_loss = self.lorentz * lorentz_distance(z_lorentz, ld_lorentz).mean()
        else:
            lorentz_loss = torch.zeros(1).to(self.device)
        
        # === Intermediate Reconstruction Loss ===
        if self.irecon:
            # Reconstruction loss using compressed-decompressed latent
            pred_xl = pred_xl * l  # Scale by total counts
            irecon_loss = -self.irecon * self._log_nb(states, pred_xl, disp).sum(-1).mean()
        else:
            irecon_loss = torch.zeros(1).to(self.device)
        
        # === KL Divergence (β-VAE) ===
        # Standard normal prior
        p_m = torch.zeros_like(q_m)  # Prior mean = 0
        p_s = torch.zeros_like(q_s)  # Prior std = 0 (will be handled in KL computation)
        
        kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()
        
        # === DIP (Disentangled Inferred Prior) Regularization ===
        if self.dip:
            dip_loss = self.dip * self._dip_loss(q_m, q_s)
        else:
            dip_loss = torch.zeros(1).to(self.device)
        
        # === β-TC (Total Correlation) Regularization ===
        if self.tc:
            tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s)
        else:
            tc_loss = torch.zeros(1).to(self.device)
        
        # === InfoVAE MMD Regularization ===
        if self.info:
            # Maximum Mean Discrepancy between latent and standard normal
            mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z))
        else:
            mmd_loss = torch.zeros(1).to(self.device)
        
        # === Total Loss Computation ===
        total_loss = (
            recon_loss + irecon_loss + lorentz_loss + kl_div + 
            dip_loss + tc_loss + mmd_loss
        )
        
        # === Optimization Step ===
        self.nn_optimizer.zero_grad()  # Clear gradients
        total_loss.backward()          # Compute gradients
        self.nn_optimizer.step()       # Update parameters

        # === Loss Tracking ===
        # Store all loss components for monitoring
        self.loss.append((
            total_loss.item(),   # Total combined loss
            recon_loss.item(),   # Primary reconstruction loss
            irecon_loss.item(),  # Intermediate reconstruction loss
            lorentz_loss.item(), # Hyperbolic geometry regularization
            kl_div.item(),       # KL divergence
            dip_loss.item(),     # DIP regularization
            tc_loss.item(),      # Total correlation penalty
            mmd_loss.item()      # InfoVAE MMD regularization
        ))
