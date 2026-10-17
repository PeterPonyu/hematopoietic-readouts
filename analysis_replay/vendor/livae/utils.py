
import torch


# =============================================================================
# Hyperbolic/Lorentzian Geometry Functions
# =============================================================================

EPS = 1e-6  # Small epsilon for numerical stability


def lorentzian_product(x, y, keepdim=False):
    """
    Compute the Lorentzian inner product between two tensors.
    
    The Lorentzian inner product is defined as: <x,y> = -x₀y₀ + x₁y₁ + ... + xₙyₙ
    where the first coordinate has a negative sign (timelike coordinate).
    
    Args:
        x (torch.Tensor): First tensor in Lorentzian space
        y (torch.Tensor): Second tensor in Lorentzian space  
        keepdim (bool): Whether to keep the reduced dimension
        
    Returns:
        torch.Tensor: Lorentzian inner product result
    """
    # Lorentzian metric: negative first coordinate, positive remaining coordinates
    res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    
    # Clamp to prevent numerical instability
    res = torch.clamp(res, min=-1e6, max=1e6)
    
    if keepdim:
        return res.unsqueeze(-1)
    return res


def lorentz_distance(x, y):
    """
    Compute the Lorentz distance between two points in hyperbolic space.
    
    The Lorentz distance formula: d(x,y) = arccosh(-<x,y>_L)
    where <x,y>_L is the Lorentzian inner product.
    
    Args:
        x (torch.Tensor): First point in Lorentzian space
        y (torch.Tensor): Second point in Lorentzian space
        
    Returns:
        torch.Tensor: Lorentz distance between the points
    """
    # Compute Lorentzian inner product
    xy_inner = lorentzian_product(x, y)
    
    # Clamp argument of arccosh to valid domain [1, ∞)
    clamped = torch.clamp(-xy_inner, min=1.0 + EPS, max=1e6)
    
    return torch.acosh(clamped)


def exp_map_at_origin(v_tangent: torch.Tensor, eps=EPS) -> torch.Tensor:
    """
    Exponential map at the origin of the Lorentz manifold.
    
    Maps tangent vectors at the origin to points on the hyperbolic manifold.
    The exponential map formula: exp₀(v) = (cosh(‖v‖), sinh(‖v‖) * v/‖v‖)
    
    Args:
        v_tangent (torch.Tensor): Tangent vector at origin (first coord should be 0)
        eps (float): Small epsilon for numerical stability
        
    Returns:
        torch.Tensor: Point on the hyperbolic manifold
    """
    # Extract spatial components (ignore time coordinate)
    v_spatial = v_tangent[..., 1:]
    
    # Compute Euclidean norm of spatial components
    v_norm = torch.norm(v_spatial, p=2, dim=-1, keepdim=True)
    
    # Handle zero norm case for numerical stability
    is_zero = v_norm < eps
    v_unit = torch.where(is_zero, torch.zeros_like(v_spatial), v_spatial / v_norm)

    # Apply exponential map formulas
    x_coord = torch.cosh(v_norm)          # Time coordinate: cosh(‖v‖)
    y_coords = torch.sinh(v_norm) * v_unit # Space coordinates: sinh(‖v‖) * v/‖v‖
    
    return torch.cat([x_coord, y_coords], dim=-1)
