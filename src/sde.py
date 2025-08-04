"""
Author: Yorch233
GitHub: https://github.com/Yorch233/DBLM-SE
Email: qyao@stmail.ujs.edu.cn
Date: 2025-08-05
"""

import warnings
from typing import Any, Callable, Dict, List, Union


class Register(dict):
    """
    A custom dictionary subclass designed to register and manage artifacts.
    Artifacts can be registered by type and name, allowing for easy retrieval.
    """

    def __init__(self, types_list: List[str] = None) -> None:
        """
        Initializes the ArtifactRegister object. Optionally initializes with a list of types.

        Parameters:
            types_list (List[str], optional): A list of types to initialize the register with. Defaults to None.
        """
        super().__init__()
        self._dict: Dict[str, Dict[str, Any]] = {}

        if types_list is not None:
            for artifact_type in types_list:
                self._dict[artifact_type] = {}

    def register(self, artifact_name: str) -> Callable[[Any], Any]:
        """
        Decorator method to register an artifact by its type and name.

        Parameters:
            artifact_name (str): The name of the artifact.

        Returns:
            Callable[[Any], Any]: A decorator function.
        """

        def decorator(artifact: Any) -> Any:
            self._dict[artifact_name] = artifact
            return artifact

        return decorator

    def fetch(self, name_or_name_list: Union[str, List[str]]) -> Any:
        """
        Retrieves a registered artifact by its type and name.

        Parameters:
            artifact_type (str): The type of the artifact.
            artifact_name (str): The name of the artifact.

        Returns:
            Any: The registered artifact.

        Raises:
            KeyError: If the artifact type or name is not registered.
        """
        if isinstance(name_or_name_list, list):
            artifacts = {}
            for name in name_or_name_list:
                if name in self._dict:
                    artifacts[name] = self._dict[name]
                else:
                    warnings.warn(
                        f"Unregistered artifact_name: '{name}'. Ignoring this artifact."
                    )
            if len(artifacts) == 0:
                raise ValueError("No registered artifacts found.")
            return artifacts
        elif isinstance(name_or_name_list, str):
            if name_or_name_list in self._dict:
                return self._dict[name_or_name_list]
            else:
                raise KeyError(
                    f"Unregistered artifact_name: '{name_or_name_list}'.")


import torch

from src.solver import SDESolver


def unsqueeze_xdim(z, xdim):
    """
    Add singleton dimensions to the tensor `z` to match the length of `xdim`.
    
    This utility function is used to broadcast scalar/time-dependent parameters
    across the spatial dimensions of tensors during SDE computations.

    Args:
        z (torch.Tensor): The input tensor to be unsqueezed.
        xdim (tuple): The target dimensions to be unsqueezed (spatial dimensions).

    Returns:
        torch.Tensor: The unsqueezed tensor with added singleton dimensions.
    """
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


class DDBM_SDE:
    """
    Abstract base class for Diffusion Bridge SDE implementations.
    
    Defines the core interface and common methods for Stochastic Differential Equations
    used in Diffusion Bridge modeling. Subclasses must implement specific marginal
    distribution functions for different SDE types.
    """

    def __init__(self, beta=0.1, t_min=3e-2, t_max=1, device="cpu"):
        """
        Initialize SDE parameters.
        
        Args:
            beta (float): Diffusion coefficient that controls the noise level.
            t_min (float): Minimum time value to prevent numerical instability.
            t_max (float): Maximum time value defining the diffusion horizon.
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.device = device
        self.beta = torch.tensor(beta).to(self.device)
        self.t_min = t_min
        self.t_max = t_max

    def marginal_log_alpha(self, t):
        """
        Compute log of alpha(t) marginal distribution (abstract method).
        
        Alpha(t) represents the signal coefficient in the forward SDE.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Log of alpha values at given times.
        """
        raise NotImplementedError()

    def marginal_alpha(self, t):
        """
        Compute alpha(t) marginal distribution.
        
        Alpha(t) = exp(log_alpha(t)) represents the signal preservation factor.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Alpha values at given times.
        """
        return torch.exp(self.marginal_log_alpha(t))

    def marginal_log_sigma(self, t):
        """
        Compute log of sigma(t) marginal distribution (abstract method).
        
        Sigma(t) represents the noise coefficient in the forward SDE.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Log of sigma values at given times.
        """
        raise NotImplementedError()

    def marginal_sigma(self, t):
        """
        Compute sigma(t) marginal distribution.
        
        Sigma(t) = exp(log_sigma(t)) represents the noise amplification factor.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Sigma values at given times.
        """
        return torch.exp(self.marginal_log_sigma(t))

    def marginal_lambda(self, t):
        """
        Compute lambda(t) marginal distribution.
        
        Lambda(t) = log_alpha(t) - log_sigma(t) is the log-SNR ratio.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Lambda values at given times.
        """
        return self.marginal_log_alpha(t) - self.marginal_log_sigma(t)

    def marginal_logSNR(self, t):
        """
        Compute log of SNR(t) marginal distribution.
        
        Signal-to-Noise Ratio: SNR(t) = exp(2 * lambda(t)).
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Log of SNR values at given times.
        """
        return 2 * self.marginal_lambda(t)

    def marginal_SNR(self, t):
        """
        Compute SNR(t) marginal distribution.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: SNR values at given times.
        """
        return torch.exp(self.marginal_logSNR(t))

    def h(self, s, t):
        """
        Compute h(s, t) marginal distribution.
        
        H(s,t) = lambda(s) - lambda(t) represents the log-SNR difference.
        
        Args:
            s (torch.Tensor): Previous time steps.
            t (torch.Tensor): Current time steps.
            
        Returns:
            torch.Tensor: H values for the time interval.
        """
        return self.marginal_lambda(s) - self.marginal_lambda(t)

    def q_sample(self, t, x0, x1):
        """
        Sampling from the conditional distribution q(x_t|x_0,x_1).
        
        This method samples intermediate states in the bridge process between
        initial state x0 and terminal state x1 at time t.
        
        Args:
            t (torch.Tensor): Time steps for sampling.
            x0 (torch.Tensor): Initial condition tensor (clean data).
            x1 (torch.Tensor): Terminal condition tensor (noisy data).
            
        Returns:
            torch.Tensor: Sampled x_t values at intermediate time steps.
        """
        batch, *xdim = x0.shape

        # Compute bridge interpolation coefficients
        m = torch.exp(2.0 * self.h(torch.ones_like(t, device=self.device) * self.t_max, t))
        mu_xT = m * self.marginal_alpha(t) / self.marginal_alpha(
            torch.ones_like(t, device=self.device) * self.t_max)
        mu_x0 = (1 - m) * self.marginal_alpha(t)
        var = self.marginal_sigma(t)**2 * (1 - m)

        # Broadcast coefficients to match tensor dimensions
        mu_x0 = unsqueeze_xdim(mu_x0, xdim)
        mu_xT = unsqueeze_xdim(mu_xT, xdim)
        var = unsqueeze_xdim(var, xdim)

        # Compute mean of the conditional distribution
        mean = mu_xT * x1 + mu_x0 * x0

        x_t = mean + var.sqrt() * torch.randn_like(mean)
        return x_t

    def p_posterior(self, t, s, x, x0):
        """
        Compute posterior distribution p(x_s|x_t,x_0) using analytical solution.
        
        This method computes the reverse transition from time t to s < t,
        conditioned on the initial state x0.
        
        Args:
            t (torch.Tensor): Current time steps (t > s).
            s (torch.Tensor): Previous time steps.
            x (torch.Tensor): Current state tensor at time t.
            x0 (torch.Tensor): Initial condition tensor.
            
        Returns:
            torch.Tensor: Sampled previous state x_s according to posterior distribution.
        """
        # Compute reverse bridge interpolation coefficients
        m = torch.exp(2.0 * self.h(s, t))
        mu_xt = m * self.marginal_alpha(t) / self.marginal_alpha(s)
        mu_x0 = (1 - m) * self.marginal_alpha(t)

        batch, *xdim = x0.shape

        # Broadcast coefficients to match tensor dimensions
        mu_x0 = unsqueeze_xdim(mu_x0, xdim)
        mu_xt = unsqueeze_xdim(mu_xt, xdim)

        # Compute mean of the posterior distribution
        mean = mu_x0 * x0 + mu_xt * x

        xt_prev = mean

        # Add noise
        if t > self.t_min:
            var = self.marginal_sigma(t)**2 * (1 - m)
            var = unsqueeze_xdim(var, xdim)
            xt_prev += var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def compute_pred_x0(self, t, xt, net_out):
        """
        Compute predicted initial state x0 from network output.
        
        This method uses the network prediction to estimate the clean signal
        at the beginning of the diffusion process.
        
        Args:
            t (torch.Tensor): Current time steps.
            xt (torch.Tensor): Noisy state at time t.
            net_out (torch.Tensor): Network output (typically velocity or score).
            
        Returns:
            torch.Tensor: Predicted initial state x0.
        """
        alpha_t = self.marginal_alpha(t)
        sigma_t = self.marginal_sigma(t)

        batch, *xdim = xt.shape
        alpha_t = unsqueeze_xdim(alpha_t, xdim)
        sigma_t = unsqueeze_xdim(sigma_t, xdim)

        # Estimate x0 using the analytical relationship
        pred_x0 = (xt - sigma_t * net_out) / alpha_t
        return pred_x0

    def get_sde_solver(self, model_fn):
        """
        Create SDE solver instance for reverse sampling.
        
        Args:
            model_fn (callable): Neural network function that predicts the score/velocity.
            
        Returns:
            SDESolver: Configured SDE solver instance for sampling.
        """
        return SDESolver(self, model_fn=model_fn, device=self.device)


class DDBM_VPSDE(DDBM_SDE):
    """
    Variance Preserving SDE implementation.
    
    Implements the specific parameterization for VP-SDE where:
    - Alpha(t) decays exponentially with time to preserve signal
    - Sigma(t) maintains stable variance throughout the diffusion process
    - The sum of squared alpha and sigma equals 1 (variance preservation)
    """

    def __init__(self, beta=0.1, t_min=3e-2, t_max=1, device="cpu"):
        """
        Initialize VP-SDE parameters.
        
        Args:
            beta (float): Controls the rate of alpha decay (β in formulas).
            t_min (float): Minimum time value (clipping threshold to prevent numerical issues).
            t_max (float): Maximum time value (diffusion horizon).
            device (str): Computation device ('cpu' or 'cuda').
        """
        super().__init__(
            beta=beta,
            t_min=t_min,
            t_max=t_max,
            device=device,
        )

    def marginal_log_alpha(self, t):
        """
        Compute log-alpha(t) for VP-SDE: log(α(t)) = -0.5 * β * t.
        
        This represents exponential signal decay in the forward process.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Log of alpha values at given times.
        """
        return -0.5 * t * self.beta

    def marginal_log_sigma(self, t):
        """
        Compute log-sigma(t) for VP-SDE: log(σ(t)) = 0.5*log(1 - exp(-β*t)).
        
        Ensures σ²(t) + α²(t) = 1 to preserve variance throughout the process.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Log of sigma values at given times.
        """
        return 0.5 * torch.log(1.0 - torch.exp(2.0 * self.marginal_log_alpha(t)))


class DDBM_VESDE(DDBM_SDE):
    """
    Variance Exploding SDE implementation.
    
    Implements the specific parameterization for VE-SDE where:
    - Alpha(t) remains constant (α(t)=1) - no signal decay
    - Sigma(t) grows exponentially with time - noise increases over time
    """

    def __init__(self, beta=0.1, t_max=1, device="cpu"):
        """
        Initialize VE-SDE parameters.
        
        Args:
            beta (float): Scaling factor for sigma growth (controls noise explosion rate).
            t_max (float): Maximum time value (diffusion horizon).
            device (str): Computation device ('cpu' or 'cuda').
        """
        super().__init__(beta=beta, t_max=t_max, device=device)

    def marginal_log_alpha(self, t):
        """
        Compute log-alpha(t) for VE-SDE: log(α(t)) = 0 (α(t)=1).
        
        Signal strength remains constant throughout the process.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Zero tensor matching input shape.
        """
        return torch.zeros_like(t, device=self.device)

    def marginal_log_sigma(self, t):
        """
        Compute log-sigma(t) for VE-SDE: log(σ(t)) = log(t).
        
        Results in σ²(t) = t² variance growth - noise increases quadratically.
        
        Args:
            t (torch.Tensor): Time steps tensor.
            
        Returns:
            torch.Tensor: Logarithm of time values.
        """
        return torch.log(t)