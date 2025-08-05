"""
Author: Yorch233
GitHub: https://github.com/Yorch233/DBLM-SE
Email: qyao@stmail.ujs.edu.cn
Date: 2025-08-05
"""
from typing import Callable

import torch


class Solver:
    """Abstract base class for diffusion model solvers.
    
    Defines the common interface for all solver implementations used in
    reverse-time sampling of diffusion processes.
    """

    def sampling(self, x: torch.Tensor):
        """Main sampling method to be implemented by subclasses.
        
        Args:
            x (torch.Tensor): Initial state tensor to start sampling from.
            
        Returns:
            torch.Tensor: Final sampled state after reverse process.
        """
        raise NotImplementedError()


class SDESolver(Solver):
    """Stochastic Differential Equation (SDE) solver implementation.
    
    Implements the reverse-time sampling process for SDE-based diffusion models.
    Uses numerical integration to solve the reverse SDE and generate samples
    from noise to data distribution.
    
    Args:
        sde (object): Stochastic differential equation object that defines
                     the forward and reverse processes.
        model_fn (Callable): Noise prediction model function that estimates
                           the score or velocity field at each time step.
        device (torch.device): Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        sde: object,
        model_fn: Callable = None,
        device: torch.device = None,
    ):
        """Initialize SDE solver with required components.
        
        Args:
            sde (object): SDE object containing forward/reverse process definitions.
            model_fn (Callable, optional): Function to predict noise/score from model.
            device (torch.device, optional): Computation device override.
        """
        self.sde = sde
        self.model_fn = model_fn
        self.device = device if device is not None else sde.device

    def sampling(
        self, 
        x: torch.Tensor,
        num_steps: int = 5,
        t_max: float = 1,
        t_min: float = 0
    ) -> torch.Tensor:
        """Perform sampling process with time-uniform discretization.
        
        Executes the reverse SDE integration from terminal time t_max to initial time t_min
        using the provided number of sampling steps and time stepping strategy.
        
        Args:
            x (torch.Tensor): Initial state tensor (usually pure noise) to start sampling.
            num_steps (int): Number of sampling steps for the reverse process.
                           Higher values improve sample quality but increase computation time.
            skip_type (str): Time step scheduling strategy. Currently supports "time uniform".
            ot_ode (bool): Optimal transport ODE mode flag. When True, uses deterministic
                          sampling without noise injection for faster but less diverse results.
            t_max (float): Maximum time value for reverse process (default: 1).
            t_min (float): Minimum time value for reverse process (default: 0).
            
        Returns:
            torch.Tensor: Final denoised sample after completing reverse SDE integration.
        """
        # Move input tensor to specified device for computation
        x = x.to(self.device, non_blocking=True)

        # Generate uniform time steps for reverse process
        # Creates num_steps+1 time points from t_max to t_min (inclusive)
        timesteps = torch.linspace(t_max, t_min, num_steps + 1, device=self.device)

        # Execute reverse SDE integration step by step
        for i in range(0, num_steps):
            # Get current and previous time steps
            t = timesteps[i]          # Current time (larger value)
            t_prev = timesteps[i + 1] # Previous time (smaller value)

            # Predict initial state x0 using the model at current time step
            pred_x0 = self.model_fn(x, t)

            # Compute posterior distribution and sample next state
            # This performs one step of reverse SDE integration
            x = self.sde.p_posterior(t_prev, t, x, pred_x0)

        return x
