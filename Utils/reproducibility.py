"""
Utilities for ensuring reproducibility in experiments.
"""
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional, Union, Dict, Any

def set_seed(seed: int = 42) -> Dict[str, Any]:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        
    Returns:
        Dictionary containing the seed information
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    return {
        'python_seed': random.getstate(),
        'numpy_seed': np.random.get_state(),
        'torch_seed': torch.random.get_rng_state(),
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': True,
        'cudnn_benchmark': False
    }

def get_random_state() -> Dict[str, Any]:
    """
    Get the current random state of all relevant libraries.
    
    Returns:
        Dictionary containing the random states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state

def set_random_state(state: Dict[str, Any]) -> None:
    """
    Set the random state from a previously saved state.
    
    Args:
        state: Dictionary containing the random states
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])

class ReproducibilityContext:
    """Context manager for reproducibility."""
    
    def __init__(self, seed: int = 42, **kwargs):
        """
        Initialize the context manager.
        
        Args:
            seed: Random seed value
            **kwargs: Additional arguments for set_seed
        """
        self.seed = seed
        self.kwargs = kwargs
        self.state = None
    
    def __enter__(self):
        """Save the current random state and set the seed."""
        self.state = get_random_state()
        set_seed(self.seed, **self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the previous random state."""
        if self.state is not None:
            set_random_state(self.state)

def enable_deterministic() -> None:
    """Enable deterministic behavior in PyTorch."""
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def check_reproducibility(
    func: callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    n_runs: int = 3,
    seed: int = 42,
    atol: float = 1e-6,
    rtol: float = 1e-5
) -> bool:
    """
    Check if a function produces the same output across multiple runs.
    
    Args:
        func: Function to test
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        n_runs: Number of runs to perform
        seed: Random seed to use
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        
    Returns:
        True if the function is reproducible, False otherwise
    """
    if kwargs is None:
        kwargs = {}
    
    # Run the function multiple times with the same seed
    results = []
    for _ in range(n_runs):
        with ReproducibilityContext(seed):
            results.append(func(*args, **kwargs))
    
    # Check if all results are equal
    for i in range(1, n_runs):
        if not torch.allclose(
            results[0],
            results[i],
            atol=atol,
            rtol=rtol,
            equal_nan=True
        ):
            return False
    
    return True
