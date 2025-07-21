"""Progress bar utilities with automatic fallback for different environments."""

import sys

# Try to import notebook-based tqdm first, fall back to standard tqdm
try:
    # Check if we're in a Jupyter environment and required dependencies are available
    if ('ipykernel' in sys.modules or 'IPython' in sys.modules) and 'IPython' in sys.modules:
        try:
            # Try importing the notebook version first
            from tqdm.notebook import tqdm
            TQDM_BACKEND = 'notebook'
        except ImportError:
            # If notebook version fails, fall back to standard
            from tqdm import tqdm  # type: ignore
            TQDM_BACKEND = 'standard'
    else:
        # Not in Jupyter, use standard tqdm
        raise ImportError("Not in a Jupyter environment")
except ImportError:
    # Fall back to standard tqdm for non-notebook environments or missing dependencies
    from tqdm import tqdm  # type: ignore
    TQDM_BACKEND = 'standard'

__all__ = ['tqdm', 'TQDM_BACKEND']
