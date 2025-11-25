from pathlib import Path

import matplotlib.pyplot as plt


def plot_elbow(inertias, k_range=None, save_path=None):
    """
    Plot and save the elbow curve.
    
    Parameters:
    -----------
    inertias : list or array-like
        Inertia values from k-means clustering at different k values.
    k_range : range or list, optional
        Range of k values corresponding to inertias. Defaults to range(2, 2 + len(inertias)).
    save_path : str or Path, optional
        Path to save the figure. Defaults to "reports/figures/elbow_method.png".
    """
    # Handle empty inertias
    if not inertias:
        raise ValueError("inertias cannot be empty")
    
    # Determine k_range
    if k_range is None:
        k_range = range(2, 2 + len(inertias))
    
    # Determine save path
    if save_path is None:
        save_path = "reports/figures/elbow_method.png"
    
    # Ensure parent directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create and save plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.grid()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
