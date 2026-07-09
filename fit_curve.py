import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def calculate_reconstruction_error(params, x_coords, y_coords):
    # params: [theta, M, X]
    theta, m, x_shift = params
    
    # translate coordinates
    tx = x_coords - x_shift
    ty = y_coords - 42.0
    
    # rotate to decouple t
    t = tx * np.cos(theta) + ty * np.sin(theta)
    v_obs = -tx * np.sin(theta) + ty * np.cos(theta)
    
    # calculate predicted v based on t
    v_pred = np.exp(m * np.abs(t)) * np.sin(0.3 * t)
    
    # return L1 error (Manhattan distance)
    return np.sum(np.abs(v_obs - v_pred))

def main():
    try:
        data_source = pd.read_csv("xy_data.csv")
        x_observed = data_source['x'].values
        y_observed = data_source['y'].values
    except FileNotFoundError:
        return

    parameter_bounds = [
        (0.0, np.radians(50.0)),
        (-0.05, 0.05),
        (0.0, 100.0)
    ]
    
    global_result = differential_evolution(
        calculate_reconstruction_error,
        bounds=parameter_bounds,
        args=(x_observed, y_observed),
        strategy='best1bin',
        maxiter=1000,
        popsize=30,
        tol=1e-8,
        seed=42
    )
    
    if not global_result.success:
        return

    local_result = minimize(
        calculate_reconstruction_error,
        x0=global_result.x,
        args=(x_observed, y_observed),
        bounds=parameter_bounds,
        method='L-BFGS-B',
        options={'ftol': 1e-12, 'gtol': 1e-12}
    )
        
    theta_opt, m_opt, x_opt = local_result.x
    total_l1_error = local_result.fun
    mean_absolute_error = total_l1_error / len(x_observed)
    
    # Calculate inferred t values for the dataset
    t_inferred = (x_observed - x_opt) * np.cos(theta_opt) + (y_observed - 42.0) * np.sin(theta_opt)
    v_observed = -(x_observed - x_opt) * np.sin(theta_opt) + (y_observed - 42.0) * np.cos(theta_opt)
    v_predicted = np.exp(m_opt * np.abs(t_inferred)) * np.sin(0.3 * t_inferred)
    
    # Explicitly calculate L1 distance in standard (X, Y) coordinate space
    x_predicted = t_inferred * np.cos(theta_opt) - np.exp(m_opt * np.abs(t_inferred)) * np.sin(0.3 * t_inferred) * np.sin(theta_opt) + x_opt
    y_predicted = 42.0 + t_inferred * np.sin(theta_opt) + np.exp(m_opt * np.abs(t_inferred)) * np.sin(0.3 * t_inferred) * np.cos(theta_opt)
    
    l1_distance_xy = np.sum(np.abs(x_observed - x_predicted) + np.abs(y_observed - y_predicted))
    mean_l1_xy = l1_distance_xy / len(x_observed)
    
    residual_sum_squares = np.sum((v_observed - v_predicted)**2)
    total_sum_squares = np.sum((v_observed - np.mean(v_observed))**2)
    r2_score = 1.0 - (residual_sum_squares / total_sum_squares)
    residual_std = np.sqrt(residual_sum_squares / len(x_observed))
    
    print("--- Fit Results ---")
    print(f"Theta: {theta_opt:.6f} rad (roughly pi/{(np.pi/theta_opt):.1f})")
    print(f"M:     {m_opt:.6f}")
    print(f"X:     {x_opt:.6f}")
    print(f"Mean L1 Error: {mean_l1_xy:.8f}")
    
    t_grid = np.linspace(t_inferred.min(), t_inferred.max(), 2000)
    x_fitted = t_grid * np.cos(theta_opt) - np.exp(m_opt * np.abs(t_grid)) * np.sin(0.3 * t_grid) * np.sin(theta_opt) + x_opt
    y_fitted = 42.0 + t_grid * np.sin(theta_opt) + np.exp(m_opt * np.abs(t_grid)) * np.sin(0.3 * t_grid) * np.cos(theta_opt)
    
    # Plot the results
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Main plot
    ax.scatter(x_observed, y_observed, s=6, color='#2c3e50', alpha=0.4, label='Empirical Data', zorder=1)
    ax.plot(x_fitted, y_fitted, color='#c0392b', linewidth=2.0, label='Analytical Fit', zorder=2)
    
    ax.set_title('Parametric Trajectory Inversion via Rotational Decoupling', pad=15, fontweight='bold')
    ax.set_xlabel('Transformed X Coordinate $[x_i]$')
    ax.set_ylabel('Transformed Y Coordinate $[y_i]$')
    
    # Despine for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add parameter info to plot
    textstr = f"Theta = {theta_opt:.3f} rad\nM = {m_opt:.2f}\nX = {x_opt:.1f}\nMean L1 Error = {mean_l1_xy:.6f}"
    props = dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7')
    ax.text(0.03, 0.96, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    # Add an inset plot (zoom-in on a complex region to prove precision)
    zoom_x_min, zoom_x_max = 85, 105
    zoom_y_min, zoom_y_max = 57, 70
    
    axins = inset_axes(ax, width="35%", height="35%", loc='lower right', borderpad=3)
    axins.scatter(x_observed, y_observed, s=25, color='#2c3e50', alpha=0.6, zorder=1)
    axins.plot(x_fitted, y_fitted, color='#c0392b', linewidth=2.5, zorder=2)
    axins.set_xlim(zoom_x_min, zoom_x_max)
    axins.set_ylim(zoom_y_min, zoom_y_max)
    axins.set_title('Micro-Precision Fit Validation', fontsize=9, pad=5, fontweight='bold')
    axins.tick_params(axis='both', which='major', labelsize=8)
    axins.grid(True, alpha=0.2)
    
    # Draw box and connecting lines for the inset
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.4, linewidth=1.5)
    
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#bdc3c7')
    plt.savefig("fitted_curve.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
