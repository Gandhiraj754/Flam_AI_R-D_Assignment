import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import sys

def main():
    try:
        data_source = pd.read_csv("xy_data.csv")
        x_observed = data_source['x'].values
        y_observed = data_source['y'].values
    except FileNotFoundError:
        print("Error: xy_data.csv not found")
        sys.exit(1)

    theta_true = np.pi / 6
    m_true = 0.03
    x_shift_true = 55.0
    
    t_vals = np.linspace(6, 60, 500)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    ax.scatter(x_observed, y_observed, color='#58a6ff', s=10, alpha=0.3, label='Empirical Data', zorder=1)
    
    line, = ax.plot([], [], color='#ff7b72', linewidth=3, label='Extracted Parametric Curve', zorder=2)
    
    ax.set_xlim(-10, 100)
    ax.set_ylim(-30, 90)
    ax.set_title("Parametric Curve Inversion Simulation", color="white", fontsize=14, pad=15)
    ax.set_xlabel("X Coordinate", color="white")
    ax.set_ylabel("Y Coordinate", color="white")
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(color='#30363d', linestyle='--', alpha=0.5)
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', loc='upper right')
    
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        progress = (i + 1) / 60.0
        current_t_max = 6 + (60 - 6) * progress
        t_current = t_vals[t_vals <= current_t_max]
        
        x_pred = t_current * np.cos(theta_true) - np.exp(m_true * np.abs(t_current)) * np.sin(0.3 * t_current) * np.sin(theta_true) + x_shift_true
        y_pred = 42.0 + t_current * np.sin(theta_true) + np.exp(m_true * np.abs(t_current)) * np.sin(0.3 * t_current) * np.cos(theta_true)
        
        line.set_data(x_pred, y_pred)
        return line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=60, interval=50, blit=True)
    
    print("Generating simulation_demo.webp...")
    ani.save('simulation_demo.webp', writer='pillow', fps=20, dpi=100)
    print("Saved simulation_demo.webp successfully.")

if __name__ == "__main__":
    main()
