import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

## ITERATIVE METHOD - CODE WRITTEN BY CHANAKYA AND YASHWANTH
def water_filling_bisection(P_total, h, tolerance=1e-5):
    """Standard iterative Bisection method for water-filling."""
    noise_levels = 1.0 / h
    low = np.min(noise_levels)
    high = low + P_total

    while (high - low) > tolerance:
        mid = (low + high) / 2.0
        allocated_power = np.maximum(0, mid - noise_levels)
        power_used = np.sum(allocated_power)
        if power_used > P_total:
            high = mid
        else:
            low = mid

    water_level = (low + high) / 2.0
    optimal_powers = np.maximum(0, water_level - noise_levels)
    return optimal_powers, water_level

## MODIFIED METHOD - CODED BY NIMAL
def water_filling_modified(P_total, h):
    """Modified exact water-filling algorithm (Vectorized, non-iterative)."""
    N = len(h)
    noise_levels = 1.0 / h

    sorted_indices = np.argsort(noise_levels)
    sorted_noise = noise_levels[sorted_indices]

    cumsum_noise = np.cumsum(sorted_noise)
    k_array = np.arange(1, N + 1)

    water_levels = (P_total + cumsum_noise) / k_array
    valid_channels = water_levels > sorted_noise
    active_channels = np.sum(valid_channels)

    water_level = water_levels[active_channels - 1]

    optimal_powers = np.zeros(N)
    active_indices = sorted_indices[:active_channels]
    optimal_powers[active_indices] = water_level - noise_levels[active_indices]

    return optimal_powers, water_level


def calculate_capacity(powers, h):
    """Calculates total Shannon capacity."""
    return np.sum(np.log2(1 + powers * h))

## SIMULATIONS AND PLOTTING - CODED BY PRAFUL AND HIBA
def run_single_simulation(N_channels, P_total, h):
    """Runs a single instance and returns results."""
    powers_bisect, wl_bisect = water_filling_bisection(P_total, h)
    powers_mod, wl_mod = water_filling_modified(P_total, h)
    cap_bisect = calculate_capacity(powers_bisect, h)
    cap_mod = calculate_capacity(powers_mod, h)
    return powers_bisect, wl_bisect, cap_bisect, powers_mod, wl_mod, cap_mod

def run_monte_carlo(N_channels, P_total, num_iterations=100):
    """Runs Monte Carlo simulation and returns timing + capacity results."""
    times_bisect = []
    times_mod = []
    caps_bisect = []
    caps_mod = []

    for _ in range(num_iterations):
        h = np.random.exponential(scale=1.0, size=N_channels)

        start = time.perf_counter()
        p_b, _ = water_filling_bisection(P_total, h)
        times_bisect.append(time.perf_counter() - start)
        caps_bisect.append(calculate_capacity(p_b, h))

        start = time.perf_counter()
        p_m, _ = water_filling_modified(P_total, h)
        times_mod.append(time.perf_counter() - start)
        caps_mod.append(calculate_capacity(p_m, h))

    return (np.array(times_bisect), np.array(times_mod),
            np.array(caps_bisect), np.array(caps_mod))

def run_scaling_experiment(channel_sizes, P_total, num_iterations=50):
    """Tests execution time vs number of subchannels (N)."""
    avg_times_bisect = []
    avg_times_mod = []

    for N in channel_sizes:
        tb_list, tm_list = [], []
        for _ in range(num_iterations):
            h = np.random.exponential(scale=1.0, size=N)

            start = time.perf_counter()
            water_filling_bisection(P_total, h)
            tb_list.append(time.perf_counter() - start)

            start = time.perf_counter()
            water_filling_modified(P_total, h)
            tm_list.append(time.perf_counter() - start)

        avg_times_bisect.append(np.mean(tb_list))
        avg_times_mod.append(np.mean(tm_list))

    return np.array(avg_times_bisect), np.array(avg_times_mod)


COLORS = {
    'bisect': '#E74C3C',
    'mod':    '#2ECC71',
    'water':  '#3498DB',
    'noise':  '#95A5A6',
    'grid':   '#ECF0F1',
    'bg':     '#FAFAFA',
    'text':   '#2C3E50',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': COLORS['bg'],
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': COLORS['grid'],
    'grid.linewidth': 0.8,
    'axes.labelcolor': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'axes.titlepad': 12,
})

def plot_water_filling_concept(h_small, P_total):
    """
    PLOT 1: Water-Filling Concept Illustration
    Shows noise floor (1/h), allocated power, and water level for a small channel set.
    """
    N = len(h_small)
    noise = 1.0 / h_small
    powers, water_level = water_filling_modified(P_total, h_small)
    channel_ids = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.6
    # Noise floor bars
    ax.bar(channel_ids, noise, width=bar_width, color=COLORS['noise'],
           label='Noise Floor (1/hᵢ)', zorder=2, alpha=0.85)
    # Allocated power stacked on top
    ax.bar(channel_ids, powers, width=bar_width, bottom=noise,
           color=COLORS['mod'], label='Allocated Power (Pᵢ*)',
           zorder=2, alpha=0.9)

    # Water level line
    ax.axhline(water_level, color=COLORS['water'], linewidth=2.5,
               linestyle='--', label=f'Water Level (μ = {water_level:.2f})', zorder=3)

    # Annotate zero-power channels
    for i, (p, n) in enumerate(zip(powers, noise)):
        if p < 1e-8:
            ax.text(i + 1, n + 0.05, '✕', ha='center', va='bottom',
                    color=COLORS['bisect'], fontsize=12, fontweight='bold')

    ax.set_xlabel('Subchannel Index', fontsize=12)
    ax.set_ylabel('Power / Noise Level', fontsize=12)
    ax.set_title('Water-Filling Power Allocation — Concept Illustration\n'
                 '(✕ = channel below water level, receives zero power)', fontsize=13)
    ax.set_xticks(channel_ids)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0.3, N + 0.7)

    fig.tight_layout()
    fig.show()
    plt.savefig("concept")

def plot_power_allocation(h, P_total, N_show=64):
    """
    PLOT 2: Power allocation bar chart for N_show channels sorted by channel gain.
    """
    idx = np.argsort(h[:N_show])[::-1]
    h_sorted = h[:N_show][idx]
    powers, water_level = water_filling_modified(P_total, h_sorted)
    noise = 1.0 / h_sorted
    channel_ids = np.arange(1, N_show + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top: channel gains
    axes[0].bar(channel_ids, h_sorted, color=COLORS['water'], alpha=0.75, width=0.8)
    axes[0].set_ylabel('Channel Gain hᵢ', fontsize=11)
    axes[0].set_title(f'Power Allocation Across {N_show} Subchannels (sorted by gain)', fontsize=13)

    # Bottom: allocated power
    axes[1].bar(channel_ids, powers, color=COLORS['mod'],
                alpha=0.85, width=0.8, label='Allocated Power Pᵢ*')
    axes[1].set_xlabel('Subchannel Index (sorted by decreasing gain)', fontsize=11)
    axes[1].set_ylabel('Power Allocated (Pᵢ*)', fontsize=11)
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.show()
    fig.savefig("allocation.png")

def plot_monte_carlo_results(times_bisect, times_mod, caps_bisect, caps_mod):
    """
    PLOT 3: Monte Carlo results — timing distributions + capacity distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Timing histograms
    ax = axes[0]
    bins = 20
    ax.hist(times_bisect * 1e3, bins=bins, color=COLORS['bisect'],
            alpha=0.75, label=f'Bisection (avg={np.mean(times_bisect)*1e3:.3f} ms)', edgecolor='white')
    ax.hist(times_mod * 1e3, bins=bins, color=COLORS['mod'],
            alpha=0.75, label=f'Modified (avg={np.mean(times_mod)*1e3:.3f} ms)', edgecolor='white')
    ax.axvline(np.mean(times_bisect) * 1e3, color=COLORS['bisect'],
               linewidth=2, linestyle='--')
    ax.axvline(np.mean(times_mod) * 1e3, color=COLORS['mod'],
               linewidth=2, linestyle='--')
    ax.set_xlabel('Execution Time (ms)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Execution Time Distribution\n(100 Monte Carlo Iterations)', fontsize=12)
    ax.legend(fontsize=9)

    # Capacity comparison
    ax = axes[1]
    iterations = np.arange(1, len(caps_bisect) + 1)
    ax.plot(iterations, caps_bisect, color=COLORS['bisect'],
            linewidth=1.2, alpha=0.7, label='Bisection')
    ax.plot(iterations, caps_mod, color=COLORS['mod'],
            linewidth=1.2, alpha=0.7, linestyle='--', label='Modified')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Capacity (bits/s/Hz)', fontsize=11)
    ax.set_title('Channel Capacity per Iteration\n(Bisection vs Modified)', fontsize=12)
    ax.legend(fontsize=9)

    speedup = np.mean(times_bisect) / np.mean(times_mod)
    fig.suptitle(f'Monte Carlo Comparison  |  Speedup: {speedup:.2f}×',
                 fontsize=13, fontweight='bold', color=COLORS['text'])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.show()
    fig.savefig("montecarlo.png")

def plot_scaling(channel_sizes, times_bisect, times_mod):
    """
    PLOT 4: Execution time vs N (scaling analysis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear scale
    ax = axes[0]
    ax.plot(channel_sizes, times_bisect * 1e3, 'o-', color=COLORS['bisect'],
            linewidth=2, markersize=7, label='Bisection')
    ax.plot(channel_sizes, times_mod * 1e3, 's-', color=COLORS['mod'],
            linewidth=2, markersize=7, label='Modified Exact')
    ax.set_xlabel('Number of Subchannels (N)', fontsize=11)
    ax.set_ylabel('Avg. Execution Time (ms)', fontsize=11)
    ax.set_title('Execution Time vs N (Linear Scale)', fontsize=12)
    ax.legend(fontsize=10)

    # Log-log scale to show complexity
    ax = axes[1]
    ax.loglog(channel_sizes, times_bisect * 1e3, 'o-', color=COLORS['bisect'],
              linewidth=2, markersize=7, label='Bisection O(N·log(1/ε))')
    ax.loglog(channel_sizes, times_mod * 1e3, 's-', color=COLORS['mod'],
              linewidth=2, markersize=7, label='Modified O(N log N)')
    ax.set_xlabel('Number of Subchannels (N)', fontsize=11)
    ax.set_ylabel('Avg. Execution Time (ms)', fontsize=11)
    ax.set_title('Execution Time vs N (Log-Log Scale)\nSlope indicates complexity class', fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.show()
    fig.savefig("scaling.png")

def plot_speedup_vs_N(channel_sizes, times_bisect, times_mod):
    """
    PLOT 5: Speedup ratio vs N.
    """
    speedups = times_bisect / times_mod

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(channel_sizes, speedups, 'D-', color=COLORS['water'],
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgewidth=2.5)
    ax.axhline(1.0, color=COLORS['bisect'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='No speedup (1×)')
    ax.fill_between(channel_sizes, 1, speedups, alpha=0.15, color=COLORS['water'])
    for x, y in zip(channel_sizes, speedups):
        ax.annotate(f'{y:.1f}×', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9, color=COLORS['text'])
    ax.set_xlabel('Number of Subchannels (N)', fontsize=12)
    ax.set_ylabel('Speedup Factor (×)', fontsize=12)
    ax.set_title('Speedup of Modified Algorithm over Bisection vs N', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.show()
    fig.savefig("speedup.png")


if __name__ == "__main__":
    print("=" * 60)
    print("  WATER-FILLING POWER ALLOCATION — FULL SIMULATION")
    print("=" * 60)

    N = 1024
    P = 50.0
    np.random.seed(42)
    h_fixed = np.random.exponential(scale=1.0, size=N)

    # --- Single Run ---
    print("\n[1] Single Run Validation")
    p_b, wl_b, cap_b, p_m, wl_m, cap_m = run_single_simulation(N, P, h_fixed)
    print(f"    Bisection  — Water Level: {wl_b:.6f}, Capacity: {cap_b:.4f} bits/s/Hz")
    print(f"    Modified   — Water Level: {wl_m:.6f}, Capacity: {cap_m:.4f} bits/s/Hz")
    print(f"    Capacities match: {np.isclose(cap_b, cap_m)}")

    # --- Monte Carlo ---
    print("\n[2] Monte Carlo Simulation (100 iterations, N=1024)")
    t_b, t_m, c_b, c_m = run_monte_carlo(N, P, num_iterations=100)
    speedup = np.mean(t_b) / np.mean(t_m)
    print(f"    Bisection  avg time : {np.mean(t_b)*1e3:.4f} ms ± {np.std(t_b)*1e3:.4f} ms")
    print(f"    Modified   avg time : {np.mean(t_m)*1e3:.4f} ms ± {np.std(t_m)*1e3:.4f} ms")
    print(f"    Speedup             : {speedup:.2f}×")
    print(f"    Avg Capacity (both) : {np.mean(c_b):.4f} bits/s/Hz")

    # --- Scaling ---
    channel_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"\n[3] Scaling Experiment (N = {channel_sizes})")
    st_b, st_m = run_scaling_experiment(channel_sizes, P, num_iterations=50)
    for N_i, tb, tm in zip(channel_sizes, st_b, st_m):
        print(f"    N={N_i:5d}  Bisection: {tb*1e3:.4f} ms  Modified: {tm*1e3:.4f} ms  "
              f"Speedup: {tb/tm:.2f}×")

    # --- Generate All Plots ---
    print("\n[4] Generating Plots...")
   

    # Plot 1: Concept illustration (small N)
    np.random.seed(7)
    h_small = np.random.exponential(scale=1.0, size=10)
    plot_water_filling_concept(h_small, P_total=4.0)

    # Plot 2: Power allocation for first 64 channels
    plot_power_allocation(h_fixed, P, N_show=64)

    # Plot 3: Monte Carlo results
    plot_monte_carlo_results(t_b, t_m, c_b, c_m)

    # Plot 4: Scaling
    plot_scaling(channel_sizes, st_b, st_m)

    # Plot 5: Speedup vs N
    plot_speedup_vs_N(channel_sizes, st_b, st_m)

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Algorithm':<20} {'Avg Time (ms)':<18} {'Avg Capacity':<18} {'Speedup'}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*10}")
    print(f"  {'Bisection':<20} {np.mean(t_b)*1e3:<18.4f} {np.mean(c_b):<18.4f} {'1.00×'}")
    print(f"  {'Modified Exact':<20} {np.mean(t_m)*1e3:<18.4f} {np.mean(c_m):<18.4f} {speedup:.2f}×")
    print("=" * 60)
   
