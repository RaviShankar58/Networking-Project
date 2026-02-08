import matplotlib.pyplot as plt

def plot_curve(results, label):
    L_vals = [r[0] for r in results]
    C_vals = [r[1] for r in results]

    plt.plot(L_vals, C_vals, marker="o", label=label)
    plt.xlabel("Local Exit Percentage (L)")
    plt.ylabel("Expected Cost (C)")
    plt.legend()
    plt.grid()
