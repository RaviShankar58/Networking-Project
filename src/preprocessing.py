import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

def load_milano(path):
    df = pd.read_csv(path)
    print("Loaded dataset shape:", df.shape)
    print("Columns:", df.columns)
    return df

def normalize_traffic(df):
    df = df.copy()
    df["traffic_volume"] = (
        (df["traffic_volume"] - df["traffic_volume"].min())
        / (df["traffic_volume"].max() - df["traffic_volume"].min())
    )
    return df

def select_cells(df, M=49):
    cells = df["cell_id"].unique()[:M]
    df = df[df["cell_id"].isin(cells)]
    print("Selected cells:", len(cells))
    return df


def compute_grid_mapping(traffic_matrix):
    """
    traffic_matrix: shape (T, M)
    Implements Eq.(12) from paper
    """

    # pairwise correlation distances
    corr = np.corrcoef(traffic_matrix.T)
    dist = 1 - corr

    # random 2D initialization
    M = traffic_matrix.shape[1]
    p = np.random.rand(M, 2)

    # simple gradient-free refinement (lightweight)
    for _ in range(200):
        for i in range(M):
            for j in range(M):
                if i != j:
                    diff = np.linalg.norm(p[i] - p[j]) - dist[i, j]
                    p[i] -= 0.001 * diff * (p[i] - p[j])

    # grid assignment (Hungarian)
    grid = np.array([(i, j) for i in range(7) for j in range(7)])
    cost = np.linalg.norm(p[:, None, :] - grid[None, :, :], axis=2)

    row, col = linear_sum_assignment(cost)
    return col
