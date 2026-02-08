# import numpy as np
# import torch

# def create_windows(df, N=6):
#     # pivot = df.pivot(index="time", columns="cell_id", values="traffic_volume")
    
#     pivot = (
#         df.groupby(["time", "cell_id"])["traffic_volume"]
#         .sum()
#         .unstack(fill_value=0)
#     ).sort_index()
    
#     pivot = pivot.sort_index()

#     data = pivot.values
#     X, Y = [], []

#     for i in range(N, len(data)):
#         X.append(data[i-N:i])
#         Y.append(data[i])

#     X = np.array(X)
#     Y = np.array(Y)

#     print("X shape:", X.shape)
#     print("Y shape:", Y.shape)

#     return X, Y



# def to_3d_tensor(X, Y):
#     X = X.reshape(-1, 1, X.shape[1], 7, 7)
#     X = torch.tensor(X, dtype=torch.float32)
#     Y = torch.tensor(Y, dtype=torch.float32)
#     print("3D X shape:", X.shape)
#     return X, Y


import numpy as np
import torch

def create_windows(df, N=6):

    pivot = (
        df.groupby(["time", "cell_id"])["traffic_volume"]
          .sum()
          .unstack(fill_value=0)
          .sort_index()
    )

    data = pivot.values  # shape: (T, M)

    X, Y = [], []
    for t in range(N, len(data)):
        X.append(data[t - N : t])
        Y.append(data[t])

    X = np.array(X)   # (samples, N, 49)
    Y = np.array(Y)   # (samples, 49)

    print("X:", X.shape, "Y:", Y.shape)
    return X, Y


def to_3d_tensor(X, Y):
    #  Convert to 3D traffic image (N,7,7)
    X = X.reshape(-1, 1, X.shape[1], 7, 7)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return X, Y
