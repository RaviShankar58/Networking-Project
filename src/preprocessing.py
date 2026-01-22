import pandas as pd

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
