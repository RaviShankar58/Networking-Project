import torch
from torch.utils.data import DataLoader, TensorDataset
from models.ddnn_cnn_lstm import DDNN_CNN_LSTM
from loss import sla_cost

def train(X, Y, epochs=20, w=0.3):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DDNN_CNN_LSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=32,
        shuffle=True
    )

    for epoch in range(epochs):

        total_loss = 0

        for x, d in loader:
            x, d = x.to(device), d.to(device)

            y_L, y_R, _ = model(x)

            loss = (
                w * sla_cost(y_L, d)
                + (1 - w) * sla_cost(y_R, d)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "ddnn_cnn_lstm.pth")

if __name__ == "__main__":
    print("CNN-LSTM DDNN training script ready")