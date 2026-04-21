# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from utils import *
from rl_agent import *
from models import *
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paper parameters
M = 49        # VNFs (grid cells)
N = 6         # time window
H = W = 7     # grid size

data_path = "./dataset"

files = [
    "sms-call-internet-mi-2013-11-01.txt",
    "sms-call-internet-mi-2013-11-02.txt",
    "sms-call-internet-mi-2013-11-03.txt",
    "sms-call-internet-mi-2013-11-04.txt",
    "sms-call-internet-mi-2013-11-05.txt",
    "sms-call-internet-mi-2013-11-06.txt",
    "sms-call-internet-mi-2013-11-07.txt",
    "sms-call-internet-mi-2013-11-08.txt",
    "sms-call-internet-mi-2013-11-09.txt",
    "sms-call-internet-mi-2013-11-10.txt",
    "sms-call-internet-mi-2013-11-11.txt",
    "sms-call-internet-mi-2013-11-12.txt",
    "sms-call-internet-mi-2013-11-13.txt",
    "sms-call-internet-mi-2013-11-14.txt",
    "sms-call-internet-mi-2013-11-15.txt",
    "sms-call-internet-mi-2013-11-16.txt",
    "sms-call-internet-mi-2013-11-17.txt",
    "sms-call-internet-mi-2013-11-18.txt",
    "sms-call-internet-mi-2013-11-19.txt",
    "sms-call-internet-mi-2013-11-20.txt",
    "sms-call-internet-mi-2013-11-21.txt",
    "sms-call-internet-mi-2013-11-22.txt",
    "sms-call-internet-mi-2013-11-23.txt",
    "sms-call-internet-mi-2013-11-24.txt",
    "sms-call-internet-mi-2013-11-25.txt",
    "sms-call-internet-mi-2013-11-26.txt",
    "sms-call-internet-mi-2013-11-27.txt",
    "sms-call-internet-mi-2013-11-28.txt",
    "sms-call-internet-mi-2013-11-29.txt",
    "sms-call-internet-mi-2013-11-30.txt",
]

dfs = []

for f in files:
    print("Loading:", f)
    df_tmp = pd.read_csv(
        os.path.join(data_path, f),
        sep="\t",
        header=None,
        usecols=[0,1,3]  # square_id, timestamp, internet
    )
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

df.columns = ["square_id", "timestamp", "internet"]
df = df.sort_values(["timestamp", "square_id"]).reset_index(drop=True)

# Replace missing values with 0
df.fillna(0, inplace=True)

# Ensure correct types
df["square_id"] = df["square_id"].astype(int)
df["timestamp"] = df["timestamp"].astype(int)
df["internet"] = df["internet"].astype(float)

df.info()

top_squares = (
    df.groupby("square_id")["internet"]
      .sum()
      .sort_values(ascending=False)
      .head(M)
      .index
)

df = df[df["square_id"].isin(top_squares)]

# Sort by time
df = df.sort_values(["timestamp", "square_id"])

times = sorted(df["timestamp"].unique())
square_to_idx = {sq: i for i, sq in enumerate(top_squares)}

traffic = np.zeros((len(times), M))

for _, row in df.iterrows():
    t_idx = times.index(row["timestamp"])
    m_idx = square_to_idx[row["square_id"]]
    traffic[t_idx, m_idx] = row["internet"]

traffic.shape

traffic_min = traffic.min(axis=0)
traffic_max = traffic.max(axis=0)

traffic_norm = (traffic - traffic_min) / (traffic_max - traffic_min + 1e-6)

def create_windows(data, N):
    X, y = [], []
    for t in range(N, len(data)):
        X.append(data[t-N:t])
        y.append(data[t])
    return np.array(X), np.array(y)

X, y = create_windows(traffic_norm, N)

X.shape, y.shape

X = X.reshape(-1, N, H, W)
X = np.transpose(X, (0, 2, 3, 1))   # (batch, 7, 7, N)

X = torch.tensor(X).float()
y = torch.tensor(y).float()

X.shape, y.shape

X_cnn = X.permute(0, 3, 1, 2)   # [batch, 6, 7, 7]
X_cnn = X_cnn.unsqueeze(1)     # [batch, 1, 6, 7, 7]

X_cnn.shape

model_edge = EdgeModel().to(device)
X_cnn = X_cnn.to(device)

# yL, z = model_edge(X_cnn)
yL, z, _, _ = model_edge(X_cnn)

yL.shape, z.shape

############ PHASE 1: TRAIN EDGE MODEL ########################################
y = y.to(device)
optimizer = optim.Adam(model_edge.parameters(), lr=1e-3)

num_epochs = 50

for epoch in range(num_epochs):
    model_edge.train()

    optimizer.zero_grad()

    # Forward pass
    # y_pred, _ = model_edge(X_cnn)
    y_pred, _, _, _ = model_edge(X_cnn)


    # Compute SLA-aware loss
    loss = sla_cost(y_pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}")
################################################################################
############ PHASE 2: WARMUP CLOUD #########################################
model_cloud = CloudModel().to(device)
nn.init.constant_(model_cloud.fc3.bias, 0.5)
optimizer = optim.Adam(
    list(model_edge.parameters()) + list(model_cloud.parameters()),
    lr=1e-3
)

mse = torch.nn.MSELoss()

for epoch in range(20):
    optimizer.zero_grad()
    yL, z,_,_ = model_edge(X_cnn)
    yR = model_cloud(z)
    # loss = mse(yR, y)
    loss = sla_cost(yR, y)
    loss.backward()
    optimizer.step()
    print(f"Warmup Epoch {epoch+1} - MSE: {loss.item():.4f}")
################################################################################
############ PHASE 3: JOINT TRAINING #########################################
num_epochs =100
alpha = 0.3     #Range to try: 0.05 – 0.3
w=0.3           #Range to try: 0.2 – 0.5

for epoch in range(num_epochs):
    model_edge.train()
    model_cloud.train()

    optimizer.zero_grad()

    # Edge forward
    yL, z,_,_ = model_edge(X_cnn)

    # Cloud forward
    yR = model_cloud(z)

    # Smooth MSE loss
    mse_loss = F.mse_loss(yL, y) + F.mse_loss(yR, y)

    # SLA loss
    sla_loss = joint_sla_cost(yL, yR, y, w)

    # Combined loss (include SLA cost with weight alpha)
    loss = 0.25 * F.mse_loss(yL, y) + 0.25 * F.mse_loss(yR, y) + alpha * sla_loss

    # Backprop
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}")
################################################################################

model_edge.eval()
model_cloud.eval()

with torch.no_grad():
    yL, z, _, _ = model_edge(X_cnn)
    yR = model_cloud(z)


CL = sla_cost_per_sample(yL, y)
CR = sla_cost_per_sample(yR, y)

CL.shape, CR.shape


print("CL > CR %:", ((CL - CR) > 0).float().mean())
print("Mean diff:", (CL - CR).mean())
print("Std diff:", (CL - CR).std())
print("Sample CL just after joint training:", CL[:30])
print("Sample CR just after joint training:", CR[:30])
print("samples percentage just after training:",((CL - CR) > 0).float().mean())

########################## RL BLOCK ###############################################
###################################################################################
# state_dim = z.shape[1] + yL.shape[1] + 1
state_dim = 2
agent = RLAgent(state_dim, device)

# rtt_values = [10, 30, 60]
results = {
    "RTT": [],
    "edge_pct": [],
    "cloud_pct": [],
    "avg_total_cost": [],
    "avg_sla_cost": [],
    "avg_latency": [],
    "avg_CL": [],
    "avg_CR": [],
    "avg_benefit": []
}

# lambda_rtt = 0.05
lambda_rtt = 0.015
rl_epochs = 350

cost_scale = CL.abs().mean() + CR.abs().mean() + 1e-6


train_rtt_values = [5, 10, 20, 30, 50, 80]
test_rtt_values = [10, 30, 60, 80, 100]
max_rtt = 100



for epoch in range(rl_epochs):

    # randomly pick RTT
    # RTT_cloud = random.choice(train_rtt_values)
    RTT_cloud = torch.tensor(
        random.choices(train_rtt_values, k=z.size(0)),
        device=device,
        dtype=torch.float32
    ).unsqueeze(1)

    # ===== STATE =====
    cost_diff = ((CL - CR) / cost_scale).unsqueeze(1)
    rtt_norm = RTT_cloud / max_rtt
    # rtt_norm = torch.full((z.size(0), 1), RTT_cloud / max_rtt, device=device)
    state = torch.cat([cost_diff, rtt_norm], dim=1)

    # ===== ACTION =====
    action = agent.select_action(state)

    # ===== REWARD =====
    rewards = []
    for i in range(len(action)):
        if action[i] == 0:  # EDGE
            reward = -(CL[i].item() / cost_scale.item())
        else:  # CLOUD
            reward = -((CR[i].item() / cost_scale.item()) + lambda_rtt * rtt_norm[i].item())
        rewards.append(reward)

    rewards = torch.tensor(rewards).to(device)

    # ===== STORE =====
    for i in range(len(action)):
        agent.store((
            state[i].unsqueeze(0),
            action[i].unsqueeze(0),
            rewards[i].unsqueeze(0)
        ))

    # ===== TRAIN =====
    agent.train()


for RTT_cloud in test_rtt_values:

    print(f"\n===== Testing for RTT = {RTT_cloud} =====")

    # ===== FINAL DECISION AFTER TRAINING =====
    cost_diff = ((CL - CR) / cost_scale).unsqueeze(1)
    rtt_norm = torch.full((z.size(0), 1), float(RTT_cloud) / max_rtt, device=device)
    state = torch.cat([cost_diff, rtt_norm], dim=1)
    offload = agent.select_action(state)

    # ======================= METRICS ==================================
    edge_pct = (offload == 0).float().mean().item()
    cloud_pct = (offload == 1).float().mean().item()

    total_total_cost = 0      # cost + latency
    total_sla_cost = 0        # only SLA cost
    total_latency = 0
    total_CL = 0
    total_CR = 0
    total_benefit = 0

    for i in range(len(offload)):

        cl = CL[i].item()
        cr = CR[i].item()
        benefit = cl - cr

        total_CL += cl
        total_CR += cr
        total_benefit += benefit

        if offload[i] == 0:  # EDGE
            total_total_cost += cl
            total_sla_cost += cl
            total_latency += 0
        else:  # CLOUD
            total_total_cost += (cr / cost_scale.item() + lambda_rtt * rtt_norm[i].item())
            total_latency += rtt_norm[i].item()
            total_sla_cost += cr

    # averages
    n = len(offload)

    avg_total_cost = total_total_cost / n
    avg_sla_cost = total_sla_cost / n
    avg_latency = total_latency / n
    avg_CL = total_CL / n
    avg_CR = total_CR / n
    avg_benefit = total_benefit / n

    # ====================== STORE RESULTS =============================
    # results["avg_cost"].append(avg_cost)
    results["RTT"].append(RTT_cloud)
    results["edge_pct"].append(edge_pct)
    results["cloud_pct"].append(cloud_pct)
    results["avg_total_cost"].append(avg_total_cost)
    results["avg_sla_cost"].append(avg_sla_cost)
    results["avg_latency"].append(avg_latency)
    results["avg_CL"].append(avg_CL)
    results["avg_CR"].append(avg_CR)
    results["avg_benefit"].append(avg_benefit)

    print(f"Edge %: {edge_pct:.2f}, Cloud %: {cloud_pct:.2f}")
    print(f"Avg Total Cost: {avg_total_cost:.4f}")
    print(f"Avg SLA Cost: {avg_sla_cost:.4f}")
    print(f"Avg Latency: {avg_latency:.2f}")
    print(f"Avg Benefit: {avg_benefit:.4f}")



############################### PRINT RESULTS ################################################
####################################################################################

# print("Offloading rate:", offload_rate)
print("Offloading rate:", (offload == 1).float().mean().item())
print("Average Edge Cost:", CL.mean().item())
print("Average Cloud Cost:", CR.mean().item())


print("RTT Sensitivity Analysis")
print("-------------------------")

for i in range(len(results["RTT"])):
    print(f"\nRTT: {results['RTT'][i]}")
    print(f"  Edge %: {results['edge_pct'][i]:.2f}")
    print(f"  Cloud %: {results['cloud_pct'][i]:.2f}")
    print(f"  Avg Total Cost: {results['avg_total_cost'][i]:.4f}")
    print(f"  Avg SLA Cost: {results['avg_sla_cost'][i]:.4f}")
    print(f"  Avg Latency: {results['avg_latency'][i]:.2f}")
    print(f"  Avg Benefit: {results['avg_benefit'][i]:.4f}")


######################## PLOTS ############################################################
####################################################################################

# ==================== PLOT 1 ====================
plt.figure()
plt.plot(results["RTT"], results["edge_pct"], label="Edge %")
plt.plot(results["RTT"], results["cloud_pct"], label="Cloud %")
plt.xlabel("RTT")
plt.ylabel("Decision Percentage")
plt.title("RTT vs Offloading Behavior")
plt.legend()
# plt.show()
plt.savefig("./results/plot1.png")
plt.close()

# ==================== PLOT 2 ====================
plt.figure()
plt.plot(results["RTT"], results["avg_sla_cost"], label="SLA Cost")
plt.plot(results["RTT"], results["avg_latency"], label="Latency")
plt.plot(results["RTT"], results["edge_pct"], label="Edge %")
plt.plot(results["RTT"], results["avg_benefit"], label="Benefit")
plt.xlabel("RTT")
plt.ylabel("Values")
plt.title("RTT vs Metrics")
plt.legend()
plt.savefig("./results/plot_metrics(plot2).png")
plt.close()

# ==================== ATTENTION ===========================================
# ==================== PLOT 3 =======================
model_edge.eval()
with torch.no_grad():
    yL, z, spatial_attn, temporal_attn = model_edge(X_cnn[:1])

# Take first sample
attn_map = spatial_attn[0, 0].cpu().numpy()
plt.figure()
plt.imshow(attn_map, cmap='hot')
plt.colorbar()
plt.title("Spatial Attention (Important Regions)")
# plt.show()
plt.savefig("./results/plot3.png")
plt.close()

# Bright areas → high importance
# Dark areas → low importance

# ==================== PLOT 4 ====================
attn_time = temporal_attn[0].cpu().numpy().flatten()
plt.figure()
plt.plot(attn_time, marker='o')
plt.title("Temporal Attention (Importance of Time Steps)")
plt.xlabel("Time Step")
plt.ylabel("Weight")
plt.grid(True)
# plt.show()
plt.savefig("./results/plot4.png")
plt.close()


print("Plots saved in ./results/")
####################################################################################
########################### RESULTS SAVE #####################################################

# Make sure results folder exists
results_path = "./results/"
os.makedirs(results_path, exist_ok=True)

# Save Phase 5 outputs
np.savetxt(results_path + "offloading_decisions.txt", offload.cpu().numpy())
np.savetxt(results_path + "edge_cost.txt", CL.cpu().numpy())
np.savetxt(results_path + "cloud_cost.txt", CR.cpu().numpy())

print("Phase 5 results saved successfully")

###################################################################################
###################################################################################

