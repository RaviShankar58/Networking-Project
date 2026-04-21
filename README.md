# 📊 Edge–Cloud Offloading using CNN-LSTM-Attention + RL

This project implements an intelligent **edge vs cloud offloading system** using:
- Deep Learning (CNN + LSTM + Attention)
- SLA-aware cost optimization
- Reinforcement Learning (RL)

The objective is to minimize **total cost (SLA + latency)** by deciding whether to execute tasks on the **edge** or **cloud**.

---

# 🚀 Project Pipeline

## 1. Data Processing
- Uses Telecom Italia Milano Grid Dataset
- Extracts:
  - square_id
  - timestamp
  - internet traffic
- Selects top **M = 49 grid cells**
- Normalizes traffic data
- Creates time windows of size **N = 6**

---

## 2. Model Architecture

### Edge Model
- CNN + LSTM + Attention
- Outputs:
  - yL → Edge prediction
  - z → latent features

### Cloud Model
- Fully connected network
- Uses latent representation z
- Outputs:
  - yR → Cloud prediction

---

## 3. Training Phases

### Phase 1: Edge Training
Train only edge model using SLA loss

### Phase 2: Cloud Warmup
Train cloud model using edge features

### Phase 3: Joint Training
Train both models together

Loss Function:
Loss = 0.25*MSE(yL, y) + 0.25*MSE(yR, y) + alpha * SLA_loss

---

## 4. RL-Based Offloading

State:
[ (CL - CR), RTT ]

Actions:
0 → Edge  
1 → Cloud  

Reward:
- Penalizes SLA cost
- Penalizes cloud latency

---

## 5. Metrics

- Edge / Cloud percentage
- Total cost
- SLA cost
- Latency
- Benefit = CL - CR

---

# 📂 Project Structure

project/
│── train.py  
│── models.py  
│── rl_agent.py  
│── utils.py  
│── dataset/  
│── results/  

---

# ⚙️ Installation

## 1. Clone Repository
git clone <your-repo-link>  
cd project  

## 2. Create Virtual Environment
python -m venv venv  

Activate:
- Linux/Mac:
  source venv/bin/activate  
- Windows:
  venv\Scripts\activate  

## 3. Install Dependencies
pip install torch numpy pandas matplotlib  

---

# 📁 Dataset Setup

1. Create dataset folder:
dataset/

2. Add Milano dataset files:
sms-call-internet-mi-2013-11-01.txt  
...  
sms-call-internet-mi-2013-11-30.txt  

3. Ensure path in code:
data_path = "./dataset"

---

# ▶️ How to Run

Run the full pipeline:
python train.py

---

# 🖥️ GPU Support

The code automatically uses GPU if available:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Check GPU:
python -c "import torch; print(torch.cuda.is_available())"

---

# 📊 Output

All outputs are saved in:
results/

Generated files:
- plot1.png → RTT vs Edge/Cloud decisions
- plot_metrics(plot2).png → Metrics comparison
- plot3.png → Spatial attention
- plot4.png → Temporal attention
- offloading_decisions.txt
- edge_cost.txt
- cloud_cost.txt

---

# 📈 Key Observations

- Low RTT → Cloud preferred  
- High RTT → Edge preferred  
- RL learns adaptive offloading  
- Attention highlights important regions and time steps  

---

# 🔧 Important Parameters

M = 49  
N = 6  
alpha = 0.3  
w = 0.3  
lambda_rtt = 0.015  
rl_epochs = 350  

---

# 🧠 Future Work

- Real-time data integration  
- Advanced RL algorithms (DQN, PPO)  
- Multi-agent systems  
- Dynamic network adaptation  

---

# 👨‍💻 Author

122cs0058,122ad0031
