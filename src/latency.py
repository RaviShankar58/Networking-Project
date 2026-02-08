def expected_latency(L, TF0=0.1, TF1=0.12, TF2=0.23, RTT=42.46):
    return L * (TF0 + TF1) + (1 - L) * (TF0 + RTT + TF2)
