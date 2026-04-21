def sla_cost(y_pred, y_true, c_u=10.0, c_o=1.0):
    """
    SLA-aware cost function from the paper
    y_pred: predicted allocation (batch, 49)
    y_true: true demand (batch, 49)
    """
    under_provision = (y_pred < y_true).float()
    over_provision = (y_pred >= y_true).float()

    cost_under = under_provision * c_u
    cost_over = over_provision * c_o * (y_pred - y_true)

    cost = cost_under + cost_over
    return cost.mean()


def joint_sla_cost(yL, yR, y_true, alpha=0.5):
    CL = sla_cost(yL, y_true)
    CR = sla_cost(yR, y_true)
    return alpha * CL + (1 - alpha) * CR

def sla_cost_per_sample(y_pred, y_true, c_u=10.0, c_o=1.0):
    under = (y_pred < y_true).float()
    over = (y_pred >= y_true).float()
    cost = under * c_u + over * c_o * (y_pred - y_true)
    return cost.mean(dim=1)  # cost per sample


# def decide_execution(yL, yR, y_true, RTT_cloud, lambda_rtt):
#     CL = sla_cost_per_sample(yL, y_true)
#     CR = sla_cost_per_sample(yR, y_true)

#     # Effective cloud cost (paper-style)
#     CR_effective = CR + lambda_rtt * RTT_cloud

#     # offload = (CL > CR_effective).int()/
#     state = torch.cat([z.detach(), yL.detach()], dim=1)
#     offload = agent.select_action(state)
#     return offload

def adjust_costs(CL, CR, target_p=0.45, strength=0.1):
    cost_diff = CL - CR
    current_p = (cost_diff > 0).float().mean().item()

    # shift needed
    shift = target_p - current_p

    # apply small controlled bias
    if abs(shift) > 0.05:
        adjustment = strength * shift * cost_diff.std()

        # shift both in opposite direction (preserve structure)
        CL = CL - adjustment
        CR = CR + adjustment

    return CL, CR