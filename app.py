import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Space Leasing Optimizer", layout="wide")

st.title("Warehouse Space Leasing Optimizer")
st.write("Determine the minimum cost leasing strategy based on monthly space needs and available lease terms.")

# Sidebar: Configurable Lease Periods and Costs
st.sidebar.header("1. Lease Duration Options & Costs")
default_leases = pd.DataFrame([
    {"Duration (Months)": 1, "Cost per Sq Ft ($)": 10.0},
    {"Duration (Months)": 2, "Cost per Sq Ft ($)": 18.0},
    {"Duration (Months)": 3, "Cost per Sq Ft ($)": 24.0},
])
lease_df = st.sidebar.data_editor(
    default_leases,
    num_rows="dynamic",
    key="lease_editor"
)

# Main Section: Configurable Planning Horizon & Space Requirements
st.header("2. Monthly Required Space")
num_months = st.number_input("Number of Planning Months", min_value=1, max_value=24, value=4)

default_space = pd.DataFrame({
    "Month": [f"Month {i+1}" for i in range(int(num_months))],
    "Required Sq Ft": [1000, 1500, 1200, 800][:int(num_months)] + [1000] * max(0, int(num_months) - 4)
})

space_df = st.data_editor(
    default_space,
    num_rows="fixed",
    key="space_editor"
)

# Solver Section
st.header("3. Optimal Leasing Plan")

if st.button("Solve Optimization Problem", type="primary"):
    reqs = space_df["Required Sq Ft"].tolist()
    T = len(reqs)
    
    # Extract lease options from sidebar dataframe
    lease_costs = dict(zip(lease_df["Duration (Months)"], lease_df["Cost per Sq Ft ($)"]))
    
    # Define Decision Variables x_{i, k}: Start month i (0..T-1), Duration k
    vars_info = []
    for i in range(T):
        for duration, cost in lease_costs.items():
            duration = int(duration)
            if i + duration <= T:  # Lease starts in horizon and ends within or at boundary
                vars_info.append((i, duration, float(cost)))
                
    if not vars_info:
        st.error("No valid lease options fit within the selected planning horizon.")
    else:
        num_vars = len(vars_info)
        c = [v[2] for v in vars_info]  # Cost coefficients
        
        # Build constraint matrix: active space in month t >= required space in month t
        A_ub = np.zeros((T, num_vars))
        b_ub = np.array(reqs, dtype=float)
        
        for var_idx, (i, duration, cost) in enumerate(vars_info):
            for t in range(i, min(i + duration, T)):
                A_ub[t, var_idx] = 1.0
                
        # scipy linprog minimizes c^T x s.t. A_ub * x <= b_ub
        # Multiplying by -1 changes constraint to >=
        res = linprog(c, A_ub=-A_ub, b_ub=-b_ub, bounds=(0, None), method='highs')
        
        if res.success:
            st.success(f"Optimal Minimum Total Cost: ${res.fun:,.2f}")
            
            plan = []
            for idx, val in enumerate(res.x):
                if val > 0.001:
                    i, duration, cost = vars_info[idx]
                    plan.append({
                        "Start Month": f"Month {i+1}",
                        "Lease Duration": f"{duration} Month(s)",
                        "Sq Ft Leased": round(val, 2),
                        "Total Cost ($)": round(val * cost, 2)
                    })
            st.subheader("Recommended Decisions")
            st.dataframe(pd.DataFrame(plan), use_container_width=True)
        else:
            st.error("Optimization failed to find a valid solution.")
