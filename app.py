import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Call Center Simulator",
    page_icon="📞",
    layout="wide"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #8b949e !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #58a6ff !important;
    }
    h1, h2, h3 {
        color: #f0f6fc;
    }
    .highlight-box {
        background-color: #21262d;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #238636;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- MAIN TITLE & HEADER ---
st.title("📞 Advanced Call Center Operations & Economic Simulator")
st.markdown("""
<div class="highlight-box">
    <b>Applied Probability & Queueing Theory Project:</b> Featuring <b>M/M/c Model</b>, 
    <b>Time-Varying Poisson Arrivals (Peak Hours)</b>, and <b>Cost-Service Trade-off Optimization</b> (Currencies in SAR).
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS WITH STATISTICAL LAWS ---
st.sidebar.header("⚙️ Simulation Controls (Queueing Parameters)")

num_agents = st.sidebar.slider(
    "Number of Support Agents [c (Servers)]", 
    1, 500, 25, 1
)

base_arrival_rate = st.sidebar.slider(
    "Base Call Arrival Rate [Poisson Process - λ]", 
    1.0, 100.0, 15.0, 1.0
)

service_time = st.sidebar.slider(
    "Average Service Time [Exponential Service - 1/μ]", 
    1.0, 10.0, 2.5, 0.5
)

sim_duration = st.sidebar.slider(
    "Simulation Duration [Time Horizon - T]", 
    60, 720, 300, 60
)

st.sidebar.header("💰 Economic Parameters (Cost Analysis in SAR)")

agent_cost_per_hr = st.sidebar.slider(
    "Agent Wage [Operational Cost Rate - C_w] (SAR/hour)", 
    10, 200, 40, 5
)

loss_cost_per_call = st.sidebar.slider(
    "Estimated Loss per Lost Call [Penalty Cost - C_l] (SAR)", 
    5, 500, 50, 5
)

# --- SIMULATION CORE WITH PEAK HOURS ---
def run_advanced_simulation(sim_time, base_lam, serv_time, c):
    env = simpy.Environment()
    agents = simpy.Resource(env, capacity=c)
    
    wait_times = []
    queue_lengths = []
    time_stamps = []
    total_calls = 0
    served_calls = 0
    abandoned_calls = 0

    def customer(env, name, agents, wait_times):
        nonlocal served_calls
        arrival_time = env.now
        with agents.request() as req:
            results = yield req | env.timeout(random.uniform(2, 6)) # Exponential/Uniform patience threshold
            if req in results:
                wait = env.now - arrival_time
                wait_times.append(wait)
                served_calls += 1
                dur = random.expovariate(1.0 / serv_time)
                yield env.timeout(dur)
            else:
                nonlocal abandoned_calls
                abandoned_calls += 1

    def call_generator(env, agents, base_lam, wait_times):
        nonlocal total_calls
        i = 0
        while True:
            current_time = env.now
            # Non-homogeneous Poisson Process wave for peak hours
            peak_factor = 1.0 + 1.8 * np.sin(np.pi * current_time / sim_time)
            current_lam = max(1.0, base_lam * peak_factor)
            
            interarrival = random.expovariate(current_lam)
            yield env.timeout(interarrival)
            i += 1
            total_calls += 1
            env.process(customer(env, f'Call {i}', agents, wait_times))
            queue_lengths.append(len(agents.queue))
            time_stamps.append(env.now)

    env.process(call_generator(env, agents, base_lam, wait_times))
    env.run(until=sim_time)
    
    return wait_times, queue_lengths, time_stamps, total_calls, served_calls, abandoned_calls

# --- ACTION BUTTON ---
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_btn = st.button("🚀 Run Advanced Simulation", use_container_width=True)

if run_btn:
    with st.spinner("🔄 Running stochastic process simulation & evaluating queue metrics..."):
        random.seed(42)
        waits, q_lens, t_stamps, total_c, served_c, abandoned_c = run_advanced_simulation(
            sim_duration, base_arrival_rate, service_time, num_agents
        )
        
        avg_wait = sum(waits) / len(waits) if len(waits) > 0 else 0
        
        # Financial & Statistical calculations
        sim_hours = sim_duration / 60.0
        operational_cost = num_agents * agent_cost_per_hr * sim_hours
        loss_cost = abandoned_c * loss_cost_per_call
        total_economic_cost = operational_cost + loss_cost

        # --- SECTION 1: KPIS ---
        st.markdown("---")
        st.markdown("### 📊 Key Performance Indicators (Queueing Metrics)")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Calls Received [N]", f"{total_c:,}")
        with kpi2:
            st.metric("Successfully Served [Throughput]", f"{served_c:,}")
        with kpi3:
            st.metric("Abandoned Calls [Reneging Model]", f"{abandoned_c:,}")
        with kpi4:
            st.metric("Avg Waiting Time [W_q]", f"{avg_wait:.2f} min")

        # --- SECTION 2: COST VS SERVICE ---
        st.markdown("### 💵 Cost vs. Service Trade-off Analysis (Optimization in SAR)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Staff Operational Cost [c × Wage]", f"{operational_cost:,.2f} SAR")
        with c2:
            st.metric("Customer Loss Cost [Abandoned × Penalty]", f"{loss_cost:,.2f} SAR")
        with c3:
            st.metric("Total Economic Impact [Objective Function]", f"{total_economic_cost:,.2f} SAR")

        # --- SECTION 3: ADVANCED VISUALIZATION ---
        st.markdown("---")
        st.markdown("### 📈 Queue Length Dynamics [Stochastic Process L_q(t)]")
        
        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#161b22')
        
        ax.plot(t_stamps, q_lens, color='#58a6ff', linewidth=1.8, label='Queue Length Process [L_q(t)]')
        ax.set_xlabel("Simulation Time (Minutes) [t]", color='#f0f6fc', fontsize=11)
        ax.set_ylabel("Number of Waiting Customers [L_q]", color='#f0f6fc', fontsize=11)
        ax.tick_params(colors='#8b949e', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3, color='#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')
        
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            
        st.pyplot(fig)
        
        st.success("✨ Simulation executed successfully with SAR currencies and expanded agent capacity!")
else:
    st.info("👈 Configure your parameters in the sidebar and click **'Run Advanced Simulation'** above to generate the dashboard.")
