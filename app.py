import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Call Center Queue & Cost Simulator")

st.title("📞 Advanced Call Center Operations Simulator")
st.markdown("A simulation model featuring **Time-Varying Peak Hours** and **Cost-Service Trade-off Analysis**.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Simulation Settings")
num_agents = st.sidebar.slider("Number of Support Agents (c)", 1, 20, 5, 1)
base_arrival_rate = st.sidebar.slider("Base Call Arrival Rate (calls/min)", 1.0, 30.0, 8.0, 1.0)
service_time = st.sidebar.slider("Average Service Time (minutes)", 1.0, 10.0, 3.0, 0.5)
sim_duration = st.sidebar.slider("Simulation Duration (minutes)", 60, 720, 240, 60)

st.sidebar.header("💰 Economic Parameters")
agent_cost_per_hr = st.sidebar.slider("Agent Wage ($/hour)", 10, 50, 20, 5)
loss_cost_per_call = st.sidebar.slider("Estimated Loss per Lost Call ($)", 5, 100, 30, 5)

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
            # Wait for an agent
            results = yield req | env.timeout(random.uniform(2, 5)) # patience threshold simulation
            if req in results:
                wait = env.now - arrival_time
                wait_times.append(wait)
                served_calls += 1
                dur = random.expovariate(1.0 / serv_time)
                yield env.timeout(dur)
            else:
                # Customer hung up due to waiting
                nonlocal abandoned_calls
                abandoned_calls += 1

    def call_generator(env, agents, base_lam, wait_times):
        nonlocal total_calls
        i = 0
        while True:
            # Time-varying arrival rate (Peak Hours Simulation: peak in the middle)
            current_time = env.now
            peak_factor = 1.0 + 1.5 * np.sin(np.pi * current_time / sim_time)
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

if st.button("▶️ Run Call Center Simulation"):
    with st.spinner("Simulating call center traffic and peak hours..."):
        random.seed(42)
        waits, q_lens, t_stamps, total_c, served_c, abandoned_c = run_advanced_simulation(
            sim_duration, base_arrival_rate, service_time, num_agents
        )
        
        avg_wait = sum(waits) / len(waits) if len(waits) > 0 else 0
        
        # Financial calculations
        sim_hours = sim_duration / 60.0
        operational_cost = num_agents * agent_cost_per_hr * sim_hours
        loss_cost = abandoned_c * loss_cost_per_call
        total_economic_cost = operational_cost + loss_cost

        # --- RESULTS DASHBOARD ---
        st.markdown("### 📊 Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Calls Received", f"{total_c}")
        col2.metric("Successfully Served", f"{served_c}")
        col3.metric("Abandoned Calls", f"{abandoned_c}", delta_color="inverse")
        col4.metric("Avg Waiting Time", f"{avg_wait:.2f} min")

        st.markdown("### 💵 Cost vs. Service Analysis")
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric("Staff Operational Cost", f"${operational_cost:.2f}")
        c_col2.metric("Customer Loss Cost", f"${loss_cost:.2f}")
        c_col3.metric("Total Economic Impact", f"${total_economic_cost:.2f}")

        # --- VISUALIZATION ---
        st.markdown("### 📈 Queue Length Dynamics (Peak Hours Wave)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_stamps, q_lens, color='#2ca02c', linewidth=1.5, label='Callers Waiting')
        ax.set_xlabel("Simulation Time (Minutes)")
        ax.set_ylabel("Number of Waiting Customers")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        
        st.success("✨ Call center peak simulation completed successfully!")
else:
    st.info("👈 Adjust your parameters in the sidebar and click **'Run Call Center Simulation'**.")
