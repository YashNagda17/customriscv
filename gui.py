import streamlit as st
import os
import time
from pathlib import Path

# Important: these must be imported to run the graph
import torch
import torch.fx
from examples.demo_model import DemoModel, DemoModelConfig
from graph import build_graph

# Configure Streamlit page layout
st.set_page_config(page_title="Agentic RISC-V Compiler", layout="wide", page_icon="⚡")

st.title("⚡ Agentic RISC-V Compiler")
st.markdown("Automated neural network compilation and optimization for bare-metal RISC-V.")

# Initialize session state variables to persist across reruns
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "agentic_riscv_gui_run_01"
if "app" not in st.session_state:
    st.session_state.app = build_graph(entry_point="parse_fx")
if "status_logs" not in st.session_state:
    st.session_state.status_logs = []
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_paused" not in st.session_state:
    st.session_state.pipeline_paused = False
if "pipeline_finished" not in st.session_state:
    st.session_state.pipeline_finished = False

# ── Sidebar: Configuration ──────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox("Select Model", ["TinyLlama Demo"])
    start_from = st.selectbox("Start From", ["parse_fx", "simulate", "synthesis", "report"])
    optimize = st.checkbox("Enable Closed-Loop Optimization", value=False)
    precision = st.selectbox("Weight Precision", ["f32", "f16", "bf16", "mxfp8"])
    weight_mode = st.selectbox("Weight Mode", ["embedded", "binary"])
    
    st.markdown("---")
    
    if st.button("▶️ Start Pipeline", type="primary", use_container_width=True) and not st.session_state.pipeline_running:
        st.session_state.app = build_graph(entry_point=start_from)
        
        # Load demo model if selected
        if model_choice == "TinyLlama Demo":
            config = DemoModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, seq_len=16)
            model = DemoModel(config)
            sample_input = torch.randint(0, config.vocab_size, (1, config.seq_len))
            traced_model = torch.fx.symbolic_trace(model)
            reference_outputs = []
            
        initial_state = {
            "model_name": "TinyLlamaDemo",
            "fx_graph_str": str(traced_model.graph),
            "reference_outputs": reference_outputs,
            "enable_optimization": optimize,
            "weight_precision": precision,
            "weight_mode": weight_mode,
            "verification_attempts": 0,
            "optimization_iteration": 0,
            "human_approved": False,
            "human_feedback": "",
        }
        
        if start_from == "parse_fx":
            import agents.fx_parser
            agents.fx_parser.GLOBAL_PYTORCH_OBJECTS["model"] = model
            agents.fx_parser.GLOBAL_PYTORCH_OBJECTS["fx_graph"] = traced_model
            agents.fx_parser.GLOBAL_PYTORCH_OBJECTS["sample_input"] = sample_input

        st.session_state.initial_state = initial_state
        st.session_state.pipeline_running = True
        st.session_state.pipeline_paused = False
        st.session_state.pipeline_finished = False
        st.session_state.status_logs = ["🚀 Pipeline started..."]
        st.rerun()

# ── Main UI Layout ──────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

# Left Column: Execution Logs
with col1:
    st.subheader("📝 Pipeline Status")
    log_container = st.container(height=500)
    
    # Core Pipeline Execution Logic
    if st.session_state.pipeline_running and not st.session_state.pipeline_paused and not st.session_state.pipeline_finished:
        run_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        app = st.session_state.app
        
        try:
            # We must pass None if resuming, or initial_state if starting
            state_input = st.session_state.initial_state if "initial_state" in st.session_state else None
            
            # Run LangGraph streaming
            for event in app.stream(state_input, run_config):
                for node_name in event.keys():
                    st.session_state.status_logs.append(f"✅ Finished node: {node_name}")
            
            # Remove initial state so we don't restart from beginning on resume
            if "initial_state" in st.session_state:
                del st.session_state.initial_state
                
            # Check if paused for human review
            snapshot = app.get_state(run_config)
            if snapshot.next and "human_review" in snapshot.next:
                st.session_state.pipeline_paused = True
                st.session_state.status_logs.append("⏸️ Paused for Human Review")
            else:
                st.session_state.pipeline_running = False
                st.session_state.pipeline_finished = True
                st.session_state.status_logs.append("🎉 Pipeline completed")
                
            st.rerun()
            
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.pipeline_running = False
            
    with log_container:
        for log in st.session_state.status_logs:
            st.text(log)

# Right Column: Human Review & Metrics Dashboard
with col2:
    if st.session_state.pipeline_paused:
        st.subheader("🔍 Human Review")
        st.info("The generated bare-metal C code requires your review before compiling for RISC-V.")
        
        run_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        snapshot = st.session_state.app.get_state(run_config)
        current_state = snapshot.values
        
        code_path = current_state.get("code_path", "")
        if code_path and os.path.exists(code_path):
            with open(code_path, "r") as f:
                code_content = f.read()
            # Show code in a scrollable block
            with st.expander("📄 View Generated model.c", expanded=True):
                st.code(code_content, language="c")
            
        is_exhausted = current_state.get('verification_exhausted', False)
        if is_exhausted:
            st.error("⚠️ Max verification attempts reached! You must manually fix the code on disk before proceeding.")
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ Approve Code", use_container_width=True):
                st.session_state.app.update_state(run_config, {"human_approved": True, "human_action": "approve"})
                st.session_state.pipeline_paused = False
                st.session_state.status_logs.append("⏭️ Resuming after approval...")
                st.rerun()
        with col_btn2:
            if not is_exhausted:
                feedback = st.text_input("Feedback for Code Generator (Optional)")
                if st.button("❌ Reject & Retry", use_container_width=True):
                    st.session_state.app.update_state(run_config, {"human_approved": False, "human_feedback": feedback, "human_action": "retry"})
                    st.session_state.pipeline_paused = False
                    st.session_state.status_logs.append("🔁 Retrying generation...")
                    st.rerun()
            else:
                if st.button("🔄 Verify Manual Edits", use_container_width=True):
                    st.session_state.app.update_state(run_config, {"human_approved": False, "human_action": "verify"})
                    st.session_state.pipeline_paused = False
                    st.session_state.status_logs.append("🔁 Verifying edits...")
                    st.rerun()
                    
    elif st.session_state.pipeline_finished:
        st.subheader("📊 Final Report & Metrics")
        
        run_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        snapshot = st.session_state.app.get_state(run_config)
        current_state = snapshot.values
        
        # Render Metrics
        metrics = current_state.get("llm_metrics", {})
        if metrics:
            st.markdown("### LLM Execution Metrics (AMD MI300X)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Input Tokens", f"{metrics.get('input_tokens', 0):,}")
            m2.metric("Output Tokens", f"{metrics.get('output_tokens', 0):,}")
            latency = metrics.get('latency_sec', 0.0)
            cost = metrics.get('cost_usd', 0.0)
            m3.metric("Latency", f"{latency:.1f} sec")
            m4.metric("Estimated Cost", f"${cost:.5f}")
            st.markdown("---")
            
        # Render Final Report
        report = current_state.get("final_report", "")
        if report:
            st.markdown(report)
        else:
            st.info("Pipeline finished, but no final report was generated.")
    else:
        st.info("Select options from the sidebar and click 'Start Pipeline' to begin.")
