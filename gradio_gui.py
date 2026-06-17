import gradio as gr
import os
import time
from pathlib import Path
import torch
import torch.fx
from examples.demo_model import DemoModel, DemoModelConfig
from graph import build_graph

# Global/Closure variables to hold state across the session
class PipelineState:
    def __init__(self):
        self.thread_id = "agentic_riscv_gradio_run_01"
        self.app = None
        self.run_config = {"configurable": {"thread_id": self.thread_id}}

pipeline = PipelineState()

def run_pipeline(model_choice, start_from, optimize, precision, weight_mode):
    pipeline.app = build_graph(entry_point=start_from)
    
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

    logs = "🚀 Pipeline started...\n"
    yield logs, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""
    
    try:
        for event in pipeline.app.stream(initial_state, pipeline.run_config):
            for node_name in event.keys():
                logs += f"✅ Finished node: {node_name}\n"
                yield logs, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""
        
        snapshot = pipeline.app.get_state(pipeline.run_config)
        if snapshot.next and "human_review" in snapshot.next:
            logs += "⏸️ Paused for Human Review\n"
            current_state = snapshot.values
            code_path = current_state.get("code_path", "")
            code_content = ""
            if code_path and os.path.exists(code_path):
                with open(code_path, "r") as f:
                    code_content = f.read()
            yield logs, gr.update(visible=True), gr.update(visible=False), code_content, "", "", "", ""
        else:
            logs += "🎉 Pipeline completed\n"
            current_state = snapshot.values
            metrics = current_state.get("llm_metrics", {})
            report = current_state.get("final_report", "")
            yield logs, gr.update(visible=False), gr.update(visible=True), "", f"{metrics.get('input_tokens',0):,}", f"{metrics.get('output_tokens',0):,}", f"{metrics.get('latency_sec',0):.1f}s / ${metrics.get('cost_usd',0):.5f}", report
            
    except Exception as e:
        logs += f"❌ Error: {e}\n"
        yield logs, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""

def resume_pipeline(action_dict):
    pipeline.app.update_state(pipeline.run_config, action_dict)
    if action_dict.get("human_approved"):
        logs = "⏭️ Resuming after approval...\n"
    else:
        logs = "🔁 Retrying generation...\n"
        
    yield logs, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""
    
    try:
        for event in pipeline.app.stream(None, pipeline.run_config):
            for node_name in event.keys():
                logs += f"✅ Finished node: {node_name}\n"
                yield logs, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""
        
        snapshot = pipeline.app.get_state(pipeline.run_config)
        if snapshot.next and "human_review" in snapshot.next:
            logs += "⏸️ Paused for Human Review\n"
            current_state = snapshot.values
            code_path = current_state.get("code_path", "")
            code_content = ""
            if code_path and os.path.exists(code_path):
                with open(code_path, "r") as f:
                    code_content = f.read()
            yield logs, gr.update(visible=True), gr.update(visible=False), code_content, "", "", "", ""
        else:
            logs += "🎉 Pipeline completed\n"
            current_state = snapshot.values
            metrics = current_state.get("llm_metrics", {})
            report = current_state.get("final_report", "")
            yield logs, gr.update(visible=False), gr.update(visible=True), "", f"{metrics.get('input_tokens',0):,}", f"{metrics.get('output_tokens',0):,}", f"{metrics.get('latency_sec',0):.1f}s / ${metrics.get('cost_usd',0):.5f}", report
            
    except Exception as e:
        logs += f"❌ Error: {e}\n"
        yield logs, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""

def approve_code():
    yield from resume_pipeline({"human_approved": True, "human_action": "approve"})

def reject_code(feedback):
    yield from resume_pipeline({"human_approved": False, "human_feedback": feedback, "human_action": "retry"})


with gr.Blocks(title="Agentic RISC-V Compiler", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ Agentic RISC-V Compiler\nAutomated neural network compilation and optimization for bare-metal RISC-V.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            model_choice = gr.Dropdown(choices=["TinyLlama Demo"], value="TinyLlama Demo", label="Select Model")
            start_from = gr.Dropdown(choices=["parse_fx", "simulate", "synthesis", "report"], value="parse_fx", label="Start From")
            optimize = gr.Checkbox(label="Enable Closed-Loop Optimization", value=False)
            precision = gr.Dropdown(choices=["f32", "f16", "bf16", "mxfp8"], value="f32", label="Weight Precision")
            weight_mode = gr.Dropdown(choices=["embedded", "binary"], value="embedded", label="Weight Mode")
            start_btn = gr.Button("▶️ Start Pipeline", variant="primary")
            
            logs_output = gr.TextArea(label="📝 Pipeline Status", interactive=False, lines=15)
            
        with gr.Column(scale=2):
            with gr.Group(visible=False) as review_group:
                gr.Markdown("### 🔍 Human Review\nThe generated bare-metal C code requires your review before compiling for RISC-V.")
                code_output = gr.Code(language="c", label="Generated model.c", interactive=False)
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve Code", variant="primary")
                with gr.Row():
                    feedback_input = gr.Textbox(label="Feedback for Code Generator (Optional)")
                    reject_btn = gr.Button("❌ Reject & Retry")
                    
            with gr.Group(visible=False) as metrics_group:
                gr.Markdown("### 📊 Final Report & Metrics")
                gr.Markdown("#### LLM Execution Metrics (AMD MI300X)")
                with gr.Row():
                    in_tokens = gr.Textbox(label="Input Tokens", interactive=False)
                    out_tokens = gr.Textbox(label="Output Tokens", interactive=False)
                    latency_cost = gr.Textbox(label="Latency / Estimated Cost", interactive=False)
                gr.Markdown("---")
                report_md = gr.Markdown()
                
    start_btn.click(
        run_pipeline, 
        inputs=[model_choice, start_from, optimize, precision, weight_mode], 
        outputs=[logs_output, review_group, metrics_group, code_output, in_tokens, out_tokens, latency_cost, report_md]
    )
    approve_btn.click(
        approve_code, 
        outputs=[logs_output, review_group, metrics_group, code_output, in_tokens, out_tokens, latency_cost, report_md]
    )
    reject_btn.click(
        reject_code, 
        inputs=[feedback_input], 
        outputs=[logs_output, review_group, metrics_group, code_output, in_tokens, out_tokens, latency_cost, report_md]
    )

if __name__ == "__main__":
    app.launch()
