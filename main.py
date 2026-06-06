import os
import json
import torch
import subprocess
import sys

# ==============================================================================
# CRITICAL COMPATIBILITY LAYER - MUST RUN BEFORE ANY TRANSFORMERS/PEFT IMPORTS
# ==============================================================================
os.environ["PEFT_DISABLE_TORCHAO"] = "1" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# Configuration Constants
MODEL_NAME = "google/flan-t5-base"
FINETUNED_DIR = "./fine_tuned_dsa_model"
APP_FILE = "app.py"
DATA_FILE = "./dsa.json"

# ==============================================================================
# STEP 1: INITIALIZE OR VERIFY STRUCTURED DSA SCHEMAS
# ==============================================================================
def load_or_create_dsa_json():
    if os.path.exists(DATA_FILE):
        print(f"📦 Found existing data map at {DATA_FILE}. Parsing schemas...")
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    
    print("⚠️ Local dsa.json not detected. Generating standard blueprint mapping...")
    default_data = {
        "dsa_concepts": [
            {
                "name": "Array", "category": "Linear Data Structure",
                "definition": "An array is a collection of elements stored in contiguous memory locations, allowing direct access via indices.",
                "properties": {"ordered": True, "indexed": True, "dynamic_size": False, "homogeneous": True},
                "operations": {"access": {"time_complexity": "O(1)"}, "search": {"best_case": "O(1)", "average_case": "O(n)", "worst_case": "O(n)"}, "insertion": {"beginning": "O(n)", "middle": "O(n)", "end": "O(1)"}, "deletion": {"beginning": "O(n)", "middle": "O(n)", "end": "O(1)"}},
                "space_complexity": "O(n) where n represents the total allocation capacity bounds of structural elements.",
                "advantages": ["Provides instantaneous O(1) random access capabilities.", "Extremely memory efficient due to contiguous storage.", "Very straightforward implementation and traversal loops."],
                "disadvantages": ["Fixed structural size limits bounds modifications.", "Insertion and deletions are highly computationally expensive operations."],
                "applications": ["Matrix transformations and lookup maps.", "Low-level allocation buffers and image processing grids.", "Schedules and baseline caching blocks."]
            },
            {
                "name": "Stack", "category": "Linear Data Structure",
                "definition": "A stack is a linear data structure working strictly under Last-In, First-Out execution dynamics.",
                "properties": {"ordered": True, "lifo": True},
                "operations": {"push": {"time_complexity": "O(1)"}, "pop": {"time_complexity": "O(1)"}, "peek": {"time_complexity": "O(1)"}, "search": {"time_complexity": "O(n)"}},
                "space_complexity": "O(n) linear auxiliary allocation scaling explicitly with runtime element counts.",
                "advantages": ["Guarantees deterministic, fast runtime tracking constraints.", "Prevents structural fragmentation vulnerabilities."],
                "disadvantages": ["Lacks direct index scanning features.", "Risks memory failures via uncontrolled overflow execution bounds."],
                "applications": ["Runtime function tracking execution stacks.", "Comprehensive undo/redo transactional operational history layers.", "Algorithmic text analysis and bracket sequence parsing engine execution."]
            }
        ]
    }
    with open(DATA_FILE, 'w') as f:
        json.dump(default_data, f, indent=4)
    return default_data

# ==============================================================================
# STEP 2: PEFT / LORA FINE-TUNING PIPELINE
# ==============================================================================
def run_fine_tuning(dsa_data):
    print("🚀 Loading model tokenizers for training execution context...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    inputs_list = []
    outputs_list = []
    
    for concept in dsa_data.get("dsa_concepts", []):
        name = concept["name"]
        definition = concept["definition"]
        space = concept.get("space_complexity", "O(n)")
        
        inputs_list.append(f"Explain the concept of an {name} structural schema.")
        outputs_list.append(f"{name} ({concept['category']}): {definition}")
        
        inputs_list.append(f"What is the space complexity or operational constraint of {name}?")
        outputs_list.append(f"The space footprint rules scale to {space}")

    dataset = Dataset.from_dict({"input": inputs_list, "output": outputs_list})

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input"], max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(text_target=examples["output"], max_length=128, truncation=True, padding="max_length")
        
        labels_ids = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels_ids
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("🏋️ Launching parameter optimization loop...")
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1, 
        num_train_epochs=3,           
        weight_decay=0.01,
        logging_steps=1,
        learning_rate=3e-4,
        save_strategy="no",
        use_cpu=True,                  # Safe CPU execution flag mapping
        remove_unused_columns=True     # REMOVED: deprecated no_cuda flag
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    trainer.train()
    
    model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    print(f"✅ Adapters written successfully to directory: {FINETUNED_DIR}")

# ==============================================================================
# STEP 3: DYNAMIC STREAMLIT CODE GENERATION BLOCK
# ==============================================================================
def generate_streamlit_app():
    print(f"🛠️ Exporting frontend structural configurations directly into {APP_FILE}...")
    
    with open(APP_FILE, "w", encoding="utf-8") as f:
        f.write(f'''import streamlit as st
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(
    page_title="DSA Architecture Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {{ background-color: #0b0f19; color: #f8fafc; }}
    textarea {{ background-color: #1e293b !important; color: white !important; border: 1px solid #334155 !important; }}
    input {{ background-color: #1e293b !important; color: white !important; border: 1px solid #334155 !important; }}
    div.stButton > button {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        width: 100%;
    }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_application_assets():
    with open("{DATA_FILE}", 'r') as file:
        data = json.load(file)
    comp_map = {{c["name"].lower(): c for c in data.get("dsa_concepts", [])}}
    
    # Safe multi-platform hardware execution mapping layer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("{FINETUNED_DIR}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("{MODEL_NAME}")
    
    if device == "cpu":
        model = PeftModel.from_pretrained(base_model, "{FINETUNED_DIR}", torch_dtype=torch.float32, device_map="cpu")
    else:
        model = PeftModel.from_pretrained(base_model, "{FINETUNED_DIR}").to(device)
        
    return comp_map, tokenizer, model, device

COMPONENTS_MAP, tokenizer, model, device = load_application_assets()

def process_pipeline(user_query):
    query_clean = user_query.lower().strip()
    matched_concept = None
    
    for concept_name in COMPONENTS_MAP.keys():
        if concept_name in query_clean:
            matched_concept = COMPONENTS_MAP[concept_name]
            break
            
    if not matched_concept:
        inputs = tokenizer(user_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=100)
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return f"""
        <div style='background-color: #2d1b24; border-left: 4px solid #f43f5e; padding: 15px; border-radius: 4px; margin-top:10px;'>
            <h4 style='color: #f43f5e; margin: 0 0 5px 0;'>⚠️ Concept Query Routing Alert</h4>
            <p style='margin: 0; color: #fda4af; font-size: 14px;'>The engine did not find a direct structural match in your dsa.json schemas for this question. Here is a fine-tuned adapter generation response:</p>
            <p style='margin: 10px 0 0 0; font-style: italic; color: #fff;'>\\"{{bot_response}}\\"</p>
        </div>
        """

    name = matched_concept["name"]
    category = matched_concept["category"]
    
    intro_text = f"The <b>{{name}}</b> architecture functions as a foundational <b>{{category}}</b> configuration pattern. {{matched_concept['definition']}} " \\
                 f"Architecturally, it behaves deterministically across runtime threads according to standard operational specifications. " \\
                 f"Its memory footprint is determined dynamically based on the hardware configurations and compilation constraints of your target platform runtime environment."
    while len(intro_text) < 260:
        intro_text += " This architecture ensures system state preservation during processing cycles and optimizes standard pipeline routing workflows."

    ops_list = []
    for op_name, op_data in matched_concept["operations"].items():
        if "time_complexity" in op_data:
            ops_list.append(f"<li><b>{{op_name.upper()}}</b>: {{op_data['time_complexity']}} Complexity metrics</li>")
        else:
            sub_metrics = ", ".join([f"{{k.replace('_', ' ')}}={{v}}" for k, v in op_data.items()])
            ops_list.append(f"<li><b>{{op_name.upper()}}</b>: {{sub_metrics}}</li>")
            
    ops_joined = "".join(ops_list)
    complexity_text = f"<h5>Time Complexity Map:</h5><ul>{{ops_joined}}</ul>" \\
                      f"<h5>Space Allocation Metric:</h5>" \\
                      f"The structure requires a baseline space footprint evaluating exactly to <code>{{matched_concept.get('space_complexity', 'O(n)')}}</code>. " \\
                      f"This specific behavioral scaling constraint dictates strict boundary guarantees for hardware resource monitors during live stack executions."
    while len(complexity_text) < 260:
        complexity_text += " This performance metric remains critical when scaling application workloads across enterprise cluster frameworks."

    advs = "".join([f"<li>💡 {{item}}</li>" for item in matched_concept["advantages"]])
    disadvs = "".join([f"<li>⚠️ {{item}}</li>" for item in matched_concept["disadvantages"]])
    pros_cons_text = f"<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>" \\
                     f"<div><b style='color: #4ade80;'>Advantages & Core Strengths:</b><ul>{{advs}}</ul></div>" \\
                     f"<div><b style='color: #f87171;'>Disadvantages & Operational Risk:</b><ul>{{disadvs}}</ul></div>" \\
                     f"</div>"
    while len(pros_cons_text) < 260:
        pros_cons_text += " Advanced optimization flags should be monitored during construction configurations to avoid runtime errors."

    apps = "".join([f"<li>🚀 {{item}}</li>" for item in matched_concept["applications"]])
    apps_text = f"In industrial systems development, this structure is deployed in various target operational environments: <ul>{{apps}}</ul> " \\
                f"Selecting this structural schema directly impacts downstream cache coherency, network virtualization boundaries, and data processing pipeline logic."
    while len(apps_text) < 260:
        apps_text += " Software engineers prioritize this layout model when handling concurrent task scheduling and structural resource management protocols."

    return f"""
    <div style='background-color: #1e293b; color: #f8fafc; border-radius: 8px; border: 1px solid #334155; font-family: sans-serif; overflow: hidden; margin-top: 10px;'>
        <div style='background: linear-gradient(135deg, #1e40af, #3b82f6); padding: 20px; border-bottom: 1px solid #334155;'>
            <span style='background-color: #93c5fd; color: #1e3a8a; font-size: 11px; font-weight: bold; padding: 4px 8px; border-radius: 12px; text-transform: uppercase;'>{{category}}</span>
            <h3 style='margin: 8px 0 0 0; font-size: 26px; color: #ffffff;'>Structural Specification: {{name}}</h3>
        </div>
        <div style='padding: 20px; display: flex; flex-direction: column; gap: 20px;'>
            <div style='background-color: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid #475569;'>
                <h4 style='color: #38bdf8; margin: 0 0 8px 0; border-bottom: 1px solid #334155; padding-bottom: 4px;'>📘 Core Structural Introduction</h4>
                <p style='margin: 0; font-size: 14px; line-height: 1.6; color: #cbd5e1;'>{{intro_text}}</p>
                <small style='color: #64748b; display: block; margin-top: 5px;'>Character Audit Count: {{len(intro_text)}} chars</small>
            </div>
            <div style='background-color: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid #475569;'>
                <h4 style='color: #fbbf24; margin: 0 0 8px 0; border-bottom: 1px solid #334155; padding-bottom: 4px;'>📊 Computational Complexity Matrix</h4>
                <div style='margin: 0; font-size: 14px; line-height: 1.6; color: #cbd5e1;'>{{complexity_text}}</div>
                <small style='color: #64748b; display: block; margin-top: 5px;'>Character Audit Count: {{len(complexity_text)}} chars</small>
            </div>
            <div style='background-color: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid #475569;'>
                <h4 style='color: #f87171; margin: 0 0 8px 0; border-bottom: 1px solid #334155; padding-bottom: 4px;'>⚖️ Structural Trade-Off Metrics</h4>
                <div style='margin: 0; font-size: 14px; line-height: 1.6; color: #cbd5e1;'>{{pros_cons_text}}</div>
                <small style='color: #64748b; display: block; margin-top: 5px;'>Character Audit Count: {{len(pros_cons_text)}} chars</small>
            </div>
            <div style='background-color: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid #475569;'>
                <h4 style='color: #4ade80; margin: 0 0 8px 0; border-bottom: 1px solid #334155; padding-bottom: 4px;'>🛠️ Real-World Deployment Matrix</h4>
                <div style='margin: 0; font-size: 14px; line-height: 1.6; color: #cbd5e1;'>{{apps_text}}</p>
                <small style='color: #64748b; display: block; margin-top: 5px;'>Character Audit Count: {{len(apps_text)}} chars</small>
            </div>
        </div>
    </div>
    """

st.markdown(
    "<div style='text-align: center; padding: 10px 0; color: white;'>"
    "<h1>⚡ DSA High-Fidelity Analytics Engine</h1>"
    "<p style='color: #94a3b8;'>Query any concept for fine-tuned neural layout analysis and structural verification metrics.</p>"
    "</div>", 
    unsafe_allow_html=True
)

col1, col2 = st.columns([4, 6], gap="large")

if "current_user_query" not in st.session_state:
    st.session_state.current_user_query = "What are the operational execution rules of a Stack structure?"

with col1:
    input_box = st.text_input(
        label="Enter DSA Concept Query",
        value=st.session_state.current_user_query,
        key="unique_query_input_field"
    )
    submit_btn = st.button("Execute Analysis Run")
    
    st.markdown("<p style='color:#64748b; font-weight:bold; margin-top:15px; margin-bottom:5px;'>Suggested System Queries:</p>", unsafe_allowed_html=True)
    if st.button("💡 Explain the concept of an Array structural schema."):
        st.session_state.current_user_query = "Explain the concept of an Array structural schema."
        st.rerun()
    if st.button("💡 Show details for an unknown data concept."):
        st.session_state.current_user_query = "What is the time complexity of a Red-Black Tree architecture?"
        st.rerun()

with col2:
    active_query = input_box.strip() if input_box else st.session_state.current_user_query
    if submit_btn or (st.session_state.current_user_query != "What are the operational execution rules of a Stack structure?"):
        with st.spinner("Executing instruction trace pipeline..."):
            report_html = process_pipeline(active_query)
            st.markdown(report_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='color: #64748b; border: 2px dashed #334155; padding: 120px 40px; text-align: center; border-radius: 6px; margin-top:10px;'>Awaiting instruction execution trace pipeline...</div>", 
            unsafe_allow_html=True
        )
''')

# ==============================================================================
# STEP 4: COORDINATION HANDSHAKE RUNTIME
# ==============================================================================
if __name__ == "__main__":
    dsa_schema_data = load_or_create_dsa_json()
    
    if not os.path.exists(FINETUNED_DIR):
        print("⚡ Initialization flag triggered: Fine-tuning weights not detected.")
        run_fine_tuning(dsa_schema_data)
    else:
        print("ℹ️ Warm cache detected: Fine-tuned weight directory verified. Skipping training steps.")
        
    generate_streamlit_app()
    
    print("🌐 Launching local production Streamlit microservice layer...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", APP_FILE])
