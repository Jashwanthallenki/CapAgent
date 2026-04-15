# 🚀 CapAgent: Controllable Image Captioning Agent

CapAgent is an **agent-based image captioning system** that transforms simple user instructions into **professional, detailed, and context-aware captions**.

It bridges the gap between **vague user queries** and **high-quality outputs** by combining:
- Query refinement
- Tool-based reasoning
- Multimodal understanding

---

## 📌 Motivation

Users often provide minimal instructions like:

> "Describe this image"

But expect:
- Rich descriptions  
- Context awareness  
- Spatial relationships  
- High semantic quality  

Traditional captioning models struggle with this mismatch.

👉 **CapAgent solves this by automatically refining user intent and orchestrating expert tools.**

---

## 🧠 Key Features

- ✨ **Automatic Instruction Refinement**  
  Converts simple queries into structured, professional prompts  

- 🧩 **Agent-Based Architecture**  
  Planning → Tool Usage → Observation  

- 🛠️ **Tool-Augmented Reasoning**  
  Uses specialized vision models for better understanding  

- 🔄 **RAG-based Reasoning Support**  
  Retrieves past reasoning examples to guide decisions  

- ⚙️ **Modular & Extensible**  
  Easily plug in new tools and models  

---

## 🏗️ System Architecture

CapAgent follows a **three-stage agent pipeline**:

### 1. Planning
- Understands user query + image  
- Breaks task into sub-steps  

### 2. Tool Usage
- Dynamically invokes expert models:
  - Object detection  
  - Depth estimation  
  - External search (optional)  

### 3. Observation
- Aggregates outputs from tools  
- Generates final caption using an LLM  

---

## 🧭 Architecture Diagram

```text
            ┌────────────────────┐
            │     User Input     │
            │ (Image + Query)   │
            └─────────┬──────────┘
                      │
                      ▼
            ┌────────────────────┐
            │      Planning      │
            │ (LLM Reasoning)    │
            └─────────┬──────────┘
                      │
                      ▼
            ┌────────────────────┐
            │    Tool Usage      │
            │ ┌───────────────┐  │
            │ │ GroundingDINO │  │
            │ │ Depth Model   │  │
            │ │ Search APIs   │  │
            │ └───────────────┘  │
            └─────────┬──────────┘
                      │
                      ▼
            ┌────────────────────┐
            │    Observation     │
            │ (Aggregate Info)   │
            └─────────┬──────────┘
                      │
                      ▼
            ┌────────────────────┐
            │   Final Caption    │
            └────────────────────┘
```
🔧 Tools & Models Used
GroundingDINO → Object detection & localization
Depth-Anything-V2 → Depth estimation
LLMs (GPT-4o or similar) → Planning & caption generation
Google Search / Lens (Optional) → External knowledge
🔄 Workflow Example

Input:

User: "Describe this image"

Process:

Plan → detect objects, scene, depth
Call tools → extract structured visual info
Refine instruction
Generate caption

Output:

"A brown dog playing with a ball in the foreground of a park, with trees and people in the background."

1. Clone Repository
git clone <repo-url>
cd CapAgent
2. Create Environment
conda create -n capagent python=3.10
conda activate capagent
pip install -r requirements.txt
3. Set API Keys
export OPENAI_API_KEY=<your-openai-api-key>
export SERP_API_KEY=<your-serp-api-key>  # optional

## 🧪 Install Expert Models

### GroundingDINO
cd expert_models  
git clone https://github.com/IDEA-Research/GroundingDINO.git  
cd GroundingDINO  

conda create -n groundingdino python=3.10  
conda activate groundingdino  
pip install -e .  

### Depth-Anything-V2
cd ../  
git clone https://github.com/DepthAnything/Depth-Anything-V2.git  
cd Depth-Anything-V2  

conda create -n depthanything python=3.10  
conda activate depthanything  
pip install -e .  

python app.py  

---

## ▶️ Running the Project

### Start Image Server
python launch_image_server.py  

### Run Inference
python run.py  

### Gradio Demo
python gradio_demo.py  

---

## 🧠 RAG Setup

Generate embeddings for Chain-of-Thought examples:  

bash init_rag_database.sh  

---

## 📊 Advantages

- High-quality captions  
- Better reasoning using tools  
- Modular and scalable  
- Improved controllability  

---

## ⚠️ Limitations

- Higher latency due to multiple tool calls  
- Increased system complexity  
- Higher cost (LLM + APIs + models)  
- Error propagation across tools  

---

## 🔮 Future Work

- Add more expert tools (segmentation, OCR, etc.)  
- Improve reasoning with advanced agent frameworks  
- Optimize latency and cost  
- Extend to video and multi-image understanding  

---

## 🙌 Acknowledgements

- GroundingDINO  
- Depth-Anything-V2  
- VisualSketchPad  
