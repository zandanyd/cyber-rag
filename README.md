<img width="680" height="235" alt="image" src="https://github.com/user-attachments/assets/fc0825da-9e5c-4848-acd9-ba6012fe0766" />
# 🛡️ Cyber RAG

Cyber RAG is a command-line tool that automatically answers predefined cybersecurity questions from blog posts to help analysts quickly analyze and summarize threat intelligence articles.
It uses Retrieval-Augmented Generation (RAG) to combine blog content with small and efficient language models (~4.7B), delivering accurate answers without relying on large, expensive LLMs (70B).

---

## ✨ Features

- 🔍 **🌐 Blog Text Extraction** : Automatically extracts clean text from any cybersecurity blog URL
- ❓ ** Question Answering with RAG**: Answers predefined cybersecurity questions based on the blog content using Retrieval-Augmented Generation (RAG).
- 💬 **Supports Small, Fast Models:**: Supports lightweight models like mistralai/mistral-small-3-1-24b-instruct-2503, which can run locally using Ollama, or remotely via WatsonX — no need for large or expensive LLMs.
---
## 🚀 Quick Start

```bash
git clone https://github.com/your-username/cyber-rag.git
cd cyber-rag
python run.py
```

## 📦 Installation and Configuration
1. Install Python 3.11 or later  👉 https://www.python.org/downloads/
2. (Optional) Install uv for dependency management 👉 https://astral.sh/uv/
  - For Linux and MacOS, you can use the following command:
      ```bash
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```
   - For Windows, you can use the following command:
      ```bash
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      ```
3. Install dependencies:
 ```bash
      pip install -r requirements.txt
 ```
4. Edit the .env file and choose your preferred LLM provider:
  - If using WatsonX Fill in the required fields
    You can get your credentials at 👉 https://dataplatform.cloud.ibm.com
  - If using Ollama: Make sure Ollama is installed and running locally.
    Start the Mistral model with:
     ```bash
      ollama run mistral
      ```
    No API keys are needed — Cyber RAG will connect to Ollama automatically via http://localhost:11434.

    



