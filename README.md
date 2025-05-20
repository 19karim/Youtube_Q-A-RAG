# YouTube Q&A RAG Tool 🎥🤖

Here’s how the Q&A tool looks in action:
![YouTube Q&A RAG Screenshot](assets/screenshot.png)

This tool leverages **OpenAI’s language models**, **vector search**, and **YouTube transcript analysis** to make videos more **accessible**, **searchable**, and **interactive**—without watching the entire video.

---

## 🚀 Objective

Create an intelligent Q&A assistant for YouTube videos that enables users to query content based on its transcript and get meaningful answers instantly.

---

## 🎯 Key Features

- 🔎 Accepts **YouTube video IDs** as input  
- 🧠 Answers questions based on the video **transcript** using **Retrieval-Augmented Generation (RAG)**
- 💬 **Saves and displays** all questions asked during a session
- 📄 Allows **exporting the full Q&A chat as a PDF**

---

## 🛠️ Technologies Used

- **Python**
- **OpenAI API** (for language understanding)
- **LangChain** or custom RAG pipeline
- **FAISS** or other vector store (for similarity search)
- **YouTube Transcript API**
- **PDF generation** (e.g., `fpdf`, `reportlab`, or `pdfkit`)

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/19karim/Youtube_Q-A-RAG.git
cd Youtube_Q-A-RAG

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # or manually create .env
