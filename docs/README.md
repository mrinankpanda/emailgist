# EmailGist 📧✨

## Problem Statement

**The Problem**: Professionals are overwhelmed by lengthy emails containing critical information buried in verbose text. Long-form business communications, client updates, and project threads lead to missed deadlines, overlooked action items, and decision paralysis as people struggle to quickly identify what's important.

**The Solution**: EmailGist transforms lengthy emails into actionable insights using AI-powered summarization and intelligent highlighting.

**Target Users**: 
- Busy executives and managers dealing with email overload
- Project coordinators tracking multiple client communications
- Sales professionals managing prospect conversations
- Anyone who regularly receives lengthy, information-dense emails

## Why I'm Building This

I'm proving that **modern NLP can make email management effortless without sacrificing important context**. This project demonstrates:

- **Technical Mastery**: Full-stack AI application with production-ready deployment
- **Real Value Creation**: Solving a genuine productivity pain point experienced by millions
- **Scalable Architecture**: Building systems that can handle enterprise-level email volumes
- **User-Centric Design**: Creating intuitive interfaces that encourage daily adoption

This isn't just another summarization tool—it's a productivity multiplier that gives time back to professionals.

## Features

*   **AI-Powered Summarization**: The `/summarize` endpoint uses a HuggingFace Transformers model (DistilBART) to generate concise, meaningful summaries of long email texts.
*   **Intelligent Highlight Extraction**: Automatically pulls out key entities like dates, names, organizations, and action items using spaCy for advanced NER. A regex-based fallback ensures key terms are always found.
*   **Health Check Endpoints**: `GET /` and `GET /health` for easy service monitoring.
*   **Interactive API Docs**: FastAPI provides a Swagger UI at `/docs` for easy testing and exploration.

## Tech Stack

### Frontend
- **React** - Modern UI framework for building the user interface.
- **Vite** - High-performance build tool for frontend development.
- **Tailwind CSS** - A utility-first CSS framework for rapid styling.
- **Lucide-React** - Beautiful and consistent icons.

### Backend
*   **FastAPI** – Asynchronous Python web framework for building the API.
*   **Python 3.11** – The core language and runtime.
*   **Uvicorn** – High-performance ASGI server.
*   **HuggingFace Transformers** – State-of-the-art NLP library for summarization.
*   **spaCy** – Industrial-strength NLP for Named Entity Recognition (NER) and highlighting.
*   **Pydantic** – For robust data validation and settings management.

## How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/emailgist.git
cd emailgist
```

### 2. Set Up & Run the Backend

```bash
cd backend
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Run the API server
uvicorn main:app --reload
```
The backend will be running at `http://localhost:8000`.

### 3. Set Up & Run the Frontend

In a new terminal:
```bash
cd frontend
# Install dependencies
npm install
# Run the development server
npm run dev
```
The frontend will be running at `http://localhost:5173` and will connect to the backend automatically.


## API Endpoints

### `GET /`

Basic health message.

### `GET /health`

Full status report with service name and timestamp.

### `POST /summarize`

Accepts a JSON object with `email_content` and returns an AI-generated summary and a list of extracted highlights.

---

## Roadmap

| Week   | Goals                                                                    | Status |
| ------ | ------------------------------------------------------------------------ | ------ |
| Week 1 | Setup FastAPI server, define data models, build summarization endpoint | ✅ Done |
| Week 2 | Add NLP logic (BART summarization, spaCy NER)                         | ✅ Done |
| Week 3 | ⚙️ Refactor, improve performance, add caching                            | In Progress |
| Week 4 | 🚀 Deploy backend to Render + connect frontend (React)                   | In Progress |

---

## My Commitment

I'm building EmailGist to:

* ✅ Show that backend-focused projects can create real value
* ✅ Sharpen my skills in full-stack AI engineering
* ✅ Launch fast, iterate fast, and deliver utility from day one

Even without a frontend, the API is working—and that’s already saving time.

## License

Apache 2.0 License — see [LICENSE](LICENSE) file for details.
