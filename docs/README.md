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

* **Summarization API**: `/summarize` endpoint that generates brief, useful summaries of email text
* **Highlight Extraction**: Returns dummy highlights (soon powered by spaCy)
* **Health Check Endpoint**: Quickly verify server status and uptime
* **FastAPI Docs**: Auto-generated Swagger UI for easy API testing

## Tech Stack

### Frontend
- **React** - Modern UI framework
- **Tailwind CSS** - Utility-first CSS framework for styling

### Backend

* **FastAPI** – Async Python web framework
* **Python 3.11** – Primary language/runtime
* **Uvicorn** – ASGI server for running the app

### NLP Tools (Planned/Stubbed)

* **HuggingFace Transformers** – Summarization (coming soon)
* **spaCy** – NER for highlights (next phase)
* **Regex** – Lightweight entity matching

## How to Use It (Backend Only)

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/emailgist.git
cd emailgist/backend
```

### 2. Set Up Python Environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Run the API Server

```bash
uvicorn main:app --reload
```

### 4. Visit Swagger Docs

[http://localhost:8000/docs](http://localhost:8000/docs)

Try the `/summarize` endpoint with JSON input like:

```json
{
  "email_content": "Hi team, please finalize the budget by Friday. We have a client review on Monday."
}
```

## API Endpoints

### `GET /`

Basic health message.

### `GET /health`

Full status report with service name and timestamp.

### `POST /summarize`

Returns a dummy summary, highlights, and processing time.

---

## Roadmap (Backend-First Approach)

| Week   | Goals                                                                    |
| ------ | ------------------------------------------------------------------------ |
| Week 1 | ✅ Setup FastAPI server, define data models, build summarization endpoint |
| Week 2 | 🛠 Add NLP logic (BART summarization, spaCy NER)                         |
| Week 3 | ⚙️ Refactor, improve performance, add caching                            |
| Week 4 | 🚀 Deploy backend to Render + connect frontend (React)                   |

---

## My Commitment

I'm building EmailGist to:

* ✅ Show that backend-focused projects can create real value
* ✅ Sharpen my skills in full-stack AI engineering
* ✅ Launch fast, iterate fast, and deliver utility from day one

Even without a frontend, the API is working—and that’s already saving time.

## License

Apache 2.0 License — see [LICENSE](LICENSE) file for details.
