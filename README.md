<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Google_Gemini-2.5_Flash-8E8E8E?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🔬 AI Research Tutor</h1>

<p align="center">
  <strong>An intelligent, AI-powered research assistant that helps you summarize, analyze, and interact with academic research papers using Retrieval-Augmented Generation (RAG).</strong>
</p>

<p align="center">
  Upload any research PDF → Get structured summaries → Ask deep questions → Compare multiple papers side-by-side
</p>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the App](#running-the-app)
- [🖥️ Usage Guide](#️-usage-guide)
- [💎 Pricing Plans](#-pricing-plans)
- [🧠 How RAG Works (Under the Hood)](#-how-rag-works-under-the-hood)
- [🎨 UI & Design](#-ui--design)
- [🛠️ API Reference](#️-api-reference)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

### 🔍 Single Paper Analysis (Research Tutor Tab)
| Feature | Description |
|---|---|
| **PDF Upload & Parsing** | Upload any academic PDF. Text is extracted page-by-page using `pypdf`, cleaned of references/URLs/DOIs, and split into semantically meaningful chunks. |
| **AI-Powered Summaries** | Generate structured summaries covering Research Problem, Methodology, Key Results, Contributions, and Limitations — all grounded strictly in the paper content. |
| **RAG-Powered Q&A Chat** | Ask natural-language questions about your paper. The system retrieves the top-k most relevant chunks via FAISS semantic search, then generates a grounded answer using Google Gemini 2.5 Flash. |
| **Source Attribution** | Every AI response includes the page number from which context was retrieved, showing exactly which sections of the paper were used. |
| **Session Persistence** | Your chat history, uploaded paper state, and vector index persist across Streamlit reruns within the same session. |

### 📊 Multi-Paper Comparison (Compare Papers Tab)
| Feature | Description |
|---|---|
| **Side-by-Side Upload** | Upload two research PDFs simultaneously with a clean Paper A vs Paper B interface. |
| **Structured Comparison Report** | Automatically generates an academic comparison covering key aspects of both papers. |
| **Individual Summaries** | Each paper's full structured summary is available in collapsible expanders for reference. |
| **Paper Metadata Cards** | Visual cards showing filename, page count, and chunk count for each paper. |

### 💎 Subscription Plans (Pricing Modal)
| Feature | Description |
|---|---|
| **3-Tier Pricing** | Starter (₹299/mo), Pro Researcher (₹799/mo), and Elite Scholar (₹1499/mo) with distinct feature sets. |
| **Dynamic Billing Toggle** | Pure CSS-powered Monthly ↔ Yearly toggle with instant 20% discount preview — no JavaScript required. |
| **Simulated Checkout** | Click any plan's CTA → a checkout modal appears → click "Pay Now" ��� plan is activated in session state. |
| **Dynamic Plan State** | After purchasing, the active plan's button changes to "Current Plan" (greyed out) across the pricing UI. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Research      │  │ Compare      │  │ Pricing Plans     │  │
│  │ Tutor Tab     │  │ Papers Tab   │  │ Modal (st.dialog) │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────┘  │
│         │                 │                                   │
│  ┌──────▼─────────────────▼──────┐                           │
│  │       RAG Pipeline            │                           │
│  │  ┌─────────┐ ┌──────────────┐ │                           │
│  │  │ pypdf   │ │ Sentence     │ │                           │
│  │  │ Extract │ │ Transformers │ │                           │
│  │  └────┬────┘ └──────┬───────┘ │                           │
│  │       │             │         │                           │
│  │  ┌────▼─────────────▼──────┐  │                           │
│  │  │    FAISS Vector Index   │  │                           │
│  │  └─────────────────────────┘  │                           │
│  └───────────────┬───────────────┘                           │
│                  │                                            │
│  ┌───────────────▼───────────────┐                           │
│  │   Google Gemini API          │                           │
│  │   (Gemini 2.5 Flash)       │                           │
│  └───────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
📦 AI-Research-Tutor/
├── 📄 app.py                 # Main Streamlit application entry point
│                              # - Page config, session state, sidebar, navigation
│                              # - Research Tutor tab (upload, summary, RAG chat)
│                              # - Pricing modal & checkout dialog logic
│
├── 📄 rag_pipeline.py         # Core RAG engine
│                              # - PDF text extraction (pypdf)
│                              # - Text cleaning (regex-based)
│                              # - Chunking (350-word sliding window)
│                              # - Embedding (all-MiniLM-L6-v2)
│                              # - FAISS indexing & semantic retrieval
│
├── 📄 openrouter_llm.py       # LLM integration layer
│                              # - Google Gemini API client
│                              # - Gemini 2.5 Flash model configuration
│                              # - System prompt for academic responses
│
├���─ ���� compare_papers.py       # Multi-paper comparison module
│                              # - Dual PDF upload UI
│                              # - Parallel summarization pipeline
│                              # - Structured comparison report
│
├── 📄 pricing_plans.py        # Pricing UI generator
│                              # - Dynamic HTML for 3 subscription tiers
│                              # - CSS-only billing toggle (Monthly/Yearly)
│                              # - Active plan detection & button state
│
├── 🎨 style.css               # Complete design system (~920 lines)
│                              # - CSS variables & dark theme tokens
│                              # - Glassmorphism card styles
│                              # - Hero section with gradient blob
│                              # - Responsive breakpoints
│                              # - Pricing modal & toggle styles
│                              # - Streamlit component overrides
│
├── 📄 requirements.txt        # Python dependencies
├── 📄 .env                    # Environment variables (API keys)
├── 📄 .gitignore              # Git ignore rules
└── 📄 README.md               # This file
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.36+ | Web UI framework with dialog modals, chat interface, and reactive state |
| **PDF Parsing** | pypdf | Extract text content page-by-page from uploaded PDFs |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | Convert text chunks into 384-dimensional semantic vectors |
| **Vector Search** | FAISS (faiss-cpu) | Blazing-fast approximate nearest neighbor search for chunk retrieval |
| **LLM** | Google Gemini API → Gemini 2.5 Flash | Generate structured summaries, answer questions, and compare papers |
| **Styling** | Custom CSS | Premium dark-mode glassmorphism theme with responsive design |
| **Environment** | python-dotenv | Secure API key management via `.env` file |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed on your system
- A **Google Gemini API key** (get one at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-research-tutor.git
   cd ai-research-tutor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the project root (or edit the existing one):

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

> ⚠️ **Important:** Never commit your `.env` file to version control. It is already included in `.gitignore`.

### Running the App

```bash
streamlit run app.py
```

The app will launch at **http://localhost:8501** in your default browser.

---

## 🖥️ Usage Guide

### 1. Upload a Research Paper

1. Open the **sidebar** (click the `>` arrow if collapsed).
2. Click **"Browse files"** under "Upload Paper" and select any academic PDF.
3. Wait for processing — the app will extract text, create chunks, and build a vector index.
4. You'll see paper stats (filename, pages, chunks) appear in the sidebar.

### 2. Generate a Summary

1. Click **"Generate Summary"** in the sidebar's Actions section.
2. The AI will produce a structured summary covering:
   - Research Problem
   - Proposed Methodology
   - Key Results and Findings
   - Main Contributions
   - Limitations and Future Work
3. The summary appears in the chat area.

### 3. Ask Questions (RAG Chat)

1. Type any question in the chat input at the bottom, e.g.:
   - *"What methodology did the authors use?"*
   - *"Explain the main findings in simple terms."*
   - *"What are the limitations of this study?"*
2. The system performs semantic search across all chunks, retrieves the most relevant chunk, and generates a grounded answer.

### 4. Compare Two Papers

1. Switch to the **"Compare Papers"** tab in the top navigation.
2. Upload **Paper A** and **Paper B** using the side-by-side uploaders.
3. Click **"Compare Papers"** — the system will:
   - Summarize each paper independently
   - Generate a structured comparison report
4. Expand individual summaries for deeper reference.

### 5. View Pricing Plans

1. Click the **"💎 Plans"** button in the top-right corner.
2. Browse three subscription tiers with feature lists.
3. Toggle between **Monthly** and **Yearly** billing (20% discount on yearly).
4. Click a plan's CTA button to open the checkout modal.
5. Click **"💳 Pay Now"** to simulate a purchase — the plan activates instantly.

---

## 💎 Pricing Plans

| Feature | Starter (₹299/mo) | Pro Researcher (₹799/mo) | Elite Scholar (₹1499/mo) |
|---|:---:|:---:|:---:|
| AI Summary Model | Basic | Advanced | Premium |
| Summaries/Month | 20 | 100 | Unlimited |
| Processing Speed | Standard | Faster | Fastest |
| Save History | ✅ | ✅ | ✅ |
| Key Insight Extraction | ❌ | ✅ | ✅ |
| Chat with Papers | ❌ | ✅ | Unlimited |
| Premium Paper Library | Limited | ✅ | Full Access |
| Export (PDF/DOCX) | ❌ | ❌ | ✅ |
| Support | Community | Priority Email | Dedicated |
| Early Access Features | ❌ | ❌ | ✅ |

> **Note:** The checkout flow is simulated — no real payment gateway is integrated. Plans are stored in Streamlit session state and reset when the session ends.

---

## 🧠 How RAG Works (Under the Hood)

```
User uploads PDF
       │
       ▼
┌──────────────────┐
│  1. EXTRACT       │  pypdf reads each page
│     Text          │  Regex cleans refs, URLs, DOIs
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  2. CHUNK         │  Split into 350-word windows
│     Text          │  Filter out tiny fragments (<150 chars)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  3. EMBED         │  all-MiniLM-L6-v2 encodes each chunk
│     Chunks        │  into a 384-dim dense vector
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  4. INDEX         │  FAISS IndexFlatL2 stores all vectors
│     in FAISS      │  for exact nearest-neighbor search
└──────────────────┘

User asks a question
       │
       ▼
┌──────────────────┐
│  5. RETRIEVE      │  Encode query → search FAISS → top-k chunks
│     Chunks        │  returned with page numbers & distances
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  6. GENERATE      │  Chunks become context in the prompt
│     Answer        │  Gemini 2.5 Flash generates grounded response
└──────────────────┘
```

**Key Design Decisions:**
- **Chunk size of 350 words** balances semantic completeness with embedding quality
- **L2 distance** (Euclidean) is used for FAISS similarity — lower is more similar
- **Top-1 retrieval** for chat (or top-3 for summaries) provides sufficient context without overwhelming the LLM context window
- **Chunk truncation at 800 chars** prevents excessively long context from consuming tokens
- **Temperature 0.3** keeps responses factual and conservative

---

## 🎨 UI & Design

The application features a **premium dark-mode interface** built with custom CSS:

- **Glassmorphism** — Cards with translucent backgrounds and subtle borders
- **Gradient Accents** — Indigo-to-violet gradients for interactive elements
- **Animated Hero Blob** — Floating gradient orb in the header section
- **Hover Micro-animations** — Cards lift and glow on hover with smooth transitions
- **Responsive Layout** — Fully adapts to mobile, tablet, and desktop viewports
- **Custom Streamlit Overrides** — Sidebar, dialogs, buttons, and chat bubbles all themed consistently
- **Typography** — Space Grotesk for headings, Inter for body text (via Google Fonts)

---

## 🛠️ API Reference

### `rag_pipeline.py`

| Function | Parameters | Returns | Description |
|---|---|---|---|---|
| `extract_text_from_pdf(uploaded_file)` | Streamlit `UploadedFile` | `list[dict]` with `page_number` and `text` | Extracts and cleans text from each PDF page |
| `chunk_text(pages, chunk_size=350)` | List of page dicts, optional chunk size | `list[dict]` with `page_number` and `content` | Splits page text into overlapping chunks |
| `create_vector_store(chunks)` | List of chunk dicts | `faiss.IndexFlatL2` | Creates FAISS index from chunk embeddings |
| `retrieve_relevant_chunks(query, index, chunks, top_k=3)` | Query string, FAISS index, chunks list, k | `list[dict]` with rank, content, page, distance | Retrieves top-k semantically similar chunks |

### `openrouter_llm.py`

| Function | Parameters | Returns | Description |
|---|---|---|---|---|
| `generate_response(prompt, max_tokens=300)` | Prompt string, optional token limit | `str` | Sends prompt to Gemini 2.5 Flash and returns the response |

### `pricing_plans.py`

| Function | Parameters | Returns | Description |
|---|---|---|---|---|
| `get_pricing_html(active_plan="free")` | Current active plan ID | `str` (HTML) | Returns the full pricing grid HTML with dynamic button states |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines
- Follow existing code style and commenting conventions
- Keep Streamlit components and custom HTML/CSS separated
- Test with multiple PDF types (single-column, double-column, scanned)
- Ensure responsive design works on all viewports

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using Streamlit, FAISS, and Google Gemini
</p>
