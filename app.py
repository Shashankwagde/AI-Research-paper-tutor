"""
app.py - AI Research Tutor
Streamlit front-end with two tabs:
  1. Research Tutor  - single-paper RAG chat
  2. Compare Papers  - handled by compare_papers.py
"""

import pathlib
import streamlit as st

from rag_pipeline import (
    extract_text_from_pdf,
    chunk_text,
    create_vector_store,
    retrieve_relevant_chunks,
)
from openrouter_llm import generate_response
from compare_papers import render_compare_tab
from pricing_plans import get_pricing_html


# Page config
st.set_page_config(
    page_title="AI Research Tutor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(file_path):
    """Helper function to load external CSS into the Streamlit app."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load external CSS
load_css("style.css")

@st.dialog(" ", width="large")
def show_pricing_modal():
    st.html(get_pricing_html(st.session_state.active_plan))

# Session state defaults
DEFAULTS = {
    "messages":       [],
    "vector_store":   None,
    "chunks":         None,
    "paper_uploaded": False,
    "paper_name":     None,
    "chunk_count":    0,
    "page_count":     0,
    "active_plan":    "free",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Checkout logic
@st.dialog("Complete Your Purchase")
def show_checkout_modal(plan, billing="monthly"):
    plan_names = {"starter": "Starter", "pro": "Pro Researcher", "elite": "Elite Scholar"}
    
    monthly_prices = {"starter": "₹299/month", "pro": "₹799/month", "elite": "₹1499/month"}
    yearly_prices = {"starter": "₹2868/year (₹239/mo)", "pro": "₹7668/year (₹639/mo)", "elite": "₹14388/year (₹1199/mo)"}
    
    prices = yearly_prices if billing == "yearly" else monthly_prices
    
    st.markdown(f"### {plan_names.get(plan, 'Premium')}")
    st.markdown(f"**Billing Cycle:** {billing.capitalize()}")
    st.markdown(f"**Total due today:** {prices.get(plan, '₹0')}")
    st.markdown("---")
    st.info("This is a simulated checkout. No real payment will be processed.")
    
    if st.button("💳 Pay Now", type="primary", use_container_width=True):
        st.session_state.active_plan = plan
        st.query_params.clear()
        st.rerun()

    if st.button("Cancel", use_container_width=True):
        st.query_params.clear()
        st.rerun()

if "checkout" in st.query_params:
    checkout_plan = st.query_params["checkout"]
    billing = st.query_params.get("billing", "monthly")
    show_checkout_modal(checkout_plan, billing)

# Prompt builders
def build_summary_prompt(full_text: str) -> str:
    """Returns a structured summary prompt for the given paper text."""
    return f"""
You are an academic research assistant.

Based strictly on the following research paper content,
generate a detailed structured summary.

Include:
1. Research Problem (2-3 paragraphs)
2. Proposed Methodology (2-3 paragraphs)
3. Key Results and Findings (2-3 paragraphs)
4. Main Contributions (bullet points with explanation)
5. Limitations and Future Work (1-2 paragraphs)

Make it medium-length (approximately 400-500 words).
Do not add information not present in the content.

Paper Content:
{full_text}
"""

def build_rag_prompt(context: str, question: str) -> str:
    """Returns a RAG prompt with intent classification instructions."""
    return f"""
You are an Intelligent Research Assistant.

RESEARCH PAPER CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Determine if the question is a general greeting/question or specifically about the research paper context provided.
2. If it is a greeting or general question NOT related to the research paper, answer politely and briefly. DO NOT mention the paper.
3. If it is about the research paper, provide a detailed academic answer strictly based on the provided context.
4. CRITICAL: You MUST start your response with exactly one of these two tags:
   [GENERAL] - For greetings or general questions.
   [RESEARCH] - For questions answered using the research paper context.

Response:
"""



# Sidebar
with st.sidebar:

    st.markdown("""
        <div class="sidebar-logo">
            <div class="icon">🔬</div>
            <div>
                <div class="title">Research Tutor</div>
                <div class="subtitle">Powered by AI · RAG</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # PDF upload for the Tutor tab
    st.markdown("#### Upload Paper")

    uploaded_file = st.file_uploader(
        label="Drop your PDF here",
        type="pdf",
        label_visibility="collapsed",
    )

    if uploaded_file:
        with st.spinner("Processing paper..."):
            pages  = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(pages)

            if chunks:
                index = create_vector_store(chunks)

                st.session_state.vector_store   = index
                st.session_state.chunks         = chunks
                st.session_state.paper_uploaded = True
                st.session_state.paper_name     = uploaded_file.name
                st.session_state.chunk_count    = len(chunks)
                st.session_state.page_count     = len(pages)

        st.success("✅ Paper ready for analysis!")

    else:
        # Reset state when file is removed
        if st.session_state.paper_uploaded:
            st.session_state.vector_store   = None
            st.session_state.chunks         = None
            st.session_state.paper_uploaded = False
            st.session_state.paper_name     = None
            st.session_state.chunk_count    = 0
            st.session_state.page_count     = 0
            st.rerun()

    # Paper stats cards (visible after upload)
    if st.session_state.paper_uploaded:
        name       = st.session_state.paper_name or "Paper"
        short_name = name[:28] + ("…" if len(name) > 28 else "")

        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div>
                    <div class="stat-label">Loaded paper</div>
                    <div class="stat-value">{short_name}</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div>
                    <div class="stat-label">Pages extracted</div>
                    <div class="stat-value">{st.session_state.page_count} pages</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📦</div>
                <div>
                    <div class="stat-label">Text chunks indexed</div>
                    <div class="stat-value">{st.session_state.chunk_count} chunks</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Action buttons
    st.markdown("#### Actions")

    if st.button("Generate Summary", use_container_width=True):
        if not st.session_state.paper_uploaded:
            st.warning("⚠️ Upload a paper first.")
        else:
            with st.spinner("Generating structured summary..."):
                full_text = " ".join(
                    chunk["content"]
                    for chunk in st.session_state.chunks[:20]
                )
                prompt   = build_summary_prompt(full_text)
                summary  = generate_response(prompt, max_tokens=1200)
                response = "## 📄 Structured Paper Summary\n\n" + summary

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            st.success("Summary added to chat!")



    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Usage tips
    st.markdown("""
        <div style="font-size:0.78rem; color:#64748b; line-height:1.45;">
            <b style="color:#94a3b8">Tips</b><br>
            • Upload any academic PDF<br>
            • Ask conceptual questions<br>
            • Request methodology breakdowns<br>
            • Ask for result explanations<br>
            • Use <b style="color:#94a3b8">Compare Papers</b> tab for multi-paper analysis
        </div>
    """, unsafe_allow_html=True)


# Top-level Navigation (Acts like tabs)
nav_col1, nav_col2 = st.columns([8, 1])

with nav_col1:
    app_mode = st.radio(
        label="",
        options=["Research Tutor", "Compare Papers"],
        horizontal=True,
        label_visibility="collapsed"
    )

with nav_col2:
    if st.button("💎 Plans", use_container_width=True):
        show_pricing_modal()

# App Mode 1: single-paper RAG chat
if app_mode == "Research Tutor":

    status_pill = "🟢 Ready" if st.session_state.paper_uploaded else "⚪ No paper loaded"

    st.markdown(f"""
        <div class="hero-header">
            <div class="hero-blob"></div>
            <div class="hero-badge">
                <span class="dot"></span> AI Research Assistant
            </div>
            <h1>Intelligent Research Tutor</h1>
            <p>Upload any research paper and have a deep academic conversation powered by RAG + LLM.</p>
            <div class="feature-pills">
                <span class="pill">RAG-Grounded</span>
                <span class="pill">Semantic Search</span>
                <span class="pill">Auto Summary</span>
                <span class="pill">Chat Interface</span>
                <span class="pill">{status_pill}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create a dedicated container for chat history (height adapts dynamically)
    chat_container = st.container(border=False)

    # Show placeholder when no messages exist
    if not st.session_state.messages:
        with chat_container:
            st.markdown("""
                <div class="empty-state">
                    <div class="es-icon">🔬</div>
                    <p>No conversation yet. Upload a paper to start.</p>
                </div>
            """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        with chat_container.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask a question about the paper…"):

        if not st.session_state.paper_uploaded:
            st.warning("⚠️ Please upload a research paper first using the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Retrieve top-3 relevant chunks via semantic search
        retrieved = retrieve_relevant_chunks(
            query=prompt,
            index=st.session_state.vector_store,
            chunks=st.session_state.chunks,
            top_k=3,
        )

        context = "\n\n".join(
            f"(Page {item['page_number']}) {item['content']}"
            for item in retrieved
        )

        rag_prompt = build_rag_prompt(context, prompt)

        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(rag_prompt, max_tokens=500)
            
            # Parse the intent tag
            is_research = response.startswith("[RESEARCH]")
            # Remove the tags for a clean display
            clean_response = response.replace("[RESEARCH]", "").replace("[GENERAL]", "").strip()
            st.markdown(clean_response)

        st.session_state.messages.append({"role": "assistant", "content": clean_response})

        # Show source chunks ONLY if the answer was research-based
        if is_research:
            with chat_container.expander("View Retrieved Context Chunks"):
                for i, item in enumerate(retrieved):
                    col_badge, col_text = st.columns([1, 8])

                    with col_badge:
                        st.markdown(f"""
                            <div class="chunk-badge">
                                Rank {item['rank']}<br>Page {item['page_number']}
                            </div>
                        """, unsafe_allow_html=True)

                    with col_text:
                        st.markdown(item["content"])

                    if i < len(retrieved) - 1:
                        st.markdown("---")


# Tab 2: comparison — all logic in compare_papers.py
elif app_mode == "Compare Papers":
    render_compare_tab()
