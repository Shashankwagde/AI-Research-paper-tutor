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
    """
    Build a comprehensive summary prompt for research papers.
    Includes more text content and asks for detailed sections.
    """
    return f"""You are an academic research paper analyzer. Provide a comprehensive and detailed summary of the following research paper.

Instructions:
1. Research Problem: Clearly state the main research problem or question being addressed. Explain why this problem is important and what gap it fills in the existing literature.
2. Methodology: Describe in detail the research methods, approaches, and techniques used to solve the problem. Include information about the dataset, experimental setup, algorithms, or theoretical framework.
3. Key Findings: Summarize the major results and contributions. Include specific numbers, metrics, or comparisons if available.
4. Implications: Discuss the broader implications of the findings and potential applications.
5. Limitations: Note any limitations or weaknesses identified in the study.
6. Conclusion: Provide a brief conclusion and future work suggestions.

Paper content:
{full_text[:5000]}

Please provide a thorough, well-structured summary covering all the sections above."""

def build_rag_prompt(context: str, question: str) -> str:
    return f"Context: {context}\n\nQuestion: {question}\n\nAnswer based on context:"


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
            with st.spinner("Generating summary..."):
                # Use more chunks for comprehensive content (up to 5 chunks)
                full_text = " ".join(
                    chunk["content"]
                    for chunk in st.session_state.chunks[:5]
                )
                prompt = build_summary_prompt(full_text)
                summary = generate_response(prompt, max_tokens=1500)
                response = "## 📄 Paper Summary\n\n" + summary

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
            • Use <b style="color:#94a3b8">Compare Papers</b> tab
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
                <span class="pill">{status_pill}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create a dedicated container for chat history
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

        # Retrieve top-1 relevant chunk only (for token limits)
        retrieved = retrieve_relevant_chunks(
            query=prompt,
            index=st.session_state.vector_store,
            chunks=st.session_state.chunks,
            top_k=1,
        )

        # Limit context to 400 chars
        context = f"Page {retrieved[0]['page_number']}: {retrieved[0]['content'][:400]}"

        rag_prompt = build_rag_prompt(context, prompt)

        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(rag_prompt, max_tokens=100)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# Tab 2: comparison
elif app_mode == "Compare Papers":
    render_compare_tab()
