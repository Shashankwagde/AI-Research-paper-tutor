"""
compare_papers.py - Research Paper Comparison Module
All comparison logic and UI for the "Compare Papers" tab.
Entry point: render_compare_tab() called from app.py.
"""

import streamlit as st

from rag_pipeline import extract_text_from_pdf, chunk_text
from openrouter_llm import generate_response


# Prompt: structured summary for a single paper
def _build_summary_prompt(full_text: str) -> str:
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


# Prompt: 6-section academic comparison of two summaries
def _build_comparison_prompt(summary_a: str, summary_b: str,
                              name_a: str, name_b: str) -> str:
    return f"""
You are an expert academic research analyst.

Below are structured summaries of two research papers.
Analyze them carefully and produce a structured comparison report.

Your comparison MUST cover all of the following sections, in order,
using the exact section headings shown:

## 1. Research Problem Comparison
## 2. Methodology Comparison
## 3. Results & Evaluation Comparison
## 4. Contributions Comparison
## 5. Limitations & Future Work
## 6. Overall Comparison Summary

Rules:
- Base your analysis ONLY on the provided summaries.
- Do not hallucinate or infer information not present in the summaries.
- Be precise, academic, and analytical in tone.
- Each section should be 2-4 paragraphs.

---
### Summary of Paper A: {name_a}
{summary_a}

---
### Summary of Paper B: {name_b}
{summary_b}
---
"""


# Extract, chunk, and summarise one uploaded PDF
def _process_paper(uploaded_file) -> tuple[str, int, int]:
    """Returns (summary, page_count, chunk_count). Raises ValueError if no text found."""
    pages  = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(pages)

    if not chunks:
        raise ValueError(
            f"No usable text could be extracted from '{uploaded_file.name}'."
        )

    full_text = " ".join(chunk["content"] for chunk in chunks[:20])
    summary   = generate_response(
        _build_summary_prompt(full_text),
        max_tokens=800,
    )

    return summary, len(pages), len(chunks)


# Render a paper meta card (name, pages, chunks)
def _render_meta_card(name: str, pages: int, chunks: int, badge_class: str) -> None:
    short_name = name[:50] + ("…" if len(name) > 50 else "")
    st.markdown(f"""
        <div class="cmp-meta-card cmp-meta-{badge_class}">
            <div class="cmp-meta-title">
                <span class="cmp-label-badge cmp-badge-{badge_class}">
                    {"A" if badge_class == "a" else "B"}
                </span>
                {short_name}
            </div>
            <div class="cmp-meta-stats">
                {pages} pages &nbsp;&middot;&nbsp; {chunks} chunks
            </div>
        </div>
    """, unsafe_allow_html=True)


# Render the full comparison result (meta cards, report, summaries, clear button)
def _render_comparison_result(result: dict) -> None:
    col_a, _, col_b = st.columns([5, 0.3, 5])
    with col_a:
        _render_meta_card(result["name_a"], result["pages_a"], result["chunks_a"], "a")
    with col_b:
        _render_meta_card(result["name_b"], result["pages_b"], result["chunks_b"], "b")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="cmp-result-header">Comparison Report</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="cmp-result-body">{result["comparison"]}</div>',
        unsafe_allow_html=True,
    )

    # Collapsible individual summaries
    with st.expander(f"View Summary - Paper A: {result['name_a']}"):
        st.markdown(result["summary_a"])

    with st.expander(f"View Summary - Paper B: {result['name_b']}"):
        st.markdown(result["summary_b"])

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    if st.button("Clear Comparison", key="clear_cmp"):
        st.session_state.cmp_result = None
        st.rerun()


# Public entry point — called by app.py inside `with tab_compare:`
def render_compare_tab() -> None:
    # Initialise result store
    if "cmp_result" not in st.session_state:
        st.session_state.cmp_result = None

    # Header
    st.markdown("""
        <div class="cmp-header">
            <div class="hero-blob"></div>
            <div class="hero-badge">
                <span class="dot"></span>&nbsp;Multi-Paper Analysis
            </div>
            <h1>Research Paper Comparison</h1>
            <p>
                Upload two research papers and get an academically structured,
                LLM-generated comparison covering methodology, results,
                contributions, and more.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Two upload columns with a VS divider
    col_a, col_sep, col_b = st.columns([5, 0.3, 5])

    with col_a:
        st.markdown("""
            <div class="cmp-upload-label">
                <span class="cmp-label-badge cmp-badge-a">Paper A</span>
            </div>
        """, unsafe_allow_html=True)

        file_a = st.file_uploader(
            label="Upload Paper A",
            type="pdf",
            key="cmp_file_a",
            label_visibility="collapsed",
        )

        if file_a:
            st.markdown(f"""
                <div class="cmp-file-info">
                    <span class="cmp-file-name">{file_a.name}</span>
                </div>
            """, unsafe_allow_html=True)

    with col_sep:
        st.markdown("""
            <div class="cmp-vs-divider">
                <div class="cmp-vs-line"></div>
                <div class="cmp-vs-badge">VS</div>
                <div class="cmp-vs-line"></div>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
            <div class="cmp-upload-label">
                <span class="cmp-label-badge cmp-badge-b">Paper B</span>
            </div>
        """, unsafe_allow_html=True)

        file_b = st.file_uploader(
            label="Upload Paper B",
            type="pdf",
            key="cmp_file_b",
            label_visibility="collapsed",
        )

        if file_b:
            st.markdown(f"""
                <div class="cmp-file-info">
                    <span class="cmp-file-name">{file_b.name}</span>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    # Centred compare button
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        compare_clicked = st.button(
            "Compare Papers",
            use_container_width=True,
            key="compare_btn_main",
            type="primary",
        )

    # Features / What's compared section to utilise space
    st.markdown("""
        <div style="margin-top: 3rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 2.5rem;">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h4 style="color: var(--text-muted); font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase;">Academic Comparison Framework</h4>
            </div>
            <div style="display: flex; justify-content: center; gap: 4rem; flex-wrap: wrap;">
                <div style="text-align: center; max-width: 180px; background: rgba(255,255,255,0.02); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 2rem; margin-bottom: 0.8rem;">🎯</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: #a5b4fc; margin-bottom: 0.4rem;">Objectives</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.4;">Compare core research problems & goals</div>
                </div>
                <div style="text-align: center; max-width: 180px; background: rgba(255,255,255,0.02); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 2rem; margin-bottom: 0.8rem;">⚙️</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: #67e8f9; margin-bottom: 0.4rem;">Methodology</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.4;">Analyze procedural & technical shifts</div>
                </div>
                <div style="text-align: center; max-width: 180px; background: rgba(255,255,255,0.02); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 2rem; margin-bottom: 0.8rem;">📊</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: #818cf8; margin-bottom: 0.4rem;">Key Impact</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.4;">Evaluate findings, results & outcomes</div>
                </div>
                <div style="text-align: center; max-width: 180px; background: rgba(255,255,255,0.02); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 2rem; margin-bottom: 0.8rem;">💡</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: #c084fc; margin-bottom: 0.4rem;">Innovation</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.4;">Identify unique academic contributions</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # Run comparison when button is clicked
    if compare_clicked:

        if not file_a and not file_b:
            st.warning("⚠️ Please upload both Paper A and Paper B before comparing.")
            return

        if not file_a:
            st.warning("⚠️ Paper A is missing. Please upload a PDF for Paper A.")
            return

        if not file_b:
            st.warning("⚠️ Paper B is missing. Please upload a PDF for Paper B.")
            return

        try:
            with st.spinner(f"Reading and summarising **{file_a.name}**..."):
                summary_a, pages_a, chunks_a = _process_paper(file_a)

            with st.spinner(f"Reading and summarising **{file_b.name}**..."):
                summary_b, pages_b, chunks_b = _process_paper(file_b)

            with st.spinner("Generating structured comparison..."):
                cmp_prompt = _build_comparison_prompt(
                    summary_a, summary_b,
                    file_a.name, file_b.name,
                )
                comparison = generate_response(cmp_prompt, max_tokens=1200)

            st.session_state.cmp_result = {
                "comparison": comparison,
                "name_a":     file_a.name,
                "name_b":     file_b.name,
                "pages_a":    pages_a,
                "pages_b":    pages_b,
                "chunks_a":   chunks_a,
                "chunks_b":   chunks_b,
                "summary_a":  summary_a,
                "summary_b":  summary_b,
            }

        except ValueError as exc:
            st.error(f"❌ Extraction error: {exc}")
            return

        except Exception as exc:
            st.error(f"❌ An unexpected error occurred: {exc}")
            return

    # Display the stored comparison result (persists across reruns)
    if st.session_state.cmp_result:
        _render_comparison_result(st.session_state.cmp_result)
