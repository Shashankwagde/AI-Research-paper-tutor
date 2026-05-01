"""
compare_papers.py - Research Paper Comparison Module
All comparison logic and UI for the "Compare Papers" tab.
Entry point: render_compare_tab() called from app.py.
"""

import streamlit as st

from rag_pipeline import extract_text_from_pdf, chunk_text
from openrouter_llm import generate_response


# Prompt: summary with structure
def _build_summary_prompt(full_text: str) -> str:
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
{full_text[:2500]}

Please provide a thorough, well-structured summary covering all the sections above."""


# Prompt: comparison
def _build_comparison_prompt(summary_a: str, summary_b: str, name_a: str, name_b: str) -> str:
    return f"Compare paper A ({name_a}) vs B ({name_b}). A: {summary_a[:200]} B: {summary_b[:200]}"


# Extract, chunk, and summarise one uploaded PDF
def _process_paper(uploaded_file) -> tuple[str, int, int]:
    """Returns (summary, page_count, chunk_count). Raises ValueError if no text found."""
    pages  = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(pages)

    if not chunks:
        raise ValueError(
            f"No usable text could be extracted from '{uploaded_file.name}'."
        )

# Use more content from chunks for comprehensive summary
    full_text = " ".join(chunk["content"] for chunk in chunks[:5])
    summary = generate_response(
        _build_summary_prompt(full_text),
        max_tokens=1500,
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


# Render the full comparison result
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


# Public entry point
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

            with st.spinner("Generating comparison..."):
                cmp_prompt = _build_comparison_prompt(
                    summary_a, summary_b,
                    file_a.name, file_b.name,
                )
                comparison = generate_response(cmp_prompt, max_tokens=150)

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

    # Display the stored comparison result
    if st.session_state.cmp_result:
        _render_comparison_result(st.session_state.cmp_result)
