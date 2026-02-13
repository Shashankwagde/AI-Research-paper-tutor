import streamlit as st
from rag_pipeline import (
    extract_text_from_pdf,
    chunk_text,
    create_vector_store,
    retrieve_relevant_chunks
)
from openrouter_llm import generate_response

st.set_page_config(
    page_title="Intelligent AI Research Tutor",
    page_icon="📘",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "paper_uploaded" not in st.session_state:
    st.session_state.paper_uploaded = False


with st.sidebar:
    st.title("📂 Paper Settings")

    uploaded_file = st.file_uploader(
        "Upload Research Paper (PDF)",
        type="pdf"
    )

    if uploaded_file:
        with st.spinner("Processing paper..."):

            pages = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(pages)

            if len(chunks) > 0:
                index = create_vector_store(chunks)

                st.session_state.vector_store = index
                st.session_state.chunks = chunks
                st.session_state.paper_uploaded = True

        st.success("✅ Paper processed successfully!")

    st.divider()


    if st.button("📄 Generate Summary for the paper"):

        if not st.session_state.paper_uploaded:
            st.warning("⚠ Upload a paper first.")
        else:
            with st.spinner("Generating detailed structured summary..."):

                # Use more chunks for richer summary
                full_text = " ".join([
                    chunk["content"]
                    for chunk in st.session_state.chunks[:20]
                ])

                summary_prompt = f"""
You are an academic research assistant.

Based strictly on the following research paper content,
generate a detailed structured summary.

Include:

1. Research Problem (2–3 paragraphs)
2. Proposed Methodology (2–3 paragraphs)
3. Key Results and Findings (2–3 paragraphs)
4. Main Contributions (bullet points with explanation)
5. Limitations and Future Work (1–2 paragraphs)

Make it medium-length (approximately 400–600 words).
Do not add information not present in the content.

Paper Content:
{full_text}
"""

                summary = generate_response(summary_prompt, max_tokens=1200)

                formatted_summary = "## 📄 Structured Medium-Length Summary\n\n" + summary

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_summary
                })

                st.success("Summary generated successfully!")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()



st.title("📘 Intelligent AI Research Tutor")
st.caption("Grounded AI assistant for understanding research papers.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("Ask something about the paper..."):

    if not st.session_state.paper_uploaded:
        st.warning("⚠ Please upload a research paper first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved_data = retrieve_relevant_chunks(
        prompt,
        st.session_state.vector_store,
        st.session_state.chunks,
        top_k=3
    )

    context = "\n\n".join([
        f"(Page {item['page_number']}) {item['content']}"
        for item in retrieved_data
    ])

    rag_prompt = f"""
Context:
{context}

Question:
{prompt}

Provide a clear academic answer strictly based on the context.
"""

    response = generate_response(rag_prompt, max_tokens=400)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)

    with st.expander("🔎 Retrieved Context from Research Paper"):
        for item in retrieved_data:
            st.markdown(
                f"**Page {item['page_number']}** "
            )
            st.markdown(item["content"])
            st.markdown("---")
