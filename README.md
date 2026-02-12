# Intelligent AI Research Tutor

An AI-powered assistant designed to help users understand and interact with research papers through Retrieval-Augmented Generation (RAG). Upload a PDF research paper, generate structured summaries, and chat with the content for grounded, academic responses.

## Features

- **PDF Upload and Processing**: Extract and clean text from research papers, removing references, URLs, and DOIs.
- **Intelligent Chunking**: Split the paper into manageable chunks while preserving page metadata.
- **Vector Search**: Use FAISS and Sentence Transformers for efficient similarity search.
- **RAG-Powered Chat**: Ask questions about the paper and get responses based strictly on the content.
- **Structured Summaries**: Generate detailed summaries covering research problem, methodology, results, contributions, and limitations.
- **Streamlit Interface**: User-friendly web app for easy interaction.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shashank_wagde/research-paper-tutor.git
   cd research-paper-tutor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your OpenRouter API key:
     ```
     OPENROUTER_API_KEY=your_openrouter_api_key_here
     ```
   - Get your API key from [OpenRouter](https://openrouter.ai/).

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload a Research Paper**:
   - In the sidebar, upload a PDF file.
   - The app will process the paper and create a vector store.

3. **Generate a Summary**:
   - Click "Generate Summary for the paper" to get a structured summary.

4. **Chat with the Paper**:
   - Enter questions in the chat input.
   - The app retrieves relevant sections and generates responses using the LLM.

## Requirements

- Python 3.8 or higher
- OpenRouter API key (for LLM responses)
- Internet connection (for API calls)

## Dependencies

- streamlit
- requests
- faiss-cpu
- python-dotenv
- pypdf
- numpy
- sentence-transformers

## How It Works

1. **Text Extraction**: Uses PyPDF to extract text from each page.
2. **Cleaning**: Removes unwanted sections like references and URLs.
3. **Chunking**: Splits text into chunks of ~350 words.
4. **Embedding**: Encodes chunks using Sentence Transformers (all-MiniLM-L6-v2).
5. **Vector Store**: Stores embeddings in FAISS for fast retrieval.
6. **Retrieval**: Finds top-k relevant chunks for user queries.
7. **Generation**: Uses OpenRouter's GPT-4o-mini to generate responses based on retrieved context.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web app framework.
- [OpenRouter](https://openrouter.ai/) for LLM API.
- [FAISS](https://github.com/facebookresearch/faiss) for vector search.
- [Sentence Transformers](https://www.sbert.net/) for embeddings.
