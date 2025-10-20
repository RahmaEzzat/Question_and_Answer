# 📘 PDF Q&A using HuggingFace + Gemini + LangChain

An interactive **Streamlit app** that allows users to upload any PDF file and ask natural-language questions about its content.  
The app combines **HuggingFace embeddings**, **Google Gemini**, and **LangChain** to provide intelligent and context-aware answers. 🤖  

---

## 🚀 Features

- 📄 Upload any PDF file  
- 🧩 Automatic text chunking using **NLTK**  
- 🧠 Vector embedding generation using **HuggingFace (all-MiniLM-L6-v2)**  
- 💾 Vector database storage with **Chroma**  
- 💬 Natural question answering using **Google Gemini (via LangChain)**  
- ⚙️ Local caching of embeddings for faster reloads  

---

## 🛠️ Tech Stack

| Component | Description |
|------------|-------------|
| **Streamlit** | Front-end framework for web app |
| **LangChain** | LLM orchestration framework |
| **HuggingFace Embeddings** | Text vectorization model |
| **Google Gemini** | LLM used for answering questions |
| **ChromaDB** | Vector database for semantic search |
| **NLTK** | Text preprocessing and chunking |

---

## 📂 Project Structure

📁 project/
│
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies
└── vector_db/ # Auto-created for embeddings and database storage


---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pdf-qa-langchain.git
   cd pdf-qa-langchain
2. Create a virtual environment
   python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows

3. Install dependencies
   ```bash 
   pip install -r requirements.txt


4. Run the Streamlit app
   ```bash
   streamlit run app.py

   
##🔑 Environment Variable

You'll need a Google API key to use Gemini:

Go to Google AI Studio

Generate an API key

Enter it in the Streamlit input box labeled
    
    🔑 Enter your Google API Key

##💡 Future Improvements

Add support for multiple document formats (DOCX, TXT)

Enable memory-based multi-turn chat

Add user authentication for personal document storage

##🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to change.
