#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import nltk
import tempfile
import asyncio
import os
import pickle
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------------------
# إعدادات أولية
# -------------------------------------------
st.set_page_config(page_title="PDF Q&A with LangChain", page_icon="🤖", layout="wide")
st.title("📘 Ask your PDF using Huggingface + Gemini + LangChain")

# تحميل موارد NLTK
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# -------------------------------------------
# رفع الملف
# -------------------------------------------
uploaded_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        file_path = tmp.name

    st.success("✅ File uploaded successfully!")

    # -------------------------------------------
    # تحميل الصفحات من PDF
    # -------------------------------------------
    loader = PyPDFLoader(file_path)
    pages = []

    async def load_pages():
        async for page in loader.alazy_load():
            pages.append(page)

    asyncio.run(load_pages())
    st.info(f"📄 Loaded {len(pages)} pages from your PDF.")

    # تقسيم النصوص
    documents = [page.page_content for page in pages]
    metadata = {"document": uploaded_file.name}
    text_splitter = NLTKTextSplitter(chunk_size=400)
    chunks = text_splitter.create_documents(documents, metadatas=[metadata] * len(documents))
    st.write(f"🧩 Total chunks created: {len(chunks)}")

    # -------------------------------------------
    # إعداد نموذج الدردشة والـ Embeddings
    # -------------------------------------------
    API_KEY = st.text_input("🔑 Enter your Google API Key:", type="password")

    if API_KEY:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        chat_model = ChatGoogleGenerativeAI(
            google_api_key=API_KEY,
            model="gemini-2.5-flash"
        )

        # -------------------------------------------
        # مسار حفظ قاعدة البيانات وملفات التضمين
        # -------------------------------------------
        base_dir = os.path.join(os.getcwd(), "vector_db")
        os.makedirs(base_dir, exist_ok=True)
        pdf_name = os.path.splitext(uploaded_file.name)[0]
        pkl_path = os.path.join(base_dir, f"{pdf_name}_embeddings.pkl")
        db_dir = os.path.join(base_dir, pdf_name)

        # -------------------------------------------
        # لو في mismatch في الأبعاد نحذف قاعدة البيانات
        # -------------------------------------------
        if os.path.exists(db_dir):
            try:
                _ = Chroma(persist_directory=db_dir, embedding_function=embeddings_model)
            except Exception as e:
                if "dimension" in str(e).lower():
                    shutil.rmtree(db_dir)
                    st.warning("⚠️ Old database deleted due to dimension mismatch.")

        # -------------------------------------------
        # تحميل embeddings من ملف pkl لو موجود
        # -------------------------------------------
        if os.path.exists(pkl_path):
            st.info("📦 Loading precomputed embeddings...")
            with open(pkl_path, "rb") as f:
                chunks = pickle.load(f)
        else:
            st.info("🧠 Generating new embeddings (first time only)...")
            # نحفظ التضمين لمرة واحدة فقط
            with open(pkl_path, "wb") as f:
                pickle.dump(chunks, f)
            st.success(f"💾 Embeddings saved as {pkl_path}")

        # -------------------------------------------
        # إنشاء أو تحميل قاعدة البيانات
        # -------------------------------------------
        vector_db = Chroma.from_documents(chunks, embeddings_model, persist_directory=db_dir)

        # -------------------------------------------
        # إدخال السؤال
        # -------------------------------------------
        question = st.text_input("❓ Ask your question:")

        if question:
            st.write("🔍 Searching for the most relevant context...")
            similar_docs = vector_db.similarity_search(question, k=5)

            qna_template = """
            Answer the next question using the provided context.
            If the answer is not contained in the context, say 'NO ANSWER IS AVAILABLE'

            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:
            """

            qna_prompt = PromptTemplate(
                template=qna_template,
                input_variables=["context", "question"],
            )

            prompt = ChatPromptTemplate.from_template(qna_template)
            chain = create_stuff_documents_chain(chat_model, prompt)



            with st.spinner("🤔 Generating answer..."):
                answer = chain.invoke({"context": similar_docs, "question": question})

            st.success("✅ Answer generated:")
            st.write(answer["output_text"])



