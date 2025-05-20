import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import os
from dotenv import load_dotenv
from fpdf import FPDF
import tempfile

# Load environment variables
load_dotenv()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App title
st.title("🎥 YouTube Q&A ")
st.write("Paste a YouTube video ID (e.g. `LPZh9BOjkQs`) and ask questions.")

# Input: YouTube video ID
video_id = st.text_input("Enter YouTube video ID:")

# Input: Question
question = st.text_input("Enter your question:")

if st.button("Run Q&A"):
    try:
        with st.spinner("Fetching and processing transcript..."):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser

            result = main_chain.invoke(question)

        # Add Q&A to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result
        })

        st.success("Answer generated!")
        st.markdown(f"**Answer:** {result}")

    except TranscriptsDisabled:
        st.error("This video has no captions available.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display entire chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🗂 Full Chat History")
    for i, entry in enumerate(st.session_state.chat_history, start=1):
        st.markdown(f"**Q{i}:** {entry['question']}")
        st.markdown(f"**A{i}:** {entry['answer']}")
        st.markdown("---")

# PDF export button
if st.session_state.chat_history:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for i, entry in enumerate(st.session_state.chat_history, start=1):
        pdf.multi_cell(0, 10, f"Q{i}: {entry['question']}\n\nA{i}: {entry['answer']}\n\n")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, "rb") as f:
        st.download_button(
            label="📄 Download Full Chat as PDF",
            data=f,
            file_name="YouTube_QnA_Chat_History.pdf",
            mime="application/pdf"
        )
