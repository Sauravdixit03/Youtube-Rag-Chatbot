import streamlit as st
from RAG_pipeline import create_rag_pipeline, ask_question

st.set_page_config(page_title="YouTube RAG Chatbot")

st.title("🎥 YouTube Video Chatbot")

# Input URL
youtube_url = st.text_input("Enter YouTube Video URL")

# Session state (VERY IMPORTANT)
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.llm = None
    st.session_state.prompt = None

@st.cache_resource
def load_pipeline(url):
    return create_rag_pipeline(url)

# Process button
if st.button("Process Video"):
    if youtube_url:
        with st.spinner("Processing video..."):
            retriever, llm, prompt = load_pipeline(youtube_url)
            
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.prompt = prompt

        st.success("✅ Video processed! You can now ask questions.")
    else:
        st.warning("⚠️ Please enter a YouTube URL")

# Question input
query = st.text_input("Ask a question about the video")

# Answer button
if st.button("Get Answer"):
    if st.session_state.retriever and query:
        with st.spinner("Thinking..."):
            answer = ask_question(
                query,
                st.session_state.retriever,
                st.session_state.llm,
                st.session_state.prompt
            )

        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please process video first and enter a question")