from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise ValueError("GROQ_API_KEY not found in environment or Streamlit secrets")


def create_rag_pipeline(youtube_url):
    loader=YoutubeLoader.from_youtube_url(youtube_url,add_video_info=False)
    docs=loader.load()

    split_text=RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks=split_text.split_documents(docs)

    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(chunks,embedding)

    retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
    llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0.2,api_key=api_key)
    prompt=PromptTemplate(
    template="""
Answer the question ONLY using the provided context.

If the answer is not in the context, say "I don't know".

IMPORTANT RULES:
- Always answer in the SAME language as the user's question.
- Ignore the language of the retrieved context while generating the response.
- If the question is in English, answer ONLY in English.
- If the question is in Hinglish, answer in Hinglish.
- If the question is in Hindi, answer ONLY in Hindi.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)
    return retriever,llm,prompt


def preprocess_query(query):

    hindi_words = ['kya', 'kaise', 'kyu', 'hai', 'ka', 'hota']

    if any(word in query.lower() for word in hindi_words):

        try:

            hindi_query = transliterate(
                query,
                sanscript.ITRANS,
                sanscript.DEVANAGARI
            )

            return hindi_query

        except:

            return query

    return query


    

def ask_question(query, retriever, llm, prompt):
    
    processed_query=preprocess_query(query)
    docs = retriever.invoke(processed_query)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)

    return response.content