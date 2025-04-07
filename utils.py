from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import streamlit as st
GEMINI_API_KEY = st.secrets["GOOGLE_AI_API_KEY"]



# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# HF_API_KEY = os.getenv("HF_API_KEY")

# llm_openai = OpenAI(api_key=OPEN_AI_API_KEY, model="gpt-3.5-turbo")
llm_gemini = ChatGoogleGenerativeAI( google_api_key= GEMINI_API_KEY, model="gemini-1.5-flash")
embeddings_open_ai = GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY,model="models/embedding-001")

# embeddings_cohere = CohereEmbeddings(api_key=COHERE_API_KEY,model="embed-multilingual-v3.0") # embed-english-v3.0
# embeddings_hunggingface = HuggingFaceInferenceAPIEmbeddings(api_key=HF_API_KEY, model="sentence-transformers/all-MiniLM-16-v2")


# def ask_gemini(prompt):
#     """
#     Sends a prompt to the Gemini AI model and returns the response content.

#     Args:
#         prompt (str): The prompt to send to the Gemini AI model.

#     Returns:
#         str: The response content from the Gemini AI model.
#     """
#     AI_Respose = llm_gemini.invoke(prompt)
#     return AI_Respose.content



def rag_with_url(target_url, prompt):
    """
    Retrieves relevant documents from a target URL and generates an AI response based on the prompt.

    Args:
        target_url (str): The URL of the target document.
        prompt (str): The prompt for generating the AI response.

    Returns:
        str: The generated AI response.

    Raises:
        Any exceptions that may occur during the execution of the function.

    """
    loader = WebBaseLoader(target_url)
    raw_document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )

    splited_document = text_splitter.split_documents(raw_document)

    vector_store = FAISS.from_documents(splited_document, embeddings_open_ai)

    retriever = vector_store.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)
    prompt_template = """
You are an AI assistant and you will be provided with different input from the user. 
Four types of inputs will be given from the user
1. Related to AsyncAPI
2. Related to AsyncAPI code
3. Greetings
4. Irrelevent content

For 1,2 and 3 you have to give respective answer based on the query whereas for 4 you have to say that you are an AI Assistant for Async API and you cannot answer his question.
"""
    final_prompt = f"""{prompt_template} \n\nQuestion: {prompt}\n\n Below are the source docs""" + " ".join([doc.page_content for doc in relevant_documents])

    AI_Respose = llm_gemini.invoke(final_prompt)

    return AI_Respose.content


