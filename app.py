import streamlit as st
import utils

# Set the website you’re building the chatbot for (hidden from user)
TARGET_URL = ["https://www.asyncapi.com/en","https://www.asyncapi.com/tools/modelina"]

# Page setup
st.set_page_config(page_title="AsyncAPI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AsyncAPI Chatbot")
st.caption("Ask anything about AsyncAPI. I’ll answer using the latest information from the site.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input prompt
prompt = st.chat_input("Ask me about AsyncAPI...")

# On user submission
if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_response = utils.rag_with_url(TARGET_URL, prompt)
            # llm_response = utils.ask_gemini(prompt)

        # Show both responses (if needed)
        # st.markdown("🧠 **RAG Response:**")
        st.markdown(rag_response)
        # st.markdown("---")
        # st.markdown("💡 **LLM Response (without RAG):**")
        # st.markdown(llm_response)

    # Store combined message in history
    full_response = f"{rag_response}"
    st.session_state.messages.append({"role": "assistant", "content": full_response})
