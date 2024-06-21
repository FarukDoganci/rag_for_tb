import streamlit as st
from chat_with_docs import configure_retrieval_chain
from utils import init_memory, load_document
from langchain.callbacks import StreamlitCallbackHandler
from utils import MEMORY

st.set_page_config(page_title=" RMIS: ChatGPT / LangChain for Tuberculosis", page_icon = "⚕️")
st.title("⚕️ RMIS: RAG for Tuberculosis")

uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type=[".pdf"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents here.")
    st.stop()


CONV_CHAIN = configure_retrieval_chain(uploaded_files)

avatars = {"human": "user", "ai": "assistant"}

if  len(MEMORY.chat_memory.messages) == 0:
    st.chat_message("assistant").markdown("Ask me anything!")

for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assistant = st.chat_message("assistant")
if user_query := st.chat_input(placeholder="Type your query/message here!"):
    st.chat_message("user").write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)
    with st.chat_message("assistant"):
        print('RUNNING CONV CHAIN')
        response = CONV_CHAIN.run({
            "question": user_query,
            "chat_history": MEMORY.chat_memory.messages
        }, callbacks=[stream_handler]
        )
        # Display the response from the chatbot
        if response:
            container.markdown(response)