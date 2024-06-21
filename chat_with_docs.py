import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from utils import MEMORY, load_document
from open_ai_key import my_openai_key

LLM = ChatOpenAI(
    model_name="gpt-4o", temperature=0, streaming=True,
    openai_api_key=my_openai_key
)


def configure_retriever(
        docs
):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=80)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=my_openai_key)
    # alternatively: HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        },
    )
    return retriever


def configure_chain(retriever: BaseRetriever):
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000,
    )
    return ConversationalRetrievalChain.from_llm(
        **params
    )


def configure_retrieval_chain(uploaded_files):
    docs = []
    base_dir = os.path.abspath('.')  # Get the absolute path of the current directory
    temp_dir = os.path.join(base_dir, "rag_custom_chatbot", "temp_dir")
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir, file.name)
        try:
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())  # Assuming file.getvalue() correctly retrieves the binary content
        except IOError as e:
            print(f"Failed to write file {file.name}: {e}")
            continue  # Skip this file and move on to the next one

        try:
            doc_content = load_document(temp_filepath)  # Ensure this function is defined to handle loading docs
            docs.extend(doc_content)
        except Exception as e:
            print(f"Failed to load document {file.name}: {e}")
            continue

    retriever = configure_retriever(docs=docs)  # Ensure this function is defined and configured correctly
    chain = configure_chain(retriever=retriever)  # Ensure this function is defined and configured correctly
    return chain