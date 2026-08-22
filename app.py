import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import tempfile
import re


@st.cache_resource
def load_embeddings():
    return  HuggingFaceEmbeddings(
                    model = "sentence-transformers/all-MiniLM-L6-v2"
                )

embeddings_model = load_embeddings()

@st.cache_resource
def load_llm():
    return ChatGroq(
    model= "openai/gpt-oss-120b",
    temperature= 0
)

llm = load_llm() 
            


def format_docs(docs):
    return "\n\n".join(
        f"[page {doc.metadata['page_label']}]\n{doc.page_content}" for doc in docs
    )


if "chain" not in st.session_state:
    st.session_state.chain = None


if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.title("HadiGPT2.6")

with st.sidebar:
    st.title("Settings")

    pdf = st.file_uploader("📄 Upload your PDF")

    if st.session_state.pdf_name:
        st.success(f"loaded:\n{st.session_state.pdf_name}")


    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()




prompt = ChatPromptTemplate.from_template("""
Answer the user's question using only the context below.

Context: {context}
Qestion: {question}

Instructions:
- Answer clearly and naturally.
- Use only the information provided in the context.
- Keep the answer concise: 3 to 5 sentences.
- If the answer is not in the context, say you don't know.
""")

api_key = st.secrets["GROQ_API_KEY"]




if pdf and st.session_state.pdf_name != pdf.name:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf.getvalue())
        temp_path = temp_file.name


    loader = PyPDFLoader(temp_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents= chunks,
        embedding= embeddings_model
    )

    retriever = vectorstore.as_retriever(
        search_kwargs = {"k":3}
    )

    chain = RunnableParallel(
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
    )

    rag_chain = chain | prompt | llm

    st.session_state.chain = chain

    st.session_state.pdf_name = pdf.name

    st.session_state.rag_chain = rag_chain


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message['role'] == 'assistant':
            for page in message.get("sources", []):
                st.write(f'📄 Page {page}')


question = st.chat_input("ask anything about your book..")

if question:

    if st.session_state.rag_chain is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    with st.chat_message('user'):
        st.markdown(question)

    st.session_state.messages.append(
        {'role': 'user',
         'content': question}
    )

    # with st.spinner("🤖 HadiGPT is thinking..."):

        # response = st.session_state.rag_chain.invoke(question)

    result = st.session_state.chain.invoke(question)

    docss = result['context']

    sources = re.findall(r"\[page (.*?)\]", docss)
    sources = list(dict.fromkeys(sources))

    # answer = response.content

    # with st.chat_message('assistant'):
    #     st.markdown(answer)
    #     for page in sources:
    #         st.write(f"📄 Page {page}")

    #     st.session_state.messages.append(
    #         {'role': 'assistant',
    #         'content': answer,
    #         'sources': sources}
    #     )

    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ''

        for chunk in st.session_state.rag_chain.stream(question):
            full_response += chunk.content
            message_placeholder.markdown(full_response)

        for page in sources:
             st.write(f"📄 Page {page}")

        st.session_state.messages.append(
            {'role': 'assistant',
             'content': full_response,
             'sources': sources

            }
        )