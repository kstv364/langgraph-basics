from typing import List, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
# Replace OpenAI embeddings with Ollama embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# Replace ChatOpenAI with ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# 1. Data loading and splitting (same as before)
news_urls = [
    "https://www.bbc.com/news",
    "https://www.cnn.com/world",
    "https://www.nytimes.com/section/world",
    "https://www.reuters.com/world/",
    "https://www.aljazeera.com/news/"
]
docs = [WebBaseLoader(url).load() for url in news_urls]
docs_list = [item for sub in docs for item in sub]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=20
)
doc_splits = text_splitter.split_documents(docs_list)

# 2. Create embeddings with Ollama
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="current-affairs-news",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# 3. Create ChatOllama-based summarizer
prompt = ChatPromptTemplate.from_template(
    """
You are a news analyst summarizing the latest current affairs.
Use the retrieved articles to provide a concise summary.
Highlight key global events and developments.

Question: {question}
News Articles: {context}
Summary:
"""
)
model = ChatOllama(model="llama3")
current_affairs_chain = (prompt | model | StrOutputParser())

class CurrentAffairsGraphState(TypedDict):
    question: str
    retrieved_news: List[str]
    generation: str

def retrieve_current_affairs(state: CurrentAffairsGraphState):
    q = state["question"]
    docs = retriever.invoke(q)
    return {"question": q, "retrieved_news": docs}

def generate_current_affairs_summary(state: CurrentAffairsGraphState):
    q = state["question"]
    docs = state.get("retrieved_news") or retriever.invoke(q)
    summary = current_affairs_chain.invoke({"question": q, "context": docs})
    return {"question": q, "retrieved_news": docs, "generation": summary}

def create_current_affairs_workflow():
    g = StateGraph(CurrentAffairsGraphState)
    g.add_node("retrieve", retrieve_current_affairs)
    g.add_node("generate", generate_current_affairs_summary)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()

# Run it
current_affairs_graph = create_current_affairs_workflow()
inputs = {"question": "What are the top sport headlines today?"}
response = current_affairs_graph.invoke(inputs)
print("\n--- CURRENT AFFAIRS SUMMARY ---")
print(response["generation"])
