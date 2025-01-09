from langchain_community.vectorstores import Qdrant
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_together import ChatTogether
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from app.config import Config
from langchain.prompts import ChatPromptTemplate


embeddings = FastEmbedEmbeddings()
url = Config.QDRANT_URL
api_key = Config.QDRANT_API_KEY

client = QdrantClient(url=url, api_key = api_key)

# Initialize Qdrant as a vector store
vector_store = Qdrant(
    client=client,
    collection_name="test_rag",
    embeddings=embeddings,
)

# Initialize Together AI LLM
llm = ChatTogether(
    together_api_key=Config.TOGETHER_API,
    model="meta-llama/Llama-3-70b-chat-hf",
)
# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Define RAG node
def rag_node(message):
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Question: {input}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": message})
  
    return result


print(rag_node("What is the treatment for second degree burns?")['answer'])