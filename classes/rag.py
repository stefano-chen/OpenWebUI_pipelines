from classes.loader import PDFLoader
from classes.text_splitter import DocumentSplitter
from classes.retriever import FAISSBM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain.prompts import PromptTemplate
from classes.query import QueryEnhancement
from classes.summarize import ChunksSummarizer

class RAG:

    QA_PROMPT = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n" \
    "If you don't know the answer, just say that you don't know. Keep the answer concise.\n" \
    "Context: {context}\n" \
    "Question: {question}"

    def __init__(self, pdf_dir_path: Path, google_llm: str, huggingface_embedding: str, datastore_dir_path: str | Path | None):
        self.llm = ChatGoogleGenerativeAI(model=google_llm)
        self.embedding_model = HuggingFaceEmbeddings(model_name=huggingface_embedding)
        if not datastore_dir_path.exists():
            self.loader = PDFLoader(dir_path=pdf_dir_path)
            loaded_docs = self.loader.load()
            splitter = DocumentSplitter(loaded_docs)
            chunks = splitter.split()
            self.retriever = FAISSBM25Retriever(embedding_model=self.embedding_model, documents=chunks, save_dir_path=datastore_dir_path)
        else:
            self.retriever = FAISSBM25Retriever(embedding_model=self.embedding_model)
            self.retriever.load(datastore_dir_path)

    def answer(self, query: str) -> str:
        query_enhancer = QueryEnhancement(query=query)
        enhanced_query = query_enhancer.enhance()
        retrieved_chunks = self.retriever.retrieve(query=enhanced_query)
        chunks_summarizer = ChunksSummarizer(retrieved_chunks)
        summarized_chunks = chunks_summarizer.summarize()
        context = "\n\n".join(chunk.page_content for chunk in summarized_chunks)
        prompt = PromptTemplate.from_template(self.QA_PROMPT)
        answering_chain = prompt | self.llm
        answer = answering_chain.invoke(input={"context": context, "question": query})
        return answer.content
        