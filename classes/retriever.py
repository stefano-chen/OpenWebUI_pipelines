from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing_extensions import List
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle

class FAISSBM25Retriever:

    K = 5

    def __init__(self, embedding_model: HuggingFaceEmbeddings, documents: List[Document] | None = None, save_dir_path: Path | None = None):
        self.embedding_model = embedding_model
        if documents:
            self.vector_store = FAISS.from_documents(documents=documents, embedding=self.embedding_model)
            self.bm25_retriever = BM25Retriever.from_documents(documents=documents)
            self.bm25_retriever.k = self.K
            self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.K})
            self._create_ensemble()
            if save_dir_path:
                if not save_dir_path.exists():
                    save_dir_path.mkdir()
                self.vector_store.save_local(folder_path=save_dir_path)
                with open(save_dir_path.joinpath("bm25_retriever.pkl"), "wb") as f:
                    pickle.dump(self.bm25_retriever, f)

    def _create_ensemble(self):
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[0.5, 0.5]
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.ensemble_retriever.invoke(input=query)
    
    def load(self, load_dir_path: Path):
        self.vector_store = FAISS.load_local(folder_path=load_dir_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        with open(load_dir_path.joinpath("bm25_retriever.pkl"), "rb") as f:
            self.bm25_retriever = pickle.load(f)
        self._create_ensemble()