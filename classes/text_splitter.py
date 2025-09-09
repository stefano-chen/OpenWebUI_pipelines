from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing_extensions import List

class DocumentSplitter:
    def __init__(self, documents:List[Document]):
        self.docs = documents
    
    def split(self, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(self.docs)
        