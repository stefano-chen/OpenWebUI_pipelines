from langchain_community.document_loaders import PyPDFDirectoryLoader
from pathlib import Path
from typing_extensions import List
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, dir_path: Path):
        self.loader = PyPDFDirectoryLoader(path=dir_path)

    def load(self) -> List[Document]:
        return self.loader.load()