from typing import List, Union, Generator, Iterator
from pathlib import Path
from classes.rag import RAG

class Pipeline:

    DATA_DIR_PATH = Path("./data")
    DATASTORE_DIR_PATH = Path("./datastore")

    def __init__(self):
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        # load_dotenv("./.env")
        self.rag_system = RAG(
            pdf_dir_path=self.DATA_DIR_PATH, 
            google_llm="gemini-2.5-flash-lite", 
            huggingface_embedding="sentence-transformers/all-mpnet-base-v2",
            datastore_dir_path=self.DATASTORE_DIR_PATH
        )

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        answer = self.rag_system.answer(query=user_message)
        return answer
