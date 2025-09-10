from typing import List, Union, Generator, Iterator, Optional
from pathlib import Path
from classes.rag import RAG
import time

from classes.loader import PDFLoader
from classes.text_splitter import DocumentSplitter
from classes.retriever import FAISSBM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain.prompts import PromptTemplate
from classes.query import QueryEnhancement
from classes.summarize import ChunksSummarizer

# Uncomment to disable SSL verification warnings if needed.
# warnings.filterwarnings('ignore', message='Unverified HTTPS request')


class Pipeline:

    DATA_DIR_PATH = Path("./data")
    DATASTORE_DIR_PATH = Path("./datastore")

    QA_PROMPT = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n" \
    "If you don't know the answer, just say that you don't know. Keep the answer concise.\n" \
    "Context: {context}\n" \
    "Question: {question}"

    def __init__(self):
        self.name = "CUSTOM RAG"
        self.description = (
            "This is a custom rag that demonstrates how to use the status event."
        )

    async def on_startup(self):
        # This function is called when the server is started.

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        if not self.DATA_DIR_PATH.exists():
            self.loader = PDFLoader(dir_path=self.DATA_DIR_PATH)
            loaded_docs = self.loader.load()
            splitter = DocumentSplitter(loaded_docs)
            chunks = splitter.split()
            self.retriever = FAISSBM25Retriever(embedding_model=self.embedding_model, documents=chunks, save_dir_path=self.DATASTORE_DIR_PATH)
        else:
            self.retriever = FAISSBM25Retriever(embedding_model=self.embedding_model)
            self.retriever.load(self.DATASTORE_DIR_PATH)

    async def on_shutdown(self):
        # This function is called when the server is shutdown.
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict,) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        if not body["stream"]:
            return None

        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Retrieving Chunks",
                    "done": False
                },
            }
        }

        time.sleep(5)

        retrieved_chunks = self.retriever.retrieve(query=user_message)

        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Aggregate Chunks",
                    "done": False
                }
            }
        }

        time.sleep(5)

        context = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)

        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Creating Prompt",
                    "done": False
                }
            }
        }

        time.sleep(5)

        prompt = PromptTemplate.from_template(self.QA_PROMPT)

        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Asking LLM",
                    "done": False
                }
            }
        }

        time.sleep(5)

        answering_chain = prompt | self.llm
        answer = answering_chain.invoke(input={"context": context, "question": user_message})

        yield {
            "event": {
                "type": "status",
                "data": {
                    "description": "Putting all together",
                    "done": True
                }
            }
        }

        yield answer.content





