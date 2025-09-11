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
from langchain_openai import ChatOpenAI

import base64

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

        # You first need to start the LM studio server by typing in the terminal
        #  > lms server start -p 9999
        # the -p flag is used to set a custom port number
        self.llm = ChatOpenAI(
            model="qwen/qwen3-4b-2507",
            base_url="http://localhost:9999/v1",
            api_key="not-needed" # LM Studio doesn't check it, but LangChain expects one
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        if not self.DATASTORE_DIR_PATH.exists():
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

    def _send_event(self, type: str, content: str | List[bytes], done: bool = False):
        event = {
            "event": {
                "type": "",
                "data": {}
                }
            }
        
        if type == "status":
            event["event"]["data"]["description"] = content
            event["event"]["data"]["done"] = done
        elif type == "message" or type == "replace":
            event["event"]["data"]["content"]= content
        elif type == "files":
            event["event"]["data"]["files"] = content

        event["event"]["type"] = type
        return event

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict,) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        if not body["stream"]:
            return None
        
        TESTING_DELAY = 2

        yield self._send_event(type="status", content="Retrieving Chunks")

        time.sleep(TESTING_DELAY)

        retrieved_chunks = self.retriever.retrieve(query=user_message)

        yield self._send_event(type="status", content="Aggregating Chunks")

        time.sleep(TESTING_DELAY)

        context = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)

        yield self._send_event(type="status", content="Creating Prompt")

        time.sleep(TESTING_DELAY)

        prompt = PromptTemplate.from_template(self.QA_PROMPT)

        yield self._send_event(type="status", content="Asking LLM")

        answering_chain = prompt | self.llm

        time.sleep(TESTING_DELAY)

        yield self._send_event(type="status", content="Putting all together", done=True)

        for token in answering_chain.stream(input={"context": context, "question": user_message}):
            yield self._send_event(type="message", content=token.content)


        # Send image/s in response

        # NOT WORKING
        # with open("./image/cat1.jpg", "rb") as img_file:
        #     encoded_img = base64.b64encode(img_file.read()).decode("utf-8")
        #     print(encoded_img)
        #     yield self._send_event(type="message", content="![image](data:image/jpeg;base64,")
        #     for chunk in encoded_img[::50]:
        #         yield self._send_event(type="message", content=chunk)
        #     yield self._send_event(type="message", content=")")

        # with open("./image/cat1.jpg", "rb") as img_file:
        #     b64_string = base64.b64encode(img_file.read()).decode("utf-8")

        # yield {
        #     "data": [
        #         {
        #             "b64_json": f"{b64_string[:200]}"
        #         }
        #     ]
        # }

        # WORKAROUND by sending a url/link of the image
        yield self._send_event(type="files", content=[{"type": "image", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/960px-Cat_November_2010-1a.jpg", "name": "cat"}])

