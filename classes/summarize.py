from langchain_core.documents import Document
from typing_extensions import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

class ChunksSummarizer:

    SUMMARY_PROMPT = "You are an assistant for summarization tasks. Summarize the following piece of context.\n" \
    "Context:\n{context}"
    
    def __init__(self, chunks: List[Document]):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        # self.llm = HuggingFacePipeline.from_model_id(
        #     model_id="mistralai/Mistral-7B-v0.3",
        #     task="text-generation"
        # )
        self.chunks = chunks

    def summarize(self) -> List[Document]:
        summarized_chunks = []
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT)
        summarization_chain = prompt | self.llm
        for chunk in self.chunks:
            summarized_chunk = summarization_chain.invoke(input={"context": chunk.page_content})
            summarized_chunks.append(Document(page_content=summarized_chunk.content))
        return summarized_chunks

        