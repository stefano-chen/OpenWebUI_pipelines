from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

class QueryEnhancement:
    
    QUERY_ENHANCEMENT_PROMPT = "You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.\n" \
    "Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.\n" \
    "Respond only with the reformulated query.\n Original query: {original_query}\n"
    
    def __init__(self, query: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.prompt = PromptTemplate.from_template(self.QUERY_ENHANCEMENT_PROMPT)
        self.query = query
        self.enhancement_chain = self.prompt | self.llm

    def enhance(self) -> str:
        result = self.enhancement_chain.invoke(input={"original_query": self.query})
        return result.content