from typing import List, Union, Generator, Iterator

class Pipeline:
    def __init__(self):
        pass

    async def on_startup(self):
        import os
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.

        print(f"USER_MESSAGE: {user_message}")
        print(f"MODEL_ID: {model_id}")
        print(f"MESSAGES: {messages}")
        print(f"BODY: {body}")

        return "Hello"
