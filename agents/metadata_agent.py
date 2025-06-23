import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from datetime import datetime
from utils import _format_file_timestamp
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from config.llm import gemini
from agents.base import BaseAgent
from schemas.agent_schema import ResponseFormat
from langgraph.checkpoint.memory import MemorySaver
from config.prompt import metadata_prompt
from utils.get_agent_response import get_agent_response
from typing import Any, AsyncIterable, Dict
from langchain_core.messages import AIMessage

# MemorySaver
memory = MemorySaver()  

@tool
def create_metadata(text: str, file_name: str, label: str):
    """
    Create a metadata dictionary for a given text document.

    Args:
        text (str): The content of the document.
        file_name (str): The name of the file.
        label (str): A label categorizing the document.

    Returns:
        dict: A dictionary containing metadata such as total characters, creation date, file name, and label.
    """
    metadata = {
        "total_characters": len(text),
        "creation_date": _format_file_timestamp(
            timestamp=datetime.now().timestamp(), include_time=True
        ),
        "file_name": str(file_name),
        "label": label,
    }
    return metadata

@tool
def save_metadata_to_xlsx(metadata: dict, xlsx_file_name: str, folder_dir: str = "Metadata"):
    """
    Save metadata to an Excel (.xlsx) file.

    Args:
        metadata (dict): The metadata dictionary to be saved.
        xlsx_file_name (str): The name of the Excel file to save the metadata.
        folder_dir (str, optional): The directory where the Excel file will be saved. Defaults to "Metadata".

    Returns:
        str: The path to the saved Excel file.
    """
    df = pd.DataFrame([metadata])
    path = os.path.join(folder_dir, xlsx_file_name)
    df.to_excel(path, index=False, engine='openpyxl')
    return path



class MetadataAgent(BaseAgent):
    """Metadata Agent backed by LangGraph."""
    
    @property
    def name(self):
        return self.agent_name

    def __init__(self):
        super().__init__(
            agent_name='MetadataAgent',
            description='Create metadata for a given text document and save it to an Excel file',
            content_types=['text', 'text/plain']
        )

        self.model = gemini

        self.graph = create_react_agent(
            self.model,
            checkpointer=memory,
            prompt=metadata_prompt,
            response_format=ResponseFormat,
            tools=[create_metadata, save_metadata_to_xlsx],
            name="Metadata Agent",
        )

    def invoke(self, query, sessionId) -> str:
        config = {'configurable': {'thread_id': sessionId}, 'recursion_limit': 50}
        response = self.graph.invoke({'messages': [('user', query)]}, config)
        return get_agent_response(self.graph, response['messages'][-1], config)
    
    
    async def stream(
            self, query, sessionId, task_id
        ) -> AsyncIterable[Dict[str, Any]]:
            inputs = {'messages': [('user', query)]}
            config = {'configurable': {'thread_id': sessionId}, 'recursion_limit': 50}


            for item in self.graph.stream(inputs, config, stream_mode='values'):
                message = item['messages'][-1]
                if isinstance(message, AIMessage):
                    yield {
                        'response_type': 'text',
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': message.content,
                    }
            yield get_agent_response(self.graph, message, config)
                    
    async def ainvoke(self, input_data, config=None):
        """Required method for langgraph-supervisor compatibility"""
        if config is None:
            config = {}
            
        # Extract the message content from the input data
        messages = input_data.get("messages", [])
        if messages and len(messages) > 0:
            last_message = messages[-1]
            
            # Handle different message types
            if hasattr(last_message, 'content'):
                query = last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                query = last_message['content']
            elif isinstance(last_message, tuple) and len(last_message) == 2:
                # Handle tuple format (role, content)
                query = last_message[1]
            else:
                query = str(last_message)
                
            session_id = config.get("configurable", {}).get("thread_id", "default")
            
            # Use the existing invoke method
            result = self.invoke(query, session_id)
            
            # Make sure result is a string
            if isinstance(result, dict):
                result = str(result)
                
            # Return properly formatted AIMessage
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=result)]}
    



if __name__ == "__main__":
    metadata_agent = MetadataAgent()
    result = metadata_agent.invoke(query = "Hãy tạo metadata cho tài liệu sau:\n"
                "- File Name: Chain_of_thought.pdf\n"
                "- Label: Chain of Thought\n"
                "- Text: This is a document about the chain of thought. The document is a PDF file.\n\n"
                "Sau đó, lưu metadata vào file có tên: Chain_of_thought.xlsx", sessionId = "123")
    print(result)
