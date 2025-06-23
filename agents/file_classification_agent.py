import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.llm import gemini
from config.prompt import file_classification_template
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from agents.base import BaseAgent
from schemas.agent_schema import ResponseFormat
from langgraph.checkpoint.memory import MemorySaver
from config.prompt import file_classification_prompt
from utils.get_agent_response import get_agent_response
from typing import Any, AsyncIterable, Dict
from langchain_core.messages import AIMessage

# MemorySaver
memory = MemorySaver()

@tool
def classify_file_tool(file_content: str) -> str:
    """Classify the type or category of a file based on its name and content."""
    prompt = PromptTemplate(
        template=file_classification_template,
        input_variables=["file_content"]
    )
    chain = prompt | gemini | StrOutputParser()
    result = chain.invoke({"file_content": file_content})
    return result


class FileClassificationAgent(BaseAgent):
    """File Classification Agent backed by LangGraph."""
    
    @property
    def name(self):
        return self.agent_name

    def __init__(self):
        super().__init__(
            agent_name='FileClassificationAgent',
            description='Classify the type or category of a file based on its name and content',
            content_types=['text', 'text/plain']
        )

        self.model = gemini

        self.graph = create_react_agent(
            self.model,
            checkpointer=memory,
            prompt=file_classification_prompt,
            response_format=ResponseFormat,
            tools=[classify_file_tool],
            name="File Classification Agent",
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
    agent = FileClassificationAgent()
    result = agent.invoke(query = "Phân loại tệp theo nội dung: 'Đây là một bài giảng về lịch sử Việt Nam thời kỳ phong kiến.'", sessionId = "123")
    print(result)


