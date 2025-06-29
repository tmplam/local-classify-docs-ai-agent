import os
import sys
import asyncio
import re
from typing import Dict, Any, List, AsyncIterable, TypedDict, Annotated, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing_extensions import AsyncGenerator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import log

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from config.llm import gemini
from agents.filesystem_agent import FilesystemAgent
from agents.metadata_agent import MetadataAgent
from agents.text_extraction_agent import TextExtractionAgent
from agents.file_classification_agent import FileClassificationAgent

# Global MemorySaver instance shared by all agents
memory = MemorySaver()

# Setup MCP client for filesystem access
mcp_client = MultiServerMCPClient({
    "document_search": {
        "command": "cmd",
        "args": [
            "/c",
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "C:\\Users\\dhuu3\\Desktop\\local-classify-docs-ai-agent\\data",
        ],
        "transport": "stdio",
    }
})

# Define the state for our multi-agent system
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]  # Messages in the conversation
    current_agents: List[str]  # Which agents are currently active or have been used
    task_complete: bool  # Whether the task is complete
    require_user_input: bool  # Whether user input is needed
    feedback_on_work: Optional[str]  # Feedback from evaluator
    success_criteria_met: bool  # Whether success criteria have been met
    used_tools: List[str]  # Tools that have been used

class MultiAgentSystem:
    """
    Multi-Agent System that orchestrates specialized agents using a worker-evaluator pattern.
    
    This system can use multiple specialized agents in sequence (FilesystemAgent, MetadataAgent,
    TextExtractionAgent, FileClassificationAgent) based on the user's query and includes
    an evaluator component to assess if the task has been completed successfully.
    """
    def __init__(self):
        """
        Initialize the MultiAgentSystem.
        """
        self.model = gemini
        self.graph = None
        self.agents = {}
        self.session_id = None
        self.all_tools = []
        
    async def initialize(self):
        """
        Asynchronously initialize all specialized agents and create the workflow graph.
        
        Returns:
            MultiAgentSystem: The initialized multi-agent system.
        """
        try:
            # Initialize specialized agents
            print("Initializing specialized agents...")
            
           
            mcp_client = MultiServerMCPClient({
                "document_search": {
                    "command": "cmd",
                    "args": [
                        "/c",
                        "npx",
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        "C:\\Users\\dhuu3\\Desktop\\local-classify-docs-ai-agent\\data",
                    ],
                    "transport": "stdio"
                }
            })
            print("Using provided MCP client for FilesystemAgent")
            
            self.agents["filesystem"] = await FilesystemAgent.create(mcp_client=mcp_client)
            self.agents["metadata"] = MetadataAgent()
            self.agents["text_extraction"] = TextExtractionAgent()
            self.agents["file_classification"] = FileClassificationAgent()
            print("All specialized agents initialized successfully")
            
            # Collect tools from all agents
            # This would require each agent to expose its tools
            # For now, we'll use a placeholder approach
            self.all_tools = []
            
            # Build the graph
            await self.build_graph()
            
            return self
            
        except Exception as e:
            print(f"Error initializing multi-agent system: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def build_graph(self):
        """
        Build the state graph for the multi-agent system using a worker-evaluator pattern.
        """
        # Create a state graph with our AgentState
        graph_builder = StateGraph(AgentState)
        
        # Add nodes for each component
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("router", self.route_query)
        graph_builder.add_node("filesystem_agent", self.run_filesystem_agent)
        graph_builder.add_node("metadata_agent", self.run_metadata_agent)
        graph_builder.add_node("text_extraction_agent", self.run_text_extraction_agent)
        graph_builder.add_node("file_classification_agent", self.run_file_classification_agent)
        graph_builder.add_node("evaluator", self.evaluator)
        
        # Add edges
        graph_builder.add_edge(START, "worker")
        
        # Worker can route to specific agents or directly to evaluator
        graph_builder.add_conditional_edges(
            "worker", 
            self.worker_router,
            {
                "router": "router",
                "evaluator": "evaluator"
            }
        )
        
        # Router determines which agent to use
        graph_builder.add_conditional_edges(
            "router",
            self.determine_agent,
            {
                "filesystem": "filesystem_agent",
                "metadata": "metadata_agent",
                "text_extraction": "text_extraction_agent",
                "file_classification": "file_classification_agent",
                "evaluator": "evaluator"
            }
        )
        
        # Connect all agent nodes back to worker
        graph_builder.add_edge("filesystem_agent", "worker")
        graph_builder.add_edge("metadata_agent", "worker")
        graph_builder.add_edge("text_extraction_agent", "worker")
        graph_builder.add_edge("file_classification_agent", "worker")
        
        # Evaluator can either end or route back to worker
        graph_builder.add_conditional_edges(
            "evaluator",
            self.route_based_on_evaluation,
            {"complete": END, "continue": "worker"}
        )
        
        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=memory)
        log("Multi-agent graph built successfully")
    
    async def worker(self, state: AgentState) -> AgentState:
        """
        Worker node that processes the user's query and determines next steps.
        """
        # Create or update system message with task information
        system_message = f"""
        Bạn là một trợ lý AI thông minh có thể sử dụng nhiều công cụ và tác tử chuyên biệt để hoàn thành nhiệm vụ.
        Bạn có thể tiếp tục làm việc với một nhiệm vụ cho đến khi hoàn thành hoặc cần thêm thông tin từ người dùng.
        
        Bạn có quyền truy cập vào các tác tử chuyên biệt sau:
        1. Filesystem Agent: Sử dụng khi cần tìm kiếm, liệt kê hoặc quản lý tệp và thư mục.
        2. Metadata Agent: Sử dụng khi cần tạo hoặc quản lý metadata cho tài liệu.
        3. Text Extraction Agent: Sử dụng khi cần trích xuất văn bản từ các tệp PDF, Word hoặc PowerPoint.
        4. File Classification Agent: Sử dụng khi cần phân loại nội dung tài liệu.
        
        Bạn có thể sử dụng nhiều tác tử trong cùng một nhiệm vụ nếu cần thiết.
        Ví dụ: Tìm tệp với Filesystem Agent, sau đó trích xuất nội dung với Text Extraction Agent.
        
        Hãy phân tích yêu cầu của người dùng và quyết định sử dụng tác tử nào hoặc kết hợp các tác tử để hoàn thành nhiệm vụ.
        """
        
        # Add feedback if available
        if state.get("feedback_on_work"):
            system_message += f"""
        
Trước đó, bạn đã thử giải quyết nhiệm vụ nhưng chưa hoàn thành. Đây là phản hồi:
{state['feedback_on_work']}

Hãy điều chỉnh cách tiếp cận của bạn dựa trên phản hồi này.
        """
        
        # Add system message if not already present
        found_system_message = False
        messages = state["messages"]
        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                messages[i] = SystemMessage(content=system_message)
                found_system_message = True
                break
        
        if not found_system_message:
            state["messages"] = [SystemMessage(content=system_message)] + state["messages"]
        
        # Analyze the query to determine next steps
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            # If the last message is not from a human, we're in a continuation
            # Just return the state for routing
            return state
            
        # Sử dụng LLM để lập kế hoạch sử dụng agent
        query = last_message.content
        
        try:
            # Tạo prompt cho LLM
            planning_prompt = f"""
            Bạn là một hệ thống điều phối các agent AI chuyên biệt. Dựa trên yêu cầu của người dùng, hãy lập kế hoạch sử dụng các agent phù hợp.
            
            Yêu cầu của người dùng: "{query}"
            
            Các agent có sẵn:
            1. filesystem - Tìm kiếm, liệt kê và quản lý tệp và thư mục
            2. metadata - Tạo và quản lý metadata cho tài liệu
            3. text_extraction - Trích xuất văn bản từ tệp PDF, Word hoặc PowerPoint
            4. file_classification - Phân loại nội dung tài liệu
            
            Hãy lập kế hoạch sử dụng các agent. Đầu tiên, trả lời với danh sách các agent cần sử dụng theo thứ tự, chỉ liệt kê tên các agent (filesystem, metadata, text_extraction, file_classification), cách nhau bằng dấu phẩy.
            
            Sau đó, viết một đoạn văn ngắn giải thích kế hoạch của bạn bằng tiếng Việt.
            """
            
            # Sử dụng LLM để lập kế hoạch
            from config.llm import gemini
            response = await gemini.ainvoke(planning_prompt)
            plan_response = response.content.strip()
            
            # Tách phần danh sách agent và phần giải thích
            parts = plan_response.split('\n', 1)
            agent_list = parts[0].strip().lower()
            plan_message = parts[1].strip() if len(parts) > 1 else f"Tôi sẽ giúp bạn với yêu cầu: '{query}'."
            
            # Xử lý danh sách agent
            needed_agents = []
            valid_agents = ["filesystem", "metadata", "text_extraction", "file_classification"]
            
            for agent in valid_agents:
                if agent in agent_list:
                    needed_agents.append(agent)
            
            if not needed_agents:
                # Mặc định sử dụng filesystem nếu không có agent nào được chọn
                needed_agents.append("filesystem")
                plan_message += "\nTôi sẽ bắt đầu với Filesystem Agent để tìm kiếm thông tin."
            
            print(f"Kế hoạch agent: {needed_agents}")
            
        except Exception as e:
            print(f"Lỗi khi lập kế hoạch sử dụng agent: {e}")
            # Sử dụng mặc định nếu có lỗi
            needed_agents = ["filesystem"]
            plan_message = f"Tôi sẽ giúp bạn với yêu cầu: '{query}'. Tôi sẽ bắt đầu với Filesystem Agent để tìm kiếm thông tin."
        
        # Update state with the plan
        state["current_agents"] = needed_agents
        state["messages"].append(AIMessage(content=plan_message))
        
        return state
        
    def worker_router(self, state: AgentState) -> str:
        """
        Route from worker node to either router or evaluator.
        """
        # Nếu không còn agent nào trong kế hoạch, chuyển sang evaluator
        if not state["current_agents"]:
            log("Không còn agent nào trong kế hoạch, chuyển sang evaluator")
            return "evaluator"
            
        # Nếu đã sử dụng quá nhiều agent, chuyển sang evaluator để tránh vòng lặp vô hạn
        if len(state.get("used_tools", [])) >= 3:
            log("Đã sử dụng quá nhiều agent, chuyển sang evaluator")
            return "evaluator"
            
        # Kiểm tra xem có đang lặp lại agent không
        if len(state.get("used_tools", [])) >= 2:
            last_two_tools = state["used_tools"][-2:]
            if last_two_tools[0] == last_two_tools[1]:
                log("Phát hiện lặp lại agent, chuyển sang evaluator")
                return "evaluator"
                
        # Tiếp tục với router
        return "router"
    
    def determine_agent(self, state: AgentState) -> str:
        """
        Determine which agent to use next based on the current_agents list.
        """
        if not state["current_agents"]:
            log("Không còn agent nào trong danh sách, chuyển sang evaluator")
            return "evaluator"
            
        # Kiểm tra xem agent tiếp theo đã được sử dụng chưa
        next_agent = state["current_agents"][0]
        used_tools = state.get("used_tools", [])
        
        # Nếu agent tiếp theo đã được sử dụng, bỏ qua và kiểm tra agent tiếp theo
        if next_agent in used_tools:
            log(f"Agent {next_agent} đã được sử dụng, bỏ qua")
            # Xóa agent này khỏi danh sách
            state["current_agents"] = state["current_agents"][1:]
            # Gọi đệ quy để tìm agent tiếp theo
            return self.determine_agent(state)
        
        # Lấy agent tiếp theo từ danh sách
        log(f"Sử dụng agent {next_agent} tiếp theo")
        
        # Xóa nó khỏi danh sách để không sử dụng lại
        state["current_agents"] = state["current_agents"][1:]
        
        return next_agent
        
    async def evaluator(self, state: AgentState) -> AgentState:
        """
        Evaluator node that assesses if the task has been completed successfully.
        """
        # Format the conversation history
        conversation = "Lịch sử hội thoại:\n\n"
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                conversation += f"Người dùng: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation += f"Trợ lý: {message.content}\n"
            elif isinstance(message, SystemMessage):
                # Skip system messages in the conversation history
                pass
        
        # Get the original query
        original_query = ""
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                original_query = message.content
                break
        
        # Get the last response
        last_response = ""
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                last_response = message.content
                break
        
        # Create evaluator prompt
        evaluator_prompt = f"""
        Bạn là một đánh giá viên xác định xem một nhiệm vụ đã được hoàn thành thành công hay chưa.
        Đánh giá phản hồi cuối cùng của Trợ lý dựa trên yêu cầu ban đầu của người dùng.
        
        Yêu cầu ban đầu của người dùng là:
        {original_query}
        
        Lịch sử hội thoại:
        {conversation}
        
        Phản hồi cuối cùng của Trợ lý:
        {last_response}
        
        Hãy đánh giá xem nhiệm vụ đã hoàn thành chưa và liệu có cần thêm thông tin từ người dùng không.
        Nếu nhiệm vụ chưa hoàn thành, hãy giải thích tại sao và đề xuất cách tiếp cận khác.
        """
        
        # For now, we'll use a simple heuristic to determine if the task is complete
        # In a real implementation, you would use the LLM to evaluate this
        
        # Check if we've used at least one agent
        task_complete = len(state.get("used_tools", [])) > 0
        
        # Check if the last response contains certain keywords indicating completion
        completion_indicators = ["đã hoàn thành", "đã tìm thấy", "kết quả", "đã xử lý", "đã trích xuất", "đã phân loại"]
        for indicator in completion_indicators:
            if indicator in last_response.lower():
                task_complete = True
                break
        
        # Check if the response addresses the original query
        query_keywords = original_query.lower().split()
        response_keywords = last_response.lower().split()
        common_keywords = set(query_keywords).intersection(set(response_keywords))
        if len(common_keywords) > 2:  # If there are at least 3 common keywords
            task_complete = True
        
        # Set the evaluation results
        feedback = ""
        if task_complete:
            feedback = "Nhiệm vụ đã được hoàn thành thành công. Phản hồi đã giải quyết yêu cầu của người dùng."
            state["success_criteria_met"] = True
        else:
            feedback = "Nhiệm vụ chưa hoàn thành. Cần sử dụng thêm tác tử hoặc công cụ để giải quyết yêu cầu của người dùng."
            state["success_criteria_met"] = False
        
        # Add the feedback to the state
        state["feedback_on_work"] = feedback
        
        # Add evaluator message to the conversation
        state["messages"].append(AIMessage(content=f"[Đánh giá nội bộ: {feedback}]"))
        
        return state
    
    def route_based_on_evaluation(self, state: AgentState) -> str:
        """
        Determine next steps based on the evaluator's assessment.
        """
        if state["success_criteria_met"] or state["require_user_input"]:
            return "complete"
        else:
            # If not complete and we don't need user input, try again with other agents
            return "continue"
    async def route_query(self, state: AgentState) -> AgentState:
        """
        Route the query to the appropriate agent.
        """
        # Just pass through the state, the routing is done in determine_agent
        return state

    async def run_filesystem_agent(self, state: AgentState) -> AgentState:
        """
        Run the filesystem agent on the current query.
        """
        try:
            # Get the last message that's not from an agent
            query = ""
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    query = message.content
                    break

            log(f"Running FilesystemAgent with query: {query}")

            # Track that we're using this agent
            if "used_tools" not in state:
                state["used_tools"] = []
            state["used_tools"].append("filesystem")

            # Get the filesystem agent graph
            filesystem_agent = self.agents["filesystem"]

            # Run the agent with the query and wait for completion
            config = {"configurable": {"thread_id": self.session_id}}
            log("Waiting for FilesystemAgent to complete...")
            response = await filesystem_agent.graph.ainvoke({"messages": state["messages"]}, config=config)
            log("FilesystemAgent completed")

            # Get the agent's response
            agent_response = None
            for message in reversed(response["messages"]):
                if isinstance(message, AIMessage):
                    agent_response = message
                    break

            if not agent_response or not agent_response.content.strip():
                # If there's no response or it's empty, create a default one
                agent_response = AIMessage(content="Tôi đã tìm kiếm nhưng không tìm thấy kết quả phù hợp.")

            # Add the agent's response to the state
            response_content = f"[Filesystem Agent]: {agent_response.content}"
            print(f"FilesystemAgent response: {response_content[:100]}...")
            state["messages"].append(AIMessage(content=response_content))
            
            # Analyze the response to suggest next agent
            print("Analyzing response to suggest next agent...")
            next_agent = await self._suggest_next_agent(agent_response.content, state)
            if next_agent and next_agent not in state["current_agents"]:
                print(f"Adding {next_agent} to current_agents")
                state["current_agents"].append(next_agent)
            else:
                print(f"No additional agent suggested or already in use")

            return state

        except Exception as e:
            print(f"Error running filesystem agent: {e}")
            # Add an error message to the state
            error_message = f"Xin lỗi, tôi gặp lỗi khi tìm kiếm tệp: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
            return state

    async def _suggest_next_agent(self, response_content: str, state: AgentState) -> str:
        """
        Sử dụng LLM để phân tích phản hồi của agent và đề xuất agent tiếp theo.
        
        Args:
            response_content: Nội dung phản hồi của agent
            state: Trạng thái hiện tại
            
        Returns:
            Tên của agent tiếp theo, hoặc None nếu không có đề xuất
        """
        # Lấy yêu cầu ban đầu của người dùng
        original_query = ""
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                original_query = message.content
                break
        
        # Danh sách các agent đã sử dụng
        used_agents = state.get("used_tools", [])
        
        # Nếu đã sử dụng quá nhiều agent, không đề xuất thêm
        if len(used_agents) >= 3:
            print("Đã sử dụng quá nhiều agent, không đề xuất thêm")
            return None
        
        # Tạo prompt cho LLM
        prompt = f"""
        Bạn là một hệ thống điều phối các agent AI chuyên biệt. Dựa trên thông tin sau, hãy quyết định agent nào nên được sử dụng tiếp theo.
        
        Yêu cầu ban đầu của người dùng: "{original_query}"
        
        Phản hồi mới nhất từ agent: "{response_content}"
        
        Các agent đã được sử dụng: {used_agents}
        
        Các agent có sẵn:
        1. filesystem - Tìm kiếm, liệt kê và quản lý tệp và thư mục
        2. metadata - Tạo và quản lý metadata cho tài liệu
        3. text_extraction - Trích xuất văn bản từ tệp PDF, Word hoặc PowerPoint
        4. file_classification - Phân loại nội dung tài liệu
        
        QUAN TRỌNG: Chỉ đề xuất một agent tiếp theo nếu thực sự cần thiết dựa trên yêu cầu ban đầu và phản hồi hiện tại.
        Nếu agent hiện tại đã giải quyết được vấn đề hoặc không cần agent khác, hãy trả lời "none".
        
        Trả lời chỉ với tên của agent (filesystem, metadata, text_extraction, file_classification) hoặc "none" nếu không cần agent nào nữa.
        """
        
        try:
            # Sử dụng LLM để quyết định
            from config.llm import gemini
            response = await gemini.ainvoke(prompt)
            suggestion = response.content.strip().lower()
            
            # Kiểm tra xem có từ "none" trong phản hồi không
            if "none" in suggestion:
                print("LLM đề xuất không cần sử dụng thêm agent")
                return None
            
            # Xử lý phản hồi
            valid_agents = ["filesystem", "metadata", "text_extraction", "file_classification"]
            
            # Chỉ đề xuất agent chưa được sử dụng
            for agent in valid_agents:
                if agent in suggestion and agent not in used_agents:
                    print(f"LLM đề xuất sử dụng {agent} tiếp theo")
                    return agent
            
            # Không có đề xuất hợp lệ
            return None
            
        except Exception as e:
            print(f"Lỗi khi đề xuất agent tiếp theo: {e}")
            return None
    
    async def run_metadata_agent(self, state: AgentState) -> AgentState:
        """
        Run the metadata agent on the current query.
        """
        try:
            # Get the last message that's not from an agent
            query = ""
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    query = message.content
                    break

            log(f"Running MetadataAgent with query: {query}")

            # Track that we're using this agent
            if "used_tools" not in state:
                state["used_tools"] = []
            state["used_tools"].append("metadata")

            # Get the metadata agent graph
            metadata_agent = self.agents["metadata"]

            # Run the agent with the query and wait for completion
            config = {"configurable": {"thread_id": self.session_id}}
            log("Waiting for MetadataAgent to complete...")
            response = await metadata_agent.graph.ainvoke({"messages": state["messages"]}, config=config)
            log("MetadataAgent completed")

            # Get the agent's response
            agent_response = None
            for message in reversed(response["messages"]):
                if isinstance(message, AIMessage):
                    agent_response = message
                    break

            if not agent_response or not agent_response.content.strip():
                # If there's no response or it's empty, create a default one
                agent_response = AIMessage(content="Tôi đã xử lý metadata nhưng không có kết quả đáng chú ý.")

            # Add the agent's response to the state
            response_content = f"[Metadata Agent]: {agent_response.content}"
            print(f"MetadataAgent response: {response_content[:100]}...")
            state["messages"].append(AIMessage(content=response_content))
            
            # Analyze the response to suggest next agent
            print("Analyzing response to suggest next agent...")
            next_agent = await self._suggest_next_agent(agent_response.content, state)
            if next_agent and next_agent not in state["current_agents"]:
                print(f"Adding {next_agent} to current_agents")
                state["current_agents"].append(next_agent)
            else:
                print(f"No additional agent suggested or already in use")

            return state

        except Exception as e:
            print(f"Error running metadata agent: {e}")
            # Add an error message to the state
            error_message = f"Xin lỗi, tôi gặp lỗi khi xử lý metadata: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
            return state

    async def run_text_extraction_agent(self, state: AgentState) -> AgentState:
        """
        Run the text extraction agent on the current query.
        """
        try:
            # Tìm file path từ các tin nhắn trước đó dựa vào câu "Tôi đã tìm thấy file:"
            file_path = None
            for message in reversed(state["messages"]):
                if isinstance(message, AIMessage):
                    log(f"Checking message for file path: {message.content[:100]}...")
                    
                    # Tìm kiếm câu "Tôi đã tìm thấy file:" trong tin nhắn
                    if "Tôi đã tìm thấy file:" in message.content:
                        log(f"Found filesystem agent message with standard format")
                        
                        # Tìm đường dẫn file sau câu "Tôi đã tìm thấy file:"
                        import re
                        
                        # Tìm sau "Tôi đã tìm thấy file:"
                        file_pattern = r'Tôi đã tìm thấy file:\s*([A-Z]:\\[^\s\n\r]+)'
                        file_matches = re.findall(file_pattern, message.content)
                        
                        if file_matches:
                            # Lấy đường dẫn và loại bỏ các ký tự không mong muốn ở cuối
                            raw_path = file_matches[0]
                            # Loại bỏ các ký tự không mong muốn có thể có ở cuối đường dẫn
                            if raw_path.endswith("'}"):
                                file_path = raw_path[:-2]
                            else:
                                file_path = raw_path
                            log(f"Extracted file path: {file_path}")
                            break
                    
                    # Dự phòng: Nếu không tìm thấy câu chuẩn, thử tìm bất kỳ đường dẫn Windows nào
                    elif any(indicator in message.content for indicator in ["Đã tìm thấy file:", "tìm thấy file", "[Filesystem Agent]:"]):
                        log(f"Found filesystem agent message with non-standard format")
                        
                        # Tìm bất kỳ đường dẫn nào trong tin nhắn
                        import re
                        file_pattern = r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.[a-zA-Z0-9]+'
                        file_matches = re.findall(file_pattern, message.content)
                        
                        if file_matches:
                            file_path = file_matches[0]
                            log(f"Extracted file path using general pattern: {file_path}")
                            break

            # Get the last message that's not from an agent
            query = ""
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    query = message.content
                    break

            # Nếu tìm thấy file path, thêm vào query với format đúng
            if file_path:
                # Sử dụng định dạng chuẩn cho TextExtractionAgent
                enhanced_query = f"Extract text from the file at path {file_path}"
                log(f"Enhanced query with file path: {enhanced_query}")
            else:
                # Không tìm thấy file path, sử dụng query gốc
                enhanced_query = query
                log(f"WARNING: No file path found! Running TextExtractionAgent with original query: {query}")

            # Track that we're using this agent
            if "used_tools" not in state:
                state["used_tools"] = []
            state["used_tools"].append("text_extraction")

            # Get the text extraction agent
            text_extraction_agent = self.agents["text_extraction"]

            # Run the agent with the query and wait for completion
            log("Waiting for TextExtractionAgent to complete...")
            response = text_extraction_agent.invoke(enhanced_query, self.session_id)
            log("TextExtractionAgent completed")
            
            # Log raw response for debugging
            log(f"Raw TextExtractionAgent response: {response}")

            # Format the response - chấp nhận nhiều định dạng trả về khác nhau
            content = ""
            
            # Trường hợp 1: Response là dict với key 'content'
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
                log(f"Extracted content from response dict with key 'content': {content[:100]}...")
            
            # Trường hợp 2: Response là string
            elif isinstance(response, str):
                content = response
                log(f"Response is already a string: {content[:100]}...")
            
            # Trường hợp 3: Response là dict nhưng không có 'content', thử tìm các key khác
            elif isinstance(response, dict):
                log(f"Response keys: {list(response.keys())}")
                
                # Thử lấy từ key 'response_type' trước
                if 'response_type' in response and 'content' in response:
                    content = response['content']
                    log(f"Using content from standard response format: {content[:100]}...")
                
                # Thử lấy giá trị từ các key khác nếu là string dài
                else:
                    for key in response.keys():
                        if isinstance(response[key], str) and len(response[key]) > 20:
                            content = response[key]
                            log(f"Using content from key '{key}': {content[:100]}...")
                            break
            
            # Trường hợp 4: Các trường hợp khác, chuyển về string
            else:
                content = str(response)
                log(f"Converted response to string: {content[:100]}...")
                
            # Kiểm tra nếu content chứa kết quả trích xuất
            if "I'll extract the text from" in content or "Here's the extracted text" in content:
                # Tìm phần nội dung trích xuất sau các câu mở đầu
                import re
                extracted_text = re.split(r"Here's the extracted text[:\s]*|I'll extract the text[^:]*:\s*", content, 1)
                if len(extracted_text) > 1:
                    content = extracted_text[1].strip()
                    log(f"Extracted the actual content after introduction: {content[:100]}...")
            
            # Kiểm tra nếu content chứa "I'll use the extract_text_from" (câu trả lờ của agent)
            if "I'll use the extract_text_from" in content and file_path:
                log("Agent response contains tool usage description but no actual extraction result")
                # Thử trực tiếp các hàm trích xuất dựa vào định dạng file
                try:
                    if file_path.lower().endswith('.pdf'):
                        from agents.text_extraction_agent import extract_text_from_pdf
                        content = extract_text_from_pdf(file_path)
                        log("Directly extracted text from PDF")
                    elif file_path.lower().endswith('.docx'):
                        from agents.text_extraction_agent import extract_text_from_word
                        content = extract_text_from_word(file_path)
                        log("Directly extracted text from Word document")
                    elif file_path.lower().endswith(('.ppt', '.pptx')):
                        from agents.text_extraction_agent import extract_text_from_powerpoint
                        content = extract_text_from_powerpoint(file_path)
                        log("Directly extracted text from PowerPoint")
                except Exception as e:
                    log(f"Error in direct extraction: {e}", level='error')

            if not content.strip():
                if file_path:
                    content = f"Tôi đã cố gắng trích xuất nội dung từ file {file_path} nhưng không có kết quả đáng chú ý."
                else:
                    content = "Tôi đã cố gắng trích xuất nội dung nhưng không có kết quả đáng chú ý."

            # Add the agent's response to the state with clear indication of extraction results
            if file_path:
                response_content = f"[Text Extraction Agent]: Kết quả trích xuất từ file {file_path}:\n\n{content}"
            else:
                response_content = f"[Text Extraction Agent]: {content}"
                
            log(f"TextExtractionAgent response: {response_content[:100]}...")
            state["messages"].append(AIMessage(content=response_content))
            
            # Analyze the response to suggest next agent
            log("Analyzing response to suggest next agent...")
            next_agent = await self._suggest_next_agent(content, state)
            if next_agent and next_agent not in state["current_agents"]:
                log(f"Adding {next_agent} to current_agents")
                state["current_agents"].append(next_agent)
            else:
                log(f"No additional agent suggested or already in use")

            return state

        except Exception as e:
            log(f"Error running text extraction agent: {e}", level='error')
            # Add an error message to the state
            error_message = f"Xin lỗi, tôi gặp lỗi khi trích xuất nội dung: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
            return state
            
    async def run_file_classification_agent(self, state: AgentState) -> AgentState:
        """
        Run the file classification agent on the current query.
        """
        try:
            # Tìm nội dung cần phân loại từ TextExtractionAgent
            content_to_classify = None
            file_path = None
            
            # Tìm kết quả từ TextExtractionAgent
            for message in reversed(state["messages"]):
                if isinstance(message, AIMessage) and "[Text Extraction Agent]:" in message.content:
                    # Trích xuất nội dung sau phần giới thiệu
                    text_parts = message.content.split(":\n\n", 1)
                    if len(text_parts) > 1:
                        content_to_classify = text_parts[1].strip()
                        log(f"Found content to classify from TextExtractionAgent: {content_to_classify[:100]}...")
                        
                        # Tìm file path trong phần giới thiệu
                        import re
                        file_pattern = r'từ file ([A-Z]:\\[^\s\n\r]+)'
                        file_matches = re.findall(file_pattern, text_parts[0])
                        if file_matches:
                            file_path = file_matches[0]
                            log(f"Found file path: {file_path}")
                        break
            
            # Nếu không tìm thấy nội dung từ TextExtractionAgent, tìm file path từ FilesystemAgent
            if not content_to_classify:
                for message in reversed(state["messages"]):
                    if isinstance(message, AIMessage) and "Tôi đã tìm thấy file:" in message.content:
                        # Trích xuất đường dẫn file từ tin nhắn
                        import re
                        file_pattern = r'Tôi đã tìm thấy file:\s*([A-Z]:\\[^\s\n\r]+)'
                        file_matches = re.findall(file_pattern, message.content)
                        if file_matches:
                            raw_path = file_matches[0]
                            # Loại bỏ các ký tự không mong muốn có thể có ở cuối đường dẫn
                            if raw_path.endswith("'}"):
                                file_path = raw_path[:-2]
                            else:
                                file_path = raw_path
                            log(f"Found file path from FilesystemAgent: {file_path}")
                            break

            # Get the last message that's not from an agent
            query = ""
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    query = message.content
                    break

            # Chuẩn bị query cho FileClassificationAgent
            if content_to_classify:
                # Nếu có nội dung, sử dụng nội dung đó để phân loại
                classification_query = f"Phân loại tệp theo nội dung: '{content_to_classify[:1000]}'"  # Giới hạn độ dài
                log(f"Using extracted content for classification")
            elif file_path:
                # Nếu chỉ có đường dẫn, yêu cầu agent phân loại dựa trên file path
                classification_query = f"Hãy phân loại file: {file_path}"
                log(f"Using file path for classification: {file_path}")
            else:
                # Không có cả nội dung và đường dẫn, sử dụng query gốc
                classification_query = query
                log(f"No content or file path found. Using original query: {query}")
            
            # Track that we're using this agent
            if "used_tools" not in state:
                state["used_tools"] = []
            state["used_tools"].append("file_classification")

            # Get the file classification agent
            file_classification_agent = self.agents["file_classification"]

            # Run the agent with the prepared query
            log(f"Running FileClassificationAgent with query: {classification_query[:100]}...")
            response = file_classification_agent.invoke(classification_query, self.session_id)
            log("FileClassificationAgent completed")
            
            # Log raw response for debugging
            log(f"Raw FileClassificationAgent response: {response}")

            # Xử lý kết quả từ FileClassificationAgent
            classification_result = ""
            
            # Trường hợp 1: Response là dict với key 'content'
            if isinstance(response, dict) and 'content' in response:
                classification_result = response['content']
                log(f"Extracted classification from response dict: {classification_result}")
            
            # Trường hợp 2: Response là string
            elif isinstance(response, str):
                classification_result = response
                log(f"Response is already a string: {classification_result}")
            
            # Trường hợp 3: Các trường hợp khác, chuyển về string
            else:
                classification_result = str(response)
                log(f"Converted response to string: {classification_result}")

            # Kiểm tra kết quả phân loại
            if not classification_result.strip():
                classification_result = "Không thể phân loại"

            # Add the agent's response to the state with clear indication of classification results
            if file_path:
                response_content = f"[File Classification Agent]: Kết quả phân loại file {file_path}: {classification_result}"
            else:
                response_content = f"[File Classification Agent]: Kết quả phân loại: {classification_result}"
                
            log(f"FileClassificationAgent response: {response_content}")
            
            # Thêm kết quả phân loại vào state
            state["messages"].append(AIMessage(content=response_content))
            
            # Analyze the response to suggest next agent
            log("Analyzing response to suggest next agent...")
            next_agent = await self._suggest_next_agent(classification_result, state)
            if next_agent and next_agent not in state["current_agents"]:
                log(f"Adding {next_agent} to current_agents")
                state["current_agents"].append(next_agent)
            else:
                log(f"No additional agent suggested or already in use")

            return state

        except Exception as e:
            log(f"Error running file classification agent: {e}", level='error')
            # Add an error message to the state
            error_message = f"Xin lỗi, tôi gặp lỗi khi phân loại tệp: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
            return state

    async def plan_agents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lập kế hoạch sử dụng các agent dựa trên yêu cầu của người dùng.
        
        Args:
            state: Trạng thái hiện tại của hệ thống
            
        Returns:
            Trạng thái đã cập nhật với kế hoạch sử dụng agent
        """
        # Lấy yêu cầu của người dùng từ tin nhắn cuối cùng
        last_message = state["messages"][-1]
        query = last_message.content
        
        try:
            # Tạo prompt cho LLM
            planning_prompt = f"""
            Bạn là một hệ thống điều phối các agent AI chuyên biệt. Dựa trên yêu cầu của người dùng, hãy lập kế hoạch sử dụng các agent phù hợp.
            
            Yêu cầu của người dùng: "{query}"
            
            Các agent có sẵn:
            1. filesystem - Tìm kiếm, liệt kê và quản lý tệp và thư mục
            2. metadata - Tạo và quản lý metadata cho tài liệu
            3. text_extraction - Trích xuất văn bản từ tệp PDF, Word hoặc PowerPoint
            4. file_classification - Phân loại nội dung tài liệu
            
            Hãy lập kế hoạch sử dụng các agent. Đầu tiên, trả lời với danh sách các agent cần sử dụng theo thứ tự, chỉ liệt kê tên các agent (filesystem, metadata, text_extraction, file_classification), cách nhau bằng dấu phẩy.
            
            Sau đó, viết một đoạn văn ngắn giải thích kế hoạch của bạn bằng tiếng Việt.
            """
            
            # Sử dụng LLM để lập kế hoạch
            from config.llm import gemini
            response = await gemini.ainvoke(planning_prompt)
            plan_response = response.content.strip()
            
            # Tách phần danh sách agent và phần giải thích
            parts = plan_response.split('\n', 1)
            agent_list = parts[0].strip().lower()
            plan_message = parts[1].strip() if len(parts) > 1 else f"Tôi sẽ giúp bạn với yêu cầu: '{query}'."
            
            # Xử lý danh sách agent
            needed_agents = []
            valid_agents = ["filesystem", "metadata", "text_extraction", "file_classification"]
            
            for agent in valid_agents:
                if agent in agent_list:
                    needed_agents.append(agent)
            
            if not needed_agents:
                # Mặc định sử dụng filesystem nếu không có agent nào được chọn
                needed_agents.append("filesystem")
                plan_message += "\nTôi sẽ bắt đầu với Filesystem Agent để tìm kiếm thông tin."
            
            log(f"Kế hoạch agent: {needed_agents}")
            
        except Exception as e:
            log(f"Lỗi khi lập kế hoạch sử dụng agent: {e}", level='error')
            # Sử dụng mặc định nếu có lỗi
            needed_agents = ["filesystem"]
            plan_message = f"Tôi sẽ giúp bạn với yêu cầu: '{query}'. Tôi sẽ bắt đầu với Filesystem Agent để tìm kiếm thông tin."
        
        # Update state with the plan
        state["current_agents"] = needed_agents
        state["messages"].append(AIMessage(content=plan_message))
        
        return state
        
    async def run(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Run the multi-agent system with the given query.
        """
        try:
            # Set session ID
            self.session_id = session_id or str(uuid.uuid4())
            
            # Initialize state
            state = {
                "messages": [HumanMessage(content=query)],
                "current_agents": [],
                "task_complete": False,
                "require_user_input": False,
                "success_criteria_met": False,
                "completed": False,
                "used_tools": [],
                "chain_of_thought": ["1. Bắt đầu xử lý yêu cầu: " + query]
            }
            
            # Plan which agents to use
            state = await self.plan_agents(state)
            log(f"Kế hoạch agent: {state['current_agents']}")
            state["chain_of_thought"].append(f"2. Lập kế hoạch sử dụng các agent: {', '.join(state['current_agents'])}")
            
            # Run the agents in the planned order
            step_count = 3
            while not state.get("completed", False) and state["current_agents"]:
                agent_name = state["current_agents"].pop(0)
                log(f"Running agent: {agent_name}")
                state["chain_of_thought"].append(f"{step_count}. Đang chạy agent: {agent_name}")
                
                # Lưu trạng thái trước khi chạy agent
                pre_run_messages_count = len(state["messages"])
                
                # Run the agent
                if agent_name == "filesystem":
                    state = await self.run_filesystem_agent(state)
                elif agent_name == "text_extraction":
                    state = await self.run_text_extraction_agent(state)
                elif agent_name == "file_classification":
                    state = await self.run_file_classification_agent(state)
                elif agent_name == "metadata":
                    state = await self.run_metadata_agent(state)
                else:
                    log(f"Unknown agent: {agent_name}")
                
                # Lấy kết quả mới nhất từ agent
                if len(state["messages"]) > pre_run_messages_count:
                    latest_message = state["messages"][-1].content
                    # Rút gọn nội dung để hiển thị trong chain of thought
                    if len(latest_message) > 200:
                        summary = latest_message[:197] + "..."
                    else:
                        summary = latest_message
                    state["chain_of_thought"].append(f"{step_count}a. Kết quả từ {agent_name}: {summary}")
                
                step_count += 1
            
            # Mark as completed
            state["completed"] = True
            state["chain_of_thought"].append(f"{step_count}. Hoàn thành xử lý")
            
            # Return the final state
            log(f"Used tools: {state.get('used_tools', [])}")
            return {
                "response_type": "data",
                "is_task_complete": True,
                "require_user_input": False,
                "content": state["messages"][-1].content if state["messages"] else "",
                "state": state,
                "chain_of_thought": state["chain_of_thought"]
            }
        except Exception as e:
            log(f"Error running multi-agent system: {e}", level='error')
            return {
                "response_type": "error",
                "content": f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
                "is_task_complete": False,
                "require_user_input": False,
                "chain_of_thought": [f"Lỗi: {str(e)}"]
            }
             
    async def stream(self, query: str, session_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the multi-agent system's response.
        
        Args:
            query: The user's query
            session_id: Session ID for memory management
            
        Yields:
            Dict with partial response content and metadata
        """
        try:
            # Initialize the agent if not already initialized
            if not self.graph:
                await self.initialize()
                
            # Set the session ID for this run
            self.session_id = session_id
            
            # Create the initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "current_agents": [],
                "task_complete": False,
                "require_user_input": False,
                "feedback_on_work": None,
                "success_criteria_met": False,
                "used_tools": []
            }
            
            # Stream the graph execution
            config = {"configurable": {"thread_id": session_id}}
            async for chunk in self.graph.astream(initial_state, config=config):
                # Extract the latest message
                if "messages" in chunk and chunk["messages"]:
                    # Find the latest non-evaluator message
                    latest_message = None
                    for message in reversed(chunk["messages"]):
                        if isinstance(message, AIMessage) and not message.content.startswith("[Đánh giá nội bộ:"):
                            latest_message = message
                            break
                    
                    if latest_message:
                        # Yield the partial response
                        yield {
                            "response_type": "text",
                            "content": latest_message.content,
                            "is_task_complete": chunk.get("success_criteria_met", False),
                            "require_user_input": chunk.get("require_user_input", False),
                            "is_partial": True,
                            "used_tools": chunk.get("used_tools", [])
                        }
            
            # Yield the final complete response
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # Find the last non-evaluator message
            final_message = None
            for message in reversed(final_state["messages"]):
                if isinstance(message, AIMessage) and not message.content.startswith("[Đánh giá nội bộ:"):
                    final_message = message
                    break
            
            if not final_message:
                final_message = final_state["messages"][-1] if final_state["messages"] else AIMessage(content="Không có phản hồi từ hệ thống.")
            
            yield {
                "response_type": "text",
                "content": final_message.content,
                "is_task_complete": final_state.get("success_criteria_met", False),
                "require_user_input": final_state.get("require_user_input", False),
                "is_partial": False,
                "used_tools": final_state.get("used_tools", [])
            }
            
        except Exception as e:
            print(f"Error streaming multi-agent system: {e}")
            yield {
                "response_type": "error",
                "content": f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
                "is_task_complete": False,
                "require_user_input": False,
                "is_partial": False
            }

# Business domain-specific agents (placeholders for future implementation)
class BusinessAnalysisAgent:
    """Agent specialized in business analysis tasks."""
    pass

class FinanceAgent:
    """Agent specialized in finance and accounting tasks."""
    pass

class HRAgent:
    """Agent specialized in HR and personnel management tasks."""
    pass


async def main():
    """
    Test the enhanced worker-evaluator multi-agent system.
    """
    try:
        # Initialize the multi-agent system
        multi_agent = await MultiAgentSystem().initialize()
        session_id = "test_session_123"
        
        # Test với câu truy vấn đơn giản
        query1 = "Tóm tắt nội dung file liên quan đến Project-Final"
        print(f"\nTest Query 1: {query1}")
        print("Running with worker-evaluator pattern...")
        result1 = await multi_agent.run(query1, session_id=f"{session_id}_1")
        print(f"Response: {result1}")
        print(f"Used tools: {result1.get('used_tools', [])}")
        
        
        
        print("\nMulti-agent tests completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
