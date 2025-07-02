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
from agents.rag_agent import RAGAgent

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
            self.agents["rag"] = RAGAgent()
            
            # Build RAG index for data directory
            data_dir = "C:\\Users\\dhuu3\\Desktop\\local-classify-docs-ai-agent\\data"
            await self.agents["rag"].build_index(data_dir)
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
        graph_builder.add_node("rag_agent", self.run_rag_agent)
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
                "rag": "rag_agent",
                "evaluator": "evaluator"
            }
        )
        
        # Connect all agent nodes back to worker
        graph_builder.add_edge("filesystem_agent", "worker")
        graph_builder.add_edge("metadata_agent", "worker")
        graph_builder.add_edge("text_extraction_agent", "worker")
        graph_builder.add_edge("file_classification_agent", "worker")
        graph_builder.add_edge("rag_agent", "worker")
        
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
        1. Filesystem Agent: Sử dụng khi cần tìm kiếm, liệt kê hoặc quản lý tệp và thư mục theo tên file.
        2. RAG Agent: Sử dụng khi cần tìm kiếm tài liệu theo nội dung hoặc ngữ nghĩa.
        3. Metadata Agent: Sử dụng khi cần tạo hoặc quản lý metadata cho tài liệu.
        4. Text Extraction Agent: Sử dụng khi cần trích xuất văn bản từ các tệp PDF, Word hoặc PowerPoint.
        5. File Classification Agent: Sử dụng khi cần phân loại nội dung tài liệu.
        
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
        
        # Kiểm tra nếu là RAG agent
        if next_agent == "rag":
            return "run_rag_agent"
        
        return next_agent
        
    async def _validate_agent_sequence(self, state: AgentState) -> AgentState:
        """
        Validate and fix the agent sequence to ensure proper workflow, especially for metadata operations.
        
        Args:
            state: Current agent state with planned agents
            
        Returns:
            Updated state with corrected agent sequence
        """
        current_agents = state.get("current_agents", [])
        log(f"Validating agent sequence: {current_agents}")
        
        # If no agents or only one agent, no need to validate sequence
        if len(current_agents) <= 1:
            return state
            
        # Check if metadata agent is included in the plan
        if "metadata" in current_agents:
            log("Phát hiện metadata agent trong kế hoạch, kiểm tra thứ tự")
            
            # Define the correct sequence for metadata operations
            metadata_workflow = []
            
            # Step 1: First agent should be filesystem or rag (search agent)
            if "filesystem" in current_agents or "rag" in current_agents:
                search_agent = "rag" if "rag" in current_agents else "filesystem"
                metadata_workflow.append(search_agent)
                log(f"Sử dụng {search_agent} làm agent tìm kiếm trong workflow metadata")
            else:
                # If no search agent is specified, add rag as default
                metadata_workflow.append("rag")
                log("Thêm rag agent vào đầu workflow vì không tìm thấy agent tìm kiếm")
                # Also add it to current_agents to ensure it's included in the final sequence
                current_agents.append("rag")
                
            # Step 2: Add text_extraction agent
            metadata_workflow.append("text_extraction")
            
            # Step 3: Add file_classification agent
            metadata_workflow.append("file_classification")
            
            # Step 4: Finally add metadata agent
            metadata_workflow.append("metadata")
            
            # Create a new agent sequence with correct order
            # Remove any agents that are already in the metadata workflow
            new_agents = [agent for agent in current_agents if agent not in metadata_workflow]
            
            # Add the metadata workflow agents in correct order
            new_agents.extend([agent for agent in metadata_workflow if agent in current_agents])
            
            # If metadata is in the plan but required agents are missing, add them
            if "metadata" in new_agents:
                metadata_index = new_agents.index("metadata")
                missing_agents = []
                
                # Check for missing required agents
                if "text_extraction" not in new_agents:
                    missing_agents.append("text_extraction")
                    log("Thêm text_extraction agent vào workflow vì cần thiết cho metadata")
                    
                if "file_classification" not in new_agents:
                    missing_agents.append("file_classification")
                    log("Thêm file_classification agent vào workflow vì cần thiết cho metadata")
                
                # Insert missing agents in the correct order before metadata
                for agent in reversed(missing_agents):
                    new_agents.insert(metadata_index, agent)
            
            # Update the state with the corrected sequence
            if new_agents != current_agents:
                log(f"Đã điều chỉnh thứ tự agent: {current_agents} -> {new_agents}")
                state["chain_of_thought"].append(f"Điều chỉnh thứ tự agent để đảm bảo workflow chính xác: {', '.join(new_agents)}")
                state["current_agents"] = new_agents
        
        return state
        
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
    
    async def _determine_search_intent(self, query: str) -> str:
        """
        Xác định mục đích tìm kiếm dựa trên câu truy vấn.
        
        Args:
            query: Câu truy vấn của người dùng
            
        Returns:
            "filesystem" nếu là tìm kiếm theo tên file, "rag" nếu tìm kiếm theo nội dung
        """
        query = query.lower()
        
        # Keywords indicating file name search
        file_name_keywords = [
            "tên file", "tìm file", "đường dẫn", "thư mục",
            "trong folder", "trong thư mục", "tìm kiếm file", "file tên"
        ]
        
        # Keywords indicating content search
        content_keywords = [
            "nội dung", "có chứa", "liên quan đến", "nói về",
            "thông tin về", "tìm kiếm thông tin", "tài liệu về", "văn bản"
        ]
        
        # Check for explicit content search
        if any(keyword in query for keyword in content_keywords):
            log(f"Phát hiện tìm kiếm theo nội dung: {query}")
            return "rag"
            
        # Check for explicit file search
        if any(keyword in query for keyword in file_name_keywords):
            log(f"Phát hiện tìm kiếm theo tên file: {query}")
            return "filesystem"
            
        # Default to content search for natural language queries
        if len(query.split()) > 3:  # Longer queries are likely content searches
            log(f"Mặc định tìm kiếm theo nội dung cho câu dài: {query}")
            return "rag"
            
        # Default to file system search for short queries
        log(f"Mặc định tìm kiếm theo tên file cho câu ngắn: {query}")
        return "filesystem"

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
            
            # Check if filesystem agent found any results
            if "Không tìm thấy" in agent_response.content or "không biết" in agent_response.content.lower() or "không tìm thấy" in agent_response.content.lower():
                print("Filesystem agent didn't find results. Trying RAG agent...")
                
                # Call RAG agent for content-based search
                rag_agent = self.agents["rag"]
                rag_result = await rag_agent.invoke(query, self.session_id)
                
                if isinstance(rag_result, dict) and 'content' in rag_result:
                    # Add RAG response to messages
                    rag_content = f"[RAG Agent]: Tìm kiếm theo nội dung file:\n\n{rag_result['content']}"
                    state["messages"].append(AIMessage(content=rag_content))
                    
                    # Add RAG to used tools
                    if "rag" not in state["used_tools"]:
                        state["used_tools"].append("rag")
                        
                    print(f"RAG agent found results: {rag_content[:100]}...")
            
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
    
    async def run_rag_agent(self, state: AgentState) -> AgentState:
        """
        Run the RAG agent to search file contents.
        """
        try:
            # Get the last message that's not from an agent
            query = ""
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    query = message.content
                    break

            log(f"Running RAGAgent with query: {query}")

            # Track that we're using this agent
            if "used_tools" not in state:
                state["used_tools"] = []
            state["used_tools"].append("rag")

            # Get the RAG agent
            rag_agent = self.agents["rag"]

            # Run the agent with the query
            log("Searching with RAGAgent...")
            response = await rag_agent.invoke(query, session_id=self.session_id)
            log("RAGAgent search completed")

            # Format the response
            if isinstance(response, dict) and 'content' in response:
                response_content = response['content']
                # Extract file_paths if present for text extraction compatibility
                file_paths = response.get('file_paths', [])
            else:
                response_content = str(response)
                file_paths = []

            # Add the agent's response to the state with file_paths as additional kwargs if available
            print(f"RAGAgent response: {response_content[:200]}...")
            if file_paths:
                log(f"RAG agent found {len(file_paths)} file paths: {file_paths}")
                state["messages"].append(AIMessage(
                    content=response_content,
                    additional_kwargs={'file_paths': file_paths}
                ))
            else:
                state["messages"].append(AIMessage(content=response_content))
                
            # Mark task as complete since RAG search is typically a terminal operation
            state["task_complete"] = True
            state["require_user_input"] = False
            
            return state
            
        except Exception as e:
            log(f"Error in RAG agent: {str(e)}", level='error')
            state["messages"].append(AIMessage(
                content=f"Có lỗi xảy ra khi tìm kiếm nội dung: {str(e)}"
            ))
            state["task_complete"] = True
            state["require_user_input"] = False
            return state

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

            # Extract file path from previous agent messages
            file_path = None
            file_content = None
            file_classification = None
            
            # First, look for file paths from RAG or filesystem agents
            for message in reversed(state["messages"]):
                if isinstance(message, AIMessage):
                    # Check for file_paths in additional_kwargs (from RAG agent)
                    if hasattr(message, '_additional_kwargs') and 'file_paths' in message._additional_kwargs:
                        file_paths = message._additional_kwargs['file_paths']
                        if file_paths and isinstance(file_paths, list) and len(file_paths) > 0:
                            file_path = file_paths[0]  # Take the first file
                            log(f"Found file path from RAG agent: {file_path}")
                            break
                    
                    # Check for file paths in message content
                    if "Tôi đã tìm thấy file:" in message.content:
                        import re
                        file_pattern = r'Tôi đã tìm thấy file:\s*([A-Z]:\\[^\s\n\r]+)'
                        file_matches = re.findall(file_pattern, message.content)
                        if file_matches:
                            file_path = file_matches[0]
                            log(f"Found file path from message content: {file_path}")
                            break
            
            # Then, look for extracted content from text extraction agent
            for message in reversed(state["messages"]):
                if not isinstance(message, AIMessage):
                    continue
                    
                # Check if this is a text extraction agent message
                is_extraction_msg = ("[Text Extraction Agent]:" in message.content or 
                                   "Nội dung trích xuất:" in message.content or
                                   "Kết quả trích xuất từ file" in message.content)
                
                if is_extraction_msg:
                    log(f"Found text extraction agent message")
                    
                    # Try to extract content using different patterns
                    content = None
                    
                    # Pattern 1: Look for content after "Nội dung trích xuất:"
                    if "Nội dung trích xuất:" in message.content:
                        parts = message.content.split("Nội dung trích xuất:", 1)
                        if len(parts) > 1:
                            content = parts[1].strip()
                    
                    # Pattern 2: Look for content after the file path
                    elif "Kết quả trích xuất từ file" in message.content:
                        # Find the first empty line after the header
                        lines = message.content.split('\n')
                        content_lines = []
                        found_header = False
                        
                        for line in lines:
                            if not found_header and ("Kết quả trích xuất từ file" in line or "Nội dung trích xuất:" in line):
                                found_header = True
                                continue
                            if found_header and line.strip():
                                content_lines.append(line.strip())
                        
                        if content_lines:
                            content = '\n'.join(content_lines).strip()
                    
                    # If we found content, clean it up and store it
                    if content:
                        # Clean up any metadata or additional text
                        if "\n\n" in content:
                            content = content.split("\n\n")[0].strip()
                        
                        # Remove any trailing metadata or instructions
                        stop_phrases = [
                            "\nLưu ý:", "\nGhi chú:", "\nThông tin thêm:",
                            "\nNote:", "\nMetadata:", "\n---"
                        ]
                        
                        for phrase in stop_phrases:
                            if phrase in content:
                                content = content.split(phrase, 1)[0].strip()
                        
                        file_content = content
                        log(f"Extracted content length: {len(file_content)} characters")
                        log(f"Content preview: {file_content[:200]}...")
                        break
            
            # Finally, look for classification from file classification agent
            for message in reversed(state["messages"]):
                if isinstance(message, AIMessage) and ("[File Classification Agent]:" in message.content or "Kết quả phân loại file" in message.content or "Giáo dục" in message.content):
                    log("Found file classification agent message")
                    # Look for classification label in different possible formats
                    import re
                    
                    # Format 1: "Phân loại: Giáo dục"
                    label_pattern1 = r'Phân loại:\s*([^\n\r]+)'
                    # Format 2: "Kết quả phân loại file ...: Giáo dục"
                    label_pattern2 = r'Kết quả phân loại file[^:]*:\s*([^\n\r]+)'
                    # Format 3: Just the label itself (e.g. "Giáo dục")
                    
                    content = message.content
                    log(f"Analyzing classification content: {content}")
                    
                    # First try the standard patterns
                    label_matches = re.findall(label_pattern1, content)
                    if not label_matches:
                        label_matches = re.findall(label_pattern2, content)
                    
                    # If still no match, check if the content itself is just the label
                    if not label_matches and len(content.strip().split()) <= 3:
                        # If content is just 1-3 words, it might be just the label
                        label_matches = [content.strip()]
                    
                    if label_matches:
                        file_classification = label_matches[0].strip()
                        log(f"Found classification label: {file_classification}")
                        break
            
            # Prepare the metadata parameters
            metadata_params = {}
            
            # Set file information
            if file_path:
                import os
                metadata_params['file_name'] = os.path.basename(file_path)
                metadata_params['file_path'] = file_path
            
            # Set classification if available
            if file_classification and file_classification.lower() not in ["không xác định", "chưa phân loại", "không có phân loại"]:
                # Clean up the classification label
                label = file_classification.split(':')[-1].strip()
                metadata_params['label'] = label
            
            # Set content if available
            if file_content:
                # Ensure content is properly formatted and not too long
                content = file_content.strip()
                if len(content) > 4000:  # Truncate if too long for the model
                    content = content[:4000] + "... [nội dung bị cắt bớt]"
                metadata_params['content'] = content
                log(f"Content length for metadata: {len(content)} characters")
            
            # Build the enhanced query for the metadata agent
            enhanced_query = "Tôi cần tạo và lưu metadata với các thông tin sau:\n\n"
            
            # Add file information if available
            if 'file_name' in metadata_params:
                enhanced_query += f"- TÊN FILE: {metadata_params['file_name']}\n"
            if 'file_path' in metadata_params:
                enhanced_query += f"- ĐƯỜNG DẪN: {metadata_params['file_path']}\n"
            if 'label' in metadata_params:
                enhanced_query += f"- PHÂN LOẠI: {metadata_params['label']} (tự động)\n"
            
            # Add content preview if available
            if 'content' in metadata_params and metadata_params['content']:
                content = metadata_params['content']
                preview_length = min(200, len(content))
                preview = content[:preview_length]
                
                enhanced_query += f"\nNỘI DUNG (XEM TRƯỚC {preview_length}/{len(content)} ký tự):\n"
                enhanced_query += f"""
================================================================================
{preview}
... [NỘI DUNG ĐẦY ĐỦ ĐƯỢC CUNG CẤP Ở PHẦN DƯỚI]
================================================================================
"""
            
            # Add clear instructions for the metadata agent
            enhanced_query += """

HƯỚNG DẪN QUAN TRỌNG:
1. ĐỌC KỸ: Toàn bộ nội dung đã được cung cấp đầy đủ ở phần dưới
2. KHÔNG yêu cầu thêm nội dung vì đã có sẵn
3. Thực hiện các bước sau:
   - Gọi create_metadata với đầy đủ thông tin
   - Lưu metadata bằng save_metadata_to_mcp
   - Trả về metadata_id đã tạo

THÔNG TIN CHI TIẾT:
"""
            
            # Include all metadata params in a clear format
            for key, value in metadata_params.items():
                if key == 'content':
                    enhanced_query += f"\nNỘI DUNG ĐẦY ĐỦ ({len(value)} ký tự):\n"
                    enhanced_query += "="*80 + "\n"
                    enhanced_query += value[:4000]  # Truncate to avoid token limits
                    if len(value) > 4000:
                        enhanced_query += "\n... [ĐÃ CẮT BỚT NỘI DUNG DO QUÁ DÀI]"
                    enhanced_query += "\n" + "="*80 + "\n"
                else:
                    enhanced_query += f"{key.upper()}: {value}\n"
            
            # Final reminder
            enhanced_query += """

LƯU Ý CUỐI CÙNG:
- Sử dụng NỘI DUNG ĐẦY ĐỦ ở trên để tạo metadata
- KHÔNG yêu cầu thêm nội dung
- Trả về metadata_id sau khi lưu thành công
"""
            
            log(f"Metadata parameters: {metadata_params}")
            log(f"Enhanced metadata query: {enhanced_query[:300]}...")

            # Get the metadata agent
            metadata_agent = self.agents["metadata"]

            # Initialize MCP connection if needed
            if not hasattr(metadata_agent, 'mcp_initialized') or not metadata_agent.mcp_initialized:
                log("Initializing MCP connection...")
                if not metadata_agent.initialize_mcp_sync():
                    error_msg = "❌ Failed to initialize MCP connection for metadata agent"
                    log(error_msg, level='error')
                    state["messages"].append(AIMessage(content=error_msg))
                    return state
            
            # Initialize variables
            metadata_id = None
            response_content = ""
            
            try:
                # Log metadata parameters for debugging
                log("Preparing to invoke MetadataAgent with the following parameters:")
                log(f"- File: {metadata_params.get('file_name', 'N/A')}")
                log(f"- Path: {metadata_params.get('file_path', 'N/A')}")
                log(f"- Label: {metadata_params.get('label', 'N/A')}")
                log(f"- Content length: {len(metadata_params.get('content', ''))} characters")
                
                # Call the metadata agent with the enhanced query and metadata
                log("Invoking MetadataAgent...")
                response = metadata_agent.invoke(
                    query=enhanced_query,
                    sessionId=self.session_id,
                    metadata={
                        'file_name': metadata_params.get('file_name'),
                        'file_path': metadata_params.get('file_path'),
                        'label': metadata_params.get('label'),
                        'content': metadata_params.get('content', '')
                    }
                )
                log("MetadataAgent completed successfully")
                
                # Process the response
                if not response or not response.strip():
                    response_content = "Tôi đã xử lý metadata nhưng không có kết quả đáng chú ý."
                    log("No response content from MetadataAgent")
                else:
                    # Clean up the response
                    response = response.strip()
                    log(f"Raw metadata agent response: {response}")
                    
                    # Try to extract metadata ID from the response
                    import re
                    
                    # Look for various possible ID formats in the response
                    id_patterns = [
                        r'metadata[_-]?id[\s:]*([a-f0-9-]+)',  # metadata-id: xxxx
                        r'id[\s:]*([a-f0-9-]{8,})',            # id: xxxxxxxx-xxxx-...
                        r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'  # UUID format
                    ]
                    
                    for pattern in id_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            metadata_id = match.group(1)
                            log(f"Found metadata ID using pattern {pattern}: {metadata_id}")
                            break
                    
                    if metadata_id:
                        response_content = f"✅ Đã lưu metadata thành công. ID: {metadata_id}"
                        log(f"Metadata saved with ID: {metadata_id}")
                        
                        # Also check if the metadata was actually saved to MCP
                        try:
                            metadata_agent = self.agents["metadata"]
                            if hasattr(metadata_agent, 'mcp_connection'):
                                result = metadata_agent.mcp_connection.call_tool_sync(
                                    "get_metadata_by_id", 
                                    {"metadata_id": metadata_id}
                                )
                                if result and 'error' not in result:
                                    log(f"Verified metadata exists in MCP: {metadata_id}")
                                else:
                                    log(f"Warning: Metadata ID {metadata_id} not found in MCP", level='warning')
                        except Exception as e:
                            log(f"Error verifying metadata in MCP: {str(e)}", level='error')
                    else:
                        # If no ID found, check if this is an error message
                        error_keywords = ['lỗi', 'error', 'failed', 'thất bại', 'không tìm thấy', 'not found']
                        if any(keyword in response.lower() for keyword in error_keywords):
                            response_content = f"❌ Lỗi khi xử lý metadata: {response}"
                        else:
                            response_content = f"ℹ️ Đã xử lý metadata: {response}"
                        
                        log(f"Metadata agent response (no ID found): {response}")
                        
                # Add the response to the conversation
                formatted_response = f"[Metadata Agent]: {response_content}"
                log(f"MetadataAgent response: {formatted_response[:200]}...")
                state["messages"].append(AIMessage(content=formatted_response))
                
                # Store metadata info in the state for future reference
                if 'metadata' not in state:
                    state['metadata'] = {}
                
                if 'metadata_ids' not in state['metadata']:
                    state['metadata']['metadata_ids'] = []
                    
                if metadata_id:  # Use the metadata_id variable that we defined earlier
                    state['metadata']['metadata_ids'].append(metadata_id)
                    log(f"Added metadata ID to state: {metadata_id}")
                else:
                    log("No metadata ID to add to state", level='warning')
                
                # Mark task as complete
                state["task_complete"] = True
                state["require_user_input"] = False
                
                return state
                
            except Exception as e:
                error_msg = f"Lỗi khi chạy MetadataAgent: {str(e)}"
                log(error_msg, level='error')
                state["messages"].append(AIMessage(
                    content=f"[Lỗi] {error_msg}. Vui lòng thử lại hoặc kiểm tra kết nối MCP server."
                ))
                state["task_complete"] = True
                state["require_user_input"] = True
                return state

        except Exception as e:
            import traceback
            print(f"Error running metadata agent: {e}")
            print(traceback.format_exc())
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
                    
                    # Kiểm tra xem tin nhắn có phải là từ RAG agent không (có thể có trường file_paths)
                    if hasattr(message, '_additional_kwargs') and 'file_paths' in message._additional_kwargs:
                        log(f"Found RAG agent message with file_paths field")
                        file_paths = message._additional_kwargs['file_paths']
                        if file_paths and isinstance(file_paths, list) and len(file_paths) > 0:
                            file_path = file_paths[0]  # Lấy file đầu tiên
                            log(f"Extracted file path from RAG agent: {file_path}")
                            break
                    
                    # Tìm kiếm câu "Tôi đã tìm thấy file:" trong tin nhắn
                    if "Tôi đã tìm thấy file:" in message.content:
                        log(f"Found agent message with standard format")
                        
                        # Tìm đường dẫn file sau câu "Tôi đã tìm thấy file:"
                        import re
                        
                        # Tìm sau "Tôi đã tìm thấy file:" - kiểm tra cả đường dẫn đầy đủ và tên file
                        full_path_pattern = r'Tôi đã tìm thấy file:\s*([A-Z]:\\[^\s\n\r]+)'
                        full_path_matches = re.findall(full_path_pattern, message.content)
                        
                        if full_path_matches:
                            # Lấy đường dẫn và loại bỏ các ký tự không mong muốn ở cuối
                            raw_path = full_path_matches[0]
                            # Loại bỏ các ký tự không mong muốn có thể có ở cuối đường dẫn
                            if raw_path.endswith("'}"):
                                file_path = raw_path[:-2]
                            else:
                                file_path = raw_path
                            log(f"Extracted full file path: {file_path}")
                            break
                    
                    # Dự phòng: Nếu không tìm thấy câu chuẩn, thử tìm bất kỳ đường dẫn Windows nào
                    elif any(indicator in message.content for indicator in ["Đã tìm thấy file:", "tìm thấy file", "[Filesystem Agent]:", "[RAG Agent]:"]):
                        log(f"Found agent message with non-standard format")
                        
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
            
    # The run_rag_agent method is already defined above (lines 653-703)
    # This duplicate definition was causing the 'Unknown agent: rag' error
            
    async def run_file_classification_agent(self, state: AgentState):
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
            # Xác định mục đích tìm kiếm (filesystem hay rag) để cung cấp gợi ý cho LLM
            search_intent = await self._determine_search_intent(query)
            intent_hint = "" if search_intent == "filesystem" else "\nGợi ý: Yêu cầu này có thể liên quan đến tìm kiếm theo nội dung, nên có thể cần sử dụng RAG agent."
            
            # Sử dụng LLM để lập kế hoạch cho mọi loại yêu cầu
            planning_prompt = f"""
            Bạn là một hệ thống điều phối các agent AI chuyên biệt. Dựa trên yêu cầu của người dùng, hãy lập kế hoạch sử dụng các agent phù hợp.
            
            Yêu cầu của người dùng: "{query}"
            
            Các agent có sẵn:
            1. filesystem - Tìm kiếm, liệt kê và quản lý tệp và thư mục theo tên file
            2. rag - Tìm kiếm tài liệu theo nội dung hoặc ngữ nghĩa (tìm kiếm theo từ khóa, chủ đề, hoặc nội dung liên quan)
            3. metadata - Tạo và quản lý metadata cho tài liệu (lưu thông tin về file như tên, loại, nhãn, mô tả vào MCP server)
            4. text_extraction - Trích xuất văn bản từ tệp PDF, Word hoặc PowerPoint
            5. file_classification - Phân loại nội dung tài liệu
            
            LƯU Ý QUAN TRỌNG (PHẢI TUÂN THỦ CHÍNH XÁC):
            - Nếu yêu cầu liên quan đến tìm kiếm theo tên file, sử dụng filesystem agent
            - Nếu yêu cầu liên quan đến tìm kiếm theo nội dung, chủ đề, hoặc ngữ nghĩa, sử dụng rag agent
            - Nếu yêu cầu liên quan đến việc lưu metadata, PHẢI tuân thủ thứ tự chính xác sau:
              1. Đầu tiên: tìm file (filesystem hoặc rag)
              2. Tiếp theo: trích xuất nội dung (text_extraction)
              3. Sau đó: phân loại (file_classification)
              4. Cuối cùng: lưu metadata (metadata)
            - KHÔNG BAO GIỜ đặt metadata agent trước text_extraction hoặc file_classification
            - Nếu yêu cầu có nhiều bước, hãy liệt kê tất cả các agent cần thiết theo đúng thứ tự logic
            {intent_hint}
            
            Hãy lập kế hoạch sử dụng các agent. Đầu tiên, trả lời với danh sách các agent cần sử dụng theo thứ tự, chỉ liệt kê tên các agent (filesystem, rag, metadata, text_extraction, file_classification), cách nhau bằng dấu phẩy.
            
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
            valid_agents = ["filesystem", "rag", "metadata", "text_extraction", "file_classification"]
            
            for agent in valid_agents:
                if agent in agent_list:
                    needed_agents.append(agent)
            
            if not needed_agents:
                # Mặc định sử dụng filesystem nếu không có agent nào được chọn
                needed_agents.append("filesystem")
                plan_message += "\nTôi sẽ bắt đầu với Filesystem Agent để tìm kiếm thông tin."
            
            log(f"Kế hoạch agent dựa trên LLM: {needed_agents}")
            
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
            log(f"Kế hoạch agent ban đầu: {state['current_agents']}")
            
            # Validate and fix agent sequence if needed
            state = await self._validate_agent_sequence(state)
            log(f"Kế hoạch agent sau khi kiểm tra: {state['current_agents']}")
            state["chain_of_thought"].append(f"2. Lập kế hoạch sử dụng các agent: {', '.join(state['current_agents'])}")
            
            # Run the agents in the planned order
            step_count = 3
            agent_execution_order = []
            
            while not state.get("completed", False) and state["current_agents"]:
                agent_name = state["current_agents"].pop(0)
                agent_execution_order.append(agent_name)
                log(f"Running agent: {agent_name} (Thứ tự thực thi: {agent_execution_order})")
                state["chain_of_thought"].append(f"{step_count}. Đang chạy agent: {agent_name}")
                
                # Add execution order to state for later analysis
                if "agent_execution_order" not in state:
                    state["agent_execution_order"] = []
                state["agent_execution_order"].append(agent_name)
                
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
                elif agent_name == "rag":
                    state = await self.run_rag_agent(state)
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
            
            # Generate execution summary
            agent_summary = ""
            if "agent_execution_order" in state:
                agent_summary = f"Thứ tự thực thi các agent: {', '.join(state['agent_execution_order'])}"
                log(f"Agent execution summary: {agent_summary}")
                state["chain_of_thought"].append(f"Tóm tắt thực thi: {agent_summary}")
            
            # Add used tools to the summary
            log(f"Used tools: {state.get('used_tools', [])}")
            
            # Return the final state with execution summary
            return {
                "response_type": "data",
                "is_task_complete": True,
                "require_user_input": False,
                "content": state["messages"][-1].content if state["messages"] else "",
                "state": state,
                "chain_of_thought": state["chain_of_thought"],
                "agent_execution_order": state.get("agent_execution_order", []),
                "used_tools": state.get("used_tools", [])
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
