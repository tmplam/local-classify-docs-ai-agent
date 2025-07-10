from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from tools import playwright_tools, other_tools
import uuid
import asyncio
from datetime import datetime
import os

load_dotenv(override=True)

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarifications, or the assistant is stuck")


class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        worker_llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("GPT_API_KEY"))
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        evaluator_llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("GPT_API_KEY"))
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    def worker(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.

AVAILABLE TOOLS:
- Web browsing tools (navigate websites, extract information)
- Python code execution (include print() statements for output)
- File management (read/write files)
- Academic research tools:
  * analyze_citations_by_year: Find papers citing a specific paper, sorted by year
  * search_academic_paper: Get detailed information about a paper
  * search_papers_by_author: Find papers by author name
- Web search and Wikipedia tools
- Push notifications

For academic research tasks, prioritize using the specialized academic tools (analyze_citations_by_year, search_academic_paper) over general web search.

The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUCCESS CRITERIA:
{state['success_criteria']}

RESPONSE FORMAT:
- If you have a question for the user, start with "Question: [your question]"
- If you've completed the task, provide the final answer without asking questions
- For academic tasks, provide comprehensive, well-formatted results with proper citations
"""
        
        if state.get("feedback_on_work"):
            system_message += f"""

PREVIOUS FEEDBACK:
Your previous response was rejected. Here's the feedback:
{state['feedback_on_work']}

Please address this feedback and continue working on the task to meet the success criteria.
"""
        
        # Add in the system message
        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True
        
        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages
        
        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)
        
        # Return updated state
        return {
            "messages": [response],
        }

    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"
        
    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation
        
    def evaluator(self, state: State) -> State:
        last_response = state["messages"][-1].content

        system_message = f"""You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. 

For academic research tasks, ensure the response includes:
- Accurate paper identification
- Comprehensive citation analysis
- Proper formatting and organization
- Year-based sorting (if requested)
- Relevant statistics and insights

Respond with your feedback, decision on success criteria completion, and whether more user input is needed."""
        
        user_message = f"""You are evaluating a conversation between the User and Assistant.

FULL CONVERSATION:
{self.format_conversation(state['messages'])}

SUCCESS CRITERIA:
{state['success_criteria']}

ASSISTANT'S FINAL RESPONSE:
{last_response}

EVALUATION REQUIREMENTS:
1. Does the response meet the success criteria?
2. Is more user input required (questions, clarifications, or if assistant is stuck)?
3. For academic tasks: Are results comprehensive, accurate, and well-formatted?

Give the Assistant benefit of the doubt for completed actions, but reject if significant work is missing.
"""
        
        if state["feedback_on_work"]:
            user_message += f"\nPREVIOUS FEEDBACK: {state['feedback_on_work']}\n"
            user_message += "If the Assistant is repeating mistakes, consider requiring user input."
        
        evaluator_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        new_state = {
            "messages": [{"role": "assistant", "content": f"Evaluator Feedback: {eval_result.feedback}"}],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed
        }
        return new_state

    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END})
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}

        state = {
            "messages": message,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False
        }
        result = await self.graph.ainvoke(state, config=config)
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply, feedback]

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())