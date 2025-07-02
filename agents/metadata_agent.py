import os
import sys
import json
import asyncio
import mcp
import pandas as pd
from datetime import datetime
import subprocess
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đảm bảo đường dẫn đúng
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from config.llm import gemini
from agents.base import BaseAgent
from schemas.agent_schema import ResponseFormat
from utils import _format_file_timestamp
from config.prompt import metadata_prompt
from utils.get_agent_response import get_agent_response
from typing import Any, AsyncIterable, Dict, Optional, List

# MCP Client imports
try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    print("Warning: MCP client not available. Install with: pip install mcp")
    MCP_AVAILABLE = False

# MemorySaver
memory = MemorySaver()

class MCPConnection:
    """Manages MCP server connection with improved error handling and thread safety"""
    
    def __init__(self):
        self.server_params = None
        self.server_process = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._connection_pool = queue.Queue(maxsize=5)
        self._pool_initialized = False
        
    def _run_async_in_thread(self, coro):
        """Run async function in a separate thread with its own event loop"""
        def run_in_new_loop():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error in async thread: {e}")
                return {"error": str(e)}
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            try:
                return future.result(timeout=30)
            except Exception as e:
                logger.error(f"Thread execution error: {e}")
                return {"error": str(e)}
    
    async def start_server(self):
        """Start MCP server with improved connection handling"""
        try:
            with self._lock:
                if self.is_connected:
                    return True
            
            # Get the absolute path to the MCP server
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            server_path = os.path.join(project_root, 'fast_mcp', 'mcp_metadata_server.py')
            
            print(f"🔍 Looking for MCP server at: {server_path}")
            
            if not os.path.exists(server_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join(os.path.dirname(__file__), 'mcp_metadata_server.py'),
                    os.path.join(project_root, 'mcp_metadata_server.py'),
                    os.path.join(current_dir, 'mcp_metadata_server.py')
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        server_path = alt_path
                        print(f"📍 Found server at: {server_path}")
                        break
                else:
                    raise FileNotFoundError(f"MCP server not found at any of: {[server_path] + alt_paths}")
            
            # Create server parameters
            params = StdioServerParameters(
                command="python",
                args=[server_path],
                env=os.environ.copy(),
                cwd=project_root
            )
            
            # Test connection
            try:
                print("🔄 Testing MCP server connection...")
                async with asyncio.timeout(15):
                    async with stdio_client(params) as streams:
                        async with mcp.ClientSession(*streams) as session:
                            await session.initialize()
                            
                            # List available tools to verify connection
                            tools = await session.list_tools()
                            tool_names = [tool.name for tool in tools.tools]
                            print(f"✅ Connected! Available tools: {tool_names}")
                            
                            # Test a simple tool call if available
                            if "list_all_metadata" in tool_names:
                                try:
                                    test_result = await session.call_tool("list_all_metadata", {})
                                    print(f"✅ Test call successful")
                                except Exception as test_e:
                                    print(f"⚠️ Test call failed but connection OK: {test_e}")
                
                with self._lock:
                    self.server_params = params
                    self.is_connected = True
                
                print("✅ MCP server connection established successfully")
                return True
                
            except asyncio.TimeoutError:
                print("❌ Connection timeout - server failed to start within 15 seconds")
                return False
            except Exception as e:
                print(f"❌ Connection test failed: {e}")
                import traceback
                print(traceback.format_exc())
                return False
            
        except Exception as e:
            print(f"❌ Failed to establish MCP connection: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def call_tool_sync(self, tool_name: str, arguments: dict):
        """Synchronous wrapper for MCP tool calls"""
        return self._run_async_in_thread(self._call_tool_async(tool_name, arguments))
    
    async def _call_tool_async(self, tool_name: str, arguments: dict):
        """Async implementation of tool call"""
        if not self.server_params:
            raise RuntimeError("MCP server not initialized. Call start_server() first.")
            
        max_retries = 2
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with asyncio.timeout(30):
                    async with stdio_client(self.server_params) as streams:
                        async with mcp.ClientSession(*streams) as session:
                            await session.initialize()
                            
                            # Call the tool
                            result = await session.call_tool(tool_name, arguments)
                            
                            # Parse the result
                            if result and hasattr(result, 'content') and result.content:
                                content_item = result.content[0]
                                if hasattr(content_item, 'text'):
                                    content = content_item.text
                                else:
                                    content = str(content_item)
                                
                                try:
                                    # Try to parse as JSON
                                    return json.loads(content) if content else {"error": "Empty response"}
                                except json.JSONDecodeError:
                                    # Return as is if not valid JSON
                                    return {"result": content}
                            else:
                                return {"error": "No content returned from tool"}
                                
            except asyncio.TimeoutError:
                logger.warning(f"Tool call timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return {"error": "Tool call timed out after multiple attempts"}
            except Exception as e:
                logger.error(f"Tool call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return {"error": f"Tool call failed: {str(e)}"}
    
    async def close(self):
        """Clean up resources"""
        try:
            with self._lock:
                self.is_connected = False
                self.server_params = None
            print("✅ MCP resources cleaned up successfully")
        except Exception as e:
            print(f"❌ Error cleaning up MCP resources: {e}")

# Global MCP connection
mcp_connection = MCPConnection()

@tool
def create_metadata(text: str, file_name: str, label: str):
    """
    Create a metadata dictionary for a given text document.
    
    Args:
        text: The document text content
        file_name: Name of the file
        label: Classification label for the document
    
    Returns:
        dict: Metadata dictionary with document information
    """
    metadata = {
        "total_characters": len(text),
        "creation_date": _format_file_timestamp(
            timestamp=datetime.now().timestamp(), include_time=True
        ),
        "file_name": str(file_name),
        "label": label,
        "content": text[:500] if len(text) > 500 else text,
    }
    return metadata

@tool
def save_metadata_to_mcp(metadata: dict):
    """
    Save metadata to MCP server using proper MCP protocol.
    
    Args:
        metadata: Dictionary containing metadata to save
    
    Returns:
        dict: Result of the save operation
    """
    try:
        # Extract required fields from metadata
        filename = metadata.get("file_name", "unknown_file")
        label = metadata.get("label", "unclassified")
        content = metadata.get("content", "")
        
        # Debug output
        print(f"💾 Saving metadata for file: {filename}, label: {label}")
        
        # Remove these fields from additional_metadata to avoid duplication
        additional_metadata = metadata.copy()
        for key in ["file_name", "label", "content"]:
            if key in additional_metadata:
                del additional_metadata[key]
        
        # Call MCP server tool synchronously
        result = mcp_connection.call_tool_sync("save_metadata_to_json", {
            "filename": filename,
            "label": label,
            "content": content,
            "additional_metadata": additional_metadata
        })
        
        print(f"📊 Raw MCP result: {result}")
        
        # Check for explicit error field
        if isinstance(result, dict) and "error" in result:
            print(f"❌ MCP reported error: {result['error']}")
            return {
                "status": "error",
                "message": f"MCP server error: {result['error']}",
                "details": result.get("traceback", "")
            }
        
        # Handle string result (which is the expected format from MCP server)
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                print(f"✅ Successfully parsed JSON result")
                return {
                    "status": "success",
                    "message": "Metadata saved to MCP server successfully",
                    "metadata_id": parsed_result.get("metadata", {}).get("id", ""),
                    "result": parsed_result
                }
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse JSON: {e}")
        
        # Handle case where result is already a dict with result field
        if isinstance(result, dict) and "result" in result:
            try:
                if isinstance(result["result"], str):
                    parsed_inner = json.loads(result["result"])
                    print(f"✅ Successfully parsed nested JSON result")
                    return {
                        "status": "success",
                        "message": "Metadata saved to MCP server successfully",
                        "metadata_id": parsed_inner.get("metadata", {}).get("id", ""),
                        "result": parsed_inner
                    }
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Failed to parse nested JSON: {e}")
        
        # Return the raw result if all else fails
        print(f"ℹ️ Returning raw result")
        return {
            "status": "success",
            "message": "Metadata saved to MCP server successfully",
            "result": result
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Failed to save metadata: {str(e)}",
            "details": traceback.format_exc()
        }

@tool
def search_metadata_in_mcp(filename: Optional[str] = None, label: Optional[str] = None):
    """
    Search for metadata in the MCP server.
    
    Args:
        filename: Optional filename to search for
        label: Optional label to search for
    
    Returns:
        dict: Search results from MCP server
    """
    try:
        # Debug output
        search_params = []
        if filename:
            search_params.append(f"filename='{filename}'")
        if label:
            search_params.append(f"label='{label}'")
        search_desc = " and ".join(search_params) if search_params else "all records"
        print(f"🔍 Searching metadata for {search_desc}")
        
        # Call MCP server tool synchronously
        result = mcp_connection.call_tool_sync("search_metadata", {
            "filename": filename,
            "label": label
        })
        
        print(f"📊 Raw search result: {result}")
        
        # Check for explicit error field
        if isinstance(result, dict) and "error" in result:
            print(f"❌ MCP search error: {result['error']}")
            return {
                "status": "error",
                "message": f"MCP server error: {result['error']}"
            }
        
        # Handle string result (which is the expected format from MCP server)
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                print(f"✅ Successfully parsed search JSON result")
                return parsed_result
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse search JSON: {e}")
        
        # Handle case where result is already a dict with result field
        if isinstance(result, dict) and "result" in result:
            try:
                if isinstance(result["result"], str):
                    parsed_inner = json.loads(result["result"])
                    print(f"✅ Successfully parsed nested search JSON result")
                    return parsed_inner
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Failed to parse nested search JSON: {e}")
        
        # Return the raw result if all else fails
        print(f"ℹ️ Returning raw search result")
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Search error: {e}\n{error_details}")
        return {
            "status": "error",
            "message": f"Failed to search metadata: {str(e)}",
            "details": error_details
        }

@tool
def get_metadata_from_mcp(metadata_id: str):
    """
    Get metadata from MCP server by ID.
    
    Args:
        metadata_id: ID of the metadata to retrieve
    
    Returns:
        dict: Metadata from MCP server
    """
    try:
        # Debug output
        print(f"🌐 Getting metadata with ID: {metadata_id}")
        
        # Call MCP server tool synchronously
        result = mcp_connection.call_tool_sync("get_metadata", {
            "metadata_id": metadata_id
        })
        
        print(f"📊 Raw get result: {result}")
        
        # Check for explicit error field
        if isinstance(result, dict) and "error" in result:
            print(f"❌ MCP get error: {result['error']}")
            return {
                "status": "error",
                "message": f"MCP server error: {result['error']}"
            }
        
        # Handle string result (which is the expected format from MCP server)
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                print(f"✅ Successfully parsed get JSON result")
                return parsed_result
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse get JSON: {e}")
        
        # Handle case where result is already a dict with result field
        if isinstance(result, dict) and "result" in result:
            try:
                if isinstance(result["result"], str):
                    parsed_inner = json.loads(result["result"])
                    print(f"✅ Successfully parsed nested get JSON result")
                    return parsed_inner
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Failed to parse nested get JSON: {e}")
        
        # Return the raw result if all else fails
        print(f"ℹ️ Returning raw get result")
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Get error: {e}\n{error_details}")
        return {
            "status": "error",
            "message": f"Failed to get metadata: {str(e)}",
            "details": error_details
        }

@tool
def list_all_metadata_from_mcp():
    """
    List all metadata from MCP server.
    
    Returns:
        dict: All metadata from MCP server
    """
    try:
        # Debug output
        print(f"📃 Listing all metadata from MCP server")
        
        # Call MCP server tool synchronously
        result = mcp_connection.call_tool_sync("list_all_metadata", {})
        
        print(f"📊 Raw list result: {result}")
        
        # Check for explicit error field
        if isinstance(result, dict) and "error" in result:
            print(f"❌ MCP list error: {result['error']}")
            return {
                "status": "error",
                "message": f"MCP server error: {result['error']}"
            }
        
        # Handle string result (which is the expected format from MCP server)
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                print(f"✅ Successfully parsed list JSON result")
                return parsed_result
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse list JSON: {e}")
        
        # Handle case where result is already a dict with result field
        if isinstance(result, dict) and "result" in result:
            try:
                if isinstance(result["result"], str):
                    parsed_inner = json.loads(result["result"])
                    print(f"✅ Successfully parsed nested list JSON result")
                    return parsed_inner
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Failed to parse nested list JSON: {e}")
        
        # Return the raw result if all else fails
        print(f"ℹ️ Returning raw list result")
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ List error: {e}\n{error_details}")
        return {
            "status": "error",
            "message": f"Failed to list metadata: {str(e)}",
            "details": error_details
        }

class MetadataAgent(BaseAgent):
    """Metadata Agent backed by LangGraph with improved MCP integration."""
    
    @property
    def name(self):
        return self.agent_name

    def __init__(self):
        super().__init__(
            agent_name='MetadataAgent',
            description='Create metadata for documents and save to MCP server JSON file',
            content_types=['text', 'text/plain']
        )

        self.model = gemini
        self.mcp_initialized = False

        self.graph = create_react_agent(
            self.model,
            checkpointer=memory,
            prompt=metadata_prompt,
            response_format=ResponseFormat,
            tools=[
                create_metadata, 
                save_metadata_to_mcp, 
                search_metadata_in_mcp, 
                get_metadata_from_mcp,
                list_all_metadata_from_mcp
            ],
            name="Metadata Agent",
        )

    def initialize_mcp_sync(self):
        """Synchronous MCP initialization"""
        if not self.mcp_initialized and MCP_AVAILABLE:
            print("🔌 Initializing MCP connection...")
            try:
                # Use the synchronous wrapper
                success = mcp_connection._run_async_in_thread(mcp_connection.start_server())
                if isinstance(success, dict) and "error" in success:
                    print(f"❌ MCP initialization failed: {success['error']}")
                    return False
                
                self.mcp_initialized = success
                if success:
                    print("✅ MCP connection initialized successfully")
                else:
                    print("❌ Failed to initialize MCP connection")
                return success
            except Exception as e:
                print(f"❌ Error initializing MCP: {e}")
                return False
        return self.mcp_initialized

    def invoke(self, query, sessionId) -> str:
        """Invoke the agent synchronously with better error handling."""
        # Initialize MCP if not done
        if not self.mcp_initialized:
            success = self.initialize_mcp_sync()
            if not success:
                return "❌ Failed to initialize MCP connection. Please check the server."
            
        config = {'configurable': {'thread_id': sessionId}, 'recursion_limit': 50}
        try:
            response = self.graph.invoke({'messages': [('user', query)]}, config)
            last_message = response['messages'][-1]
            
            if isinstance(last_message, AIMessage):
                content = last_message.content
            elif hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = str(last_message)
                
            return content
        except Exception as e:
            import traceback
            error_msg = f"Error processing request: {str(e)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())
            return error_msg
    
    async def stream(self, query, sessionId, task_id) -> AsyncIterable[Dict[str, Any]]:
        """Stream responses from the agent."""
        # Initialize MCP if not done
        if not self.mcp_initialized:
            success = await mcp_connection.start_server()
            self.mcp_initialized = success
            
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': sessionId}, 'recursion_limit': 50}

        try:
            for item in self.graph.stream(inputs, config, stream_mode='values'):
                message = item['messages'][-1]
                if isinstance(message, AIMessage):
                    yield {
                        'response_type': 'text',
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': message.content,
                    }
            
            yield {
                'response_type': 'data', 
                'is_task_complete': True,
                'require_user_input': False,
                'content': message.content if isinstance(message, AIMessage) else str(message)
            }
        except Exception as e:
            yield {
                'response_type': 'error',
                'is_task_complete': True,
                'require_user_input': False,
                'content': f"Error in streaming: {str(e)}"
            }
                    
    async def ainvoke(self, input_data, config=None):
        """Required method for langgraph-supervisor compatibility"""
        # Initialize MCP if not done
        if not self.mcp_initialized:
            await mcp_connection.start_server()
            self.mcp_initialized = True
            
        if config is None:
            config = {}
            
        try:
            messages = input_data.get("messages", [])
            if messages and len(messages) > 0:
                last_message = messages[-1]
                
                if hasattr(last_message, 'content'):
                    query = last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    query = last_message['content']
                elif isinstance(last_message, tuple) and len(last_message) == 2:
                    query = last_message[1]
                else:
                    query = str(last_message)
                    
                session_id = config.get("configurable", {}).get("thread_id", "default")
                result = self.invoke(query, session_id)
                
                if isinstance(result, dict):
                    result = str(result)
                    
                from langchain_core.messages import AIMessage
                return {"messages": [AIMessage(content=result)]}
        except Exception as e:
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=f"Error in ainvoke: {str(e)}")]}

    def __del__(self):
        """Cleanup MCP connection on deletion"""
        try:
            # Don't create new event loop in destructor
            pass
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


if __name__ == "__main__":
    def test_metadata_agent():
        print("🧪 Testing MetadataAgent with improved MCP server integration...")
        
        # Create agent
        metadata_agent = MetadataAgent()
        
        # Initialize MCP connection
        print("🔌 Initializing MCP connection...")
        success = metadata_agent.initialize_mcp_sync()
        
        if not success:
            print("❌ Failed to connect to MCP server. Please check:")
            print("1. MCP server file exists")
            print("2. MCP server dependencies are installed")
            print("3. No firewall/permission issues")
            return
        
        # Test prompt
        prompt = """
        Hãy tạo metadata cho tài liệu sau:
        - File Name: Chain_of_thought.pdf
        - Label: Chain of Thought
        - Text: This is a document about the chain of thought. The document is a PDF file containing research about reasoning methods in artificial intelligence. It discusses various approaches to breaking down complex problems into step-by-step reasoning processes.
        
        Sau đó, lưu metadata vào MCP server và báo cáo kết quả chi tiết.
        """
        
        try:
            print("📝 Calling MetadataAgent to process request...")
            result = metadata_agent.invoke(query=prompt, sessionId="test_session")
            print("\n✅ Agent result:")
            print(result)
            
            
        except Exception as e:
            import traceback
            print(f"❌ Error during testing: {e}")
            print(traceback.format_exc())
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            # Note: cleanup is handled automatically
    
    # Run the synchronous test
    test_metadata_agent()