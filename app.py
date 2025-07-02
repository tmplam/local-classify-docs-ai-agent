import gradio as gr
import asyncio
import time
import nest_asyncio
import platform
import os
import sys
import traceback
import threading
from typing import Optional, List, Tuple
import uuid
import logging
from contextlib import asynccontextmanager
import atexit
import gc
import warnings
from dotenv import load_dotenv
load_dotenv()

# CRITICAL: Fix for Python 3.13 compatibility - Apply BEFORE any other imports
# This must be the FIRST thing we do to avoid the inspect.cleandoc error
import inspect

# Completely override the problematic cleandoc function
def safe_cleandoc(doc):
    """Safe implementation of inspect.cleandoc that handles all edge cases"""
    # Handle None input
    if doc is None:
        return ''
    
    # Handle list inputs - convert to string first
    if isinstance(doc, list):
        if not doc:
            return ''
        try:
            # Join list elements with newlines, handling nested structures
            doc = '\n'.join(str(item) for item in doc)
        except Exception:
            return ''
    
    # Handle dict inputs
    if isinstance(doc, dict):
        try:
            doc = str(doc)
        except Exception:
            return ''
    
    # Convert to string if not already
    if not isinstance(doc, str):
        try:
            doc = str(doc)
        except Exception:
            return ''
    
    # Handle empty string
    if not doc or not doc.strip():
        return ''
        
    try:
        # Ensure we have a string before calling expandtabs()
        if not hasattr(doc, 'expandtabs'):
            doc = str(doc)
        
        lines = doc.expandtabs().splitlines()
        if not lines:
            return ''
        
        # Find minimum indentation (skip first line)
        indent = float('inf')
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped:
                indent = min(indent, len(line) - len(stripped))
        
        # Build result
        trimmed = [lines[0].strip()]
        if indent != float('inf') and indent > 0:
            for line in lines[1:]:
                if len(line) >= indent:
                    trimmed.append(line[indent:].rstrip())
                else:
                    trimmed.append(line.rstrip())
        else:
            for line in lines[1:]:
                trimmed.append(line.rstrip())
        
        # Remove trailing empty lines
        while trimmed and not trimmed[-1]:
            trimmed.pop()
        
        return '\n'.join(trimmed)
        
    except Exception as e:
        # Final fallback - just return string representation
        try:
            result = str(doc).strip()
            return result
        except Exception:
            return ''

# Force override the inspect.cleandoc function
inspect.cleandoc = safe_cleandoc

# Also monkey patch it in case it's already been imported elsewhere
import builtins
if hasattr(builtins, 'cleandoc'):
    builtins.cleandoc = safe_cleandoc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="grpc")

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import MultiAgentSystem
try:
    from agents.multi_agents import MultiAgentSystem
except ImportError as e:
    print(f"Cannot import MultiAgentSystem: {e}")
    
    class MockMultiAgentSystem:
        def __init__(self):
            self.initialized = False
            
        async def initialize(self):
            await asyncio.sleep(1)
            self.initialized = True
            return True
            
        async def run(self, message, request_id):
            await asyncio.sleep(0.5)
            return {
                "content": f"Mock response to: {message}",
                "chain_of_thought": [
                    "Received user message",
                    "Processing with mock system",
                    "Generating response"
                ]
            }
    
    MultiAgentSystem = MockMultiAgentSystem
    print("Using mock MultiAgentSystem for demonstration")

# Event loop configuration for Windows
if platform.system() == 'Windows':
    try:
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        else:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        print(f"Cannot set event loop policy: {e}")

nest_asyncio.apply()

def safe_str_conversion(data):
    """Safely convert any data to string, handling lists and complex structures"""
    if data is None:
        return ""
    
    if isinstance(data, str):
        return data
    
    if isinstance(data, list):
        try:
            # Handle list of strings
            if all(isinstance(item, str) for item in data):
                return '\n'.join(data)
            # Handle mixed list
            return '\n'.join(str(item) for item in data)
        except Exception:
            return str(data)
    
    if isinstance(data, dict):
        try:
            # Try to get content field first
            if 'content' in data:
                return safe_str_conversion(data['content'])
            # Otherwise convert whole dict
            return str(data)
        except Exception:
            return str(data)
    
    try:
        return str(data)
    except Exception:
        return "<Unable to convert to string>"

def safe_progress_update(progress, value, desc=""):
    """Safely update progress without causing errors"""
    try:
        if progress is not None and hasattr(progress, '__call__'):
            progress(value, desc=desc)
        return True
    except Exception as e:
        logger.warning(f"Progress update error: {e}")
        return False

class SingletonAsyncSystem:
    """
    Singleton pattern for managing async system - keeps event loop alive
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.system = None
        self.loop = None
        self.loop_thread = None
        self.initialized = False
        self.initialization_error = None
        self._initialized = True
        
    def start_event_loop(self):
        """Start persistent event loop in background thread"""
        if self.loop and not self.loop.is_closed():
            return
            
        def run_loop():
            try:
                # Create new event loop for this thread
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                
                # Keep loop running forever
                self.loop.run_forever()
            except Exception as e:
                logger.error(f"Event loop error: {e}")
            finally:
                if self.loop and not self.loop.is_closed():
                    self.loop.close()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # Wait for loop to be ready
        timeout = 10
        start_time = time.time()
        while (not self.loop or not self.loop.is_running()) and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self.loop or not self.loop.is_running():
            raise RuntimeError("Failed to start event loop")
    
    async def _initialize_system(self):
        """Internal method to initialize the system"""
        try:
            if self.system is None:
                self.system = MultiAgentSystem()
            
            logger.info("Initializing multi-agent system...")
            result = await self.system.initialize()
            
            if result:
                self.initialized = True
                self.initialization_error = None
                logger.info("System initialized successfully!")
                return True, None
            else:
                error_msg = "System initialization returned False"
                self.initialization_error = error_msg
                return False, error_msg
                
        except Exception as e:
            error_msg = f"System initialization error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.initialization_error = error_msg
            return False, error_msg
    
    def initialize_system(self):
        """Initialize the system using persistent event loop"""
        if not self.loop or not self.loop.is_running():
            self.start_event_loop()
        
        try:
            # Schedule initialization on the persistent loop
            future = asyncio.run_coroutine_threadsafe(
                self._initialize_system(), 
                self.loop
            )
            
            # Wait for result with timeout
            success, error = future.result(timeout=150)
            return success, error
            
        except Exception as e:
            error_msg = f"Failed to initialize system: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg
    
    async def _process_message(self, message, request_id):
        """Internal method to process message with enhanced error handling"""
        try:
            if not self.initialized or not self.system:
                return {"error": "System not initialized"}
            
            logger.info(f"Processing message: {message[:100]}...")
            result = await self.system.run(message, request_id)
            
            # Enhanced result processing with safe conversion
            if result is None:
                return {"error": "No response received from system"}
            
            # Handle different response formats
            if isinstance(result, dict):
                # Ensure content is properly converted to string
                if 'content' in result:
                    result['content'] = safe_str_conversion(result['content'])
                
                # Handle chain_of_thought with safe access
                if 'chain_of_thought' in result and result['chain_of_thought'] is not None:
                    try:
                        if isinstance(result['chain_of_thought'], list):
                            result['chain_of_thought'] = [safe_str_conversion(step) for step in result['chain_of_thought']]
                        else:
                            result['chain_of_thought'] = [safe_str_conversion(result['chain_of_thought'])]
                    except Exception as cot_error:
                        logger.warning(f"Chain of thought processing error: {cot_error}")
                        result['chain_of_thought'] = ["Chain of thought processing error"]
                
                return result
            else:
                # If result is not a dict, wrap it
                content = safe_str_conversion(result)
                return {"content": content, "chain_of_thought": ["Response processed successfully"]}
            
        except Exception as e:
            error_msg = f"Message processing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}
    
    def process_message(self, message, request_id):
        """Process message using persistent event loop"""
        if not self.loop or not self.loop.is_running():
            return {"error": "Event loop not running"}
        
        try:
            # Schedule processing on the persistent loop
            future = asyncio.run_coroutine_threadsafe(
                self._process_message(message, request_id), 
                self.loop
            )
            
            # Wait for result with timeout
            result = future.result(timeout=200)
            return result
            
        except Exception as e:
            error_msg = f"Failed to process message: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}
    
    def reset_system(self):
        """Reset the entire system"""
        try:
            self.system = None
            self.initialized = False
            self.initialization_error = None
            
            # Force garbage collection
            gc.collect()
            
            return "System reset successfully. Please reinitialize."
            
        except Exception as e:
            logger.error(f"Reset error: {e}")
            return f"Reset error: {str(e)}"
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.loop and self.loop.is_running():
                # Schedule loop stop
                self.loop.call_soon_threadsafe(self.loop.stop)
                
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=5)
                
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

# Create singleton instance
async_system = SingletonAsyncSystem()

class MultiAgentChatbot:
    """
    Simplified chatbot class using singleton async system
    """
    
    def __init__(self):
        self.chat_history = []
        self.chain_of_thought = []
        
    def initialize_system(self):
        """Initialize the system"""
        return async_system.initialize_system()
    
    def get_system_status(self):
        """Get current system status"""
        if not async_system.initialized:
            if async_system.initialization_error:
                return f"❌ System Error: {async_system.initialization_error}", "error"
            else:
                return "⚪ System not initialized", "warning"
        else:
            return "✅ System Ready", "success"
    
    def reset_system(self):
        """Reset the system"""
        message = async_system.reset_system()
        self.chat_history = []
        self.chain_of_thought = []
        return message, [], [], "⚪ System reset. Please reinitialize."
    
    def process_message(self, message: str, history: List, progress=None):
        """Process a user message and return the response with enhanced error handling"""
        try:
            # Input validation
            if not async_system.initialized:
                return history, self.chain_of_thought, "❌ System not initialized. Please initialize first."
            
            if not message or not message.strip():
                return history, self.chain_of_thought, "Please enter a message."
            
            # Safe progress update
            safe_progress_update(progress, 0.1, "Starting request processing...")
            
            # Create new message dict in proper format for Gradio
            user_message = {"role": "user", "content": message}
            new_history = history + [user_message] if history is not None else [user_message]
            
            # Reset chain of thought for new request
            self.chain_of_thought = []
            
            # Create unique request ID
            request_id = f"request_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
            safe_progress_update(progress, 0.3, "Processing with multi-agent system...")
            
            logger.info(f"Processing message with request_id: {request_id}")
            
            # Process message using singleton system
            result = async_system.process_message(message, request_id)
            
            safe_progress_update(progress, 0.8, "Processing response...")
            
            logger.info(f"Received result type: {type(result)}")
            
            # Enhanced error handling for result
            if result is None:
                error_msg = "No response received from system"
                logger.error(error_msg)
                
                assistant_message = {"role": "assistant", "content": f"❌ {error_msg}"}
                new_history.append(assistant_message)
                
                self.chain_of_thought.append(f"❌ Error: {error_msg}")
                return new_history, self.chain_of_thought, ""
            
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "Unknown error occurred")
                logger.error(f"Error in result: {error_msg}")
                
                assistant_message = {"role": "assistant", "content": f"❌ {error_msg}"}
                new_history.append(assistant_message)
                
                self.chain_of_thought.append(f"❌ Error: {error_msg}")
                return new_history, self.chain_of_thought, ""
            
            # Get response content with safe conversion
            response_content = safe_str_conversion(result.get("content", "Sorry, I couldn't process your request."))
            
            # Update chain of thought with safe conversion and access
            try:
                if "chain_of_thought" in result and result["chain_of_thought"]:
                    if isinstance(result["chain_of_thought"], list):
                        self.chain_of_thought = [safe_str_conversion(step) for step in result["chain_of_thought"]]
                    else:
                        self.chain_of_thought = [safe_str_conversion(result["chain_of_thought"])]
                else:
                    self.chain_of_thought = ["Response processed successfully"]
            except Exception as cot_error:
                logger.warning(f"Chain of thought processing error: {cot_error}")
                self.chain_of_thought = ["Chain of thought processing completed with minor issues"]
            
            # Add assistant response to history
            assistant_message = {"role": "assistant", "content": response_content}
            new_history.append(assistant_message)
            
            safe_progress_update(progress, 1.0, "Complete!")
            
            logger.info("Message processed successfully")
            return new_history, self.chain_of_thought, ""
            
        except Exception as e:
            error_message = f"❌ Processing error: {str(e)}"
            logger.error(f"Exception in process_message: {e}")
            logger.error(traceback.format_exc())
            
            # Safe history handling
            try:
                assistant_message = {"role": "assistant", "content": error_message}
                if history is not None:
                    new_history = history + [{"role": "user", "content": message}, assistant_message]
                else:
                    new_history = [{"role": "user", "content": message}, assistant_message]
            except Exception as history_error:
                logger.error(f"History handling error: {history_error}")
                new_history = [{"role": "assistant", "content": error_message}]
            
            self.chain_of_thought = [f"❌ Error: {str(e)}"]
            
            return new_history, self.chain_of_thought, ""

# Create chatbot instance
chatbot = MultiAgentChatbot()

def initialize_system():
    """Initialize the system and return status"""
    success, error = chatbot.initialize_system()
    if success:
        return "✅ System initialized successfully!", "success"
    else:
        return f"❌ Initialization failed: {error}", "error"

def reset_system():
    """Reset the system"""
    message, history, chain, status = chatbot.reset_system()
    return message, [], [], "⚪ System reset. Please reinitialize."

def get_status():
    """Get current system status"""
    return chatbot.get_system_status()

def format_chain_of_thought(chain_of_thought):
    """Format chain of thought for display"""
    if not chain_of_thought:
        return "No processing steps yet."
    
    formatted_steps = []
    for i, step in enumerate(chain_of_thought, 1):
        step_str = safe_str_conversion(step)
        formatted_step = f"**Step {i}:** {step_str}"
        formatted_steps.append(formatted_step)
    
    return "\n\n".join(formatted_steps)

# Register cleanup on exit
atexit.register(async_system.cleanup)

# Custom CSS
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.status-success {
    color: #4CAF50;
    font-weight: bold;
}

.status-error {
    color: #f44336;
    font-weight: bold;
}

.status-warning {
    color: #ff9800;
    font-weight: bold;
}

.chain-of-thought {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    border-left: 4px solid #2196F3;
    font-size: 14px;
    max-height: 400px;
    overflow-y: auto;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="Multi-Agent Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Agent Chatbot System")
    gr.Markdown("Advanced AI chatbot with persistent event loop management and enhanced error handling")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot_interface = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True,
                avatar_images=("👤", "🤖"),
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Enter your question here...",
                    scale=4,
                    interactive=True
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                
            with gr.Row():
                clear_btn = gr.Button("Clear History", variant="secondary")
                
        with gr.Column(scale=1):
            gr.Markdown("## 🎛️ Control Panel")
            
            status_display = gr.Markdown("⚪ System not initialized")
            
            with gr.Row():
                init_btn = gr.Button("🔄 Initialize", variant="primary")
                reset_btn = gr.Button("🗑️ Reset", variant="secondary")
            
            with gr.Accordion("🔧 System Information", open=False):
                gr.Markdown(f"""
                **Python**: {sys.version.split()[0]}
                **Platform**: {platform.system()} {platform.release()}
                **Event Loop**: Persistent Background Thread
                **Error Handling**: Enhanced with Safe String Conversion
                """)
            
            gr.Markdown("## 🔍 Processing Details")
            chain_display = gr.Markdown("No processing steps yet.", elem_classes=["chain-of-thought"])
    
    # Event handlers
    def send_message(message, history):
        """Send message with proper error handling"""
        try:
            if not message or not message.strip():
                return history, "No processing steps yet.", ""
            
            new_history, chain, empty_msg = chatbot.process_message(message, history)
            formatted_chain = format_chain_of_thought(chain)
            return new_history, formatted_chain, empty_msg
        except Exception as e:
            logger.error(f"Send message error: {e}")
            error_msg = f"❌ Error sending message: {str(e)}"
            error_history = history + [{"role": "assistant", "content": error_msg}] if history else [{"role": "assistant", "content": error_msg}]
            return error_history, f"❌ Error: {str(e)}", ""
    
    def clear_history():
        return []
    
    def update_status():
        status, status_type = get_status()
        if status_type == "success":
            return f'<span class="status-success">{status}</span>'
        elif status_type == "error":
            return f'<span class="status-error">{status}</span>'
        else:
            return f'<span class="status-warning">{status}</span>'
    
    def initialize_and_update():
        message, status_type = initialize_system()
        if status_type == "success":
            status_html = f'<span class="status-success">{message}</span>'
        else:
            status_html = f'<span class="status-error">{message}</span>'
        return status_html
    
    def reset_and_update():
        message, history, chain, status = reset_system()
        return f'<span class="status-warning">{status}</span>', [], []
    
    # Wire up events
    msg.submit(
        send_message,
        inputs=[msg, chatbot_interface],
        outputs=[chatbot_interface, chain_display, msg]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    send_btn.click(
        send_message,
        inputs=[msg, chatbot_interface],
        outputs=[chatbot_interface, chain_display, msg]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    clear_btn.click(
        clear_history,
        outputs=[chatbot_interface]
    ).then(
        lambda: "No processing steps yet.",
        outputs=[chain_display]
    )
    
    init_btn.click(
        initialize_and_update,
        outputs=[status_display]
    )
    
    reset_btn.click(
        reset_and_update,
        outputs=[status_display, chatbot_interface, chain_display]
    )
    
    demo.load(
        update_status,
        outputs=[status_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )