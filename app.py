import os
import sys
import streamlit as st
import asyncio
import time
import threading
import types
import grpc.aio
import uuid
import logging
import warnings

from agents.multi_agents import create_supervisor_agent, process
from utils.db_manager import ChatHistoryDB

# Suppress schema validation warnings
logging.getLogger('jsonschema').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*additionalProperties.*')
warnings.filterwarnings('ignore', message=r'.*\$schema.*')

# Fix for gRPC InterceptedCall.__del__ error
# Monkey patch để tránh lỗi InterceptedCall.__del__
original_del = grpc.aio._interceptor.InterceptedCall.__del__
def safe_del(self):
    try:
        original_del(self)
    except Exception as e:
        pass
grpc.aio._interceptor.InterceptedCall.__del__ = safe_del

# Thêm đường dẫn gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.multi_agents import create_supervisor_agent, process
from utils.pretty_print_message import pretty_print_messages

# Set page config
st.set_page_config(
    page_title="AI Document Classifier Chat",
    page_icon="📄",
    layout="wide"
)

# Khởi tạo cơ sở dữ liệu
db = ChatHistoryDB()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "supervisor" not in st.session_state:
    st.session_state.supervisor = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    
if "current_session_id" not in st.session_state:
    # Tạo session ID mới cho phiên chat hiện tại
    st.session_state.current_session_id = f"session_{uuid.uuid4().hex[:8]}"
    # Tạo phiên chat mới trong cơ sở dữ liệu
    db.create_session(st.session_state.current_session_id, title="Chat mới")

if "chat_sessions" not in st.session_state:
    # Lấy danh sách các phiên chat từ cơ sở dữ liệu
    st.session_state.chat_sessions = db.get_all_sessions()

# Initialize agent
async def initialize_agents():
    try:
        with st.spinner("Đang khởi tạo agents..."):
            supervisor = await create_supervisor_agent()
            st.session_state.supervisor = supervisor
            st.session_state.initialized = True
            print(f"Supervisor type: {type(supervisor)}")
            return "Khởi tạo thành công!"
    except Exception as e:
        return f"Lỗi khởi tạo agents: {str(e)}"

# Process user query
async def process_query(supervisor, query):
    """Process a query using the supervisor agent and yield streaming responses."""
    try:
        print(f"\n[app.process_query] Nhận query: '{query}'")
        print(f"[app.process_query] Supervisor type: {type(supervisor)}")
        
        # Gọi hàm process từ multi_agents.py để xử lý query
        result = await process(supervisor, query)
        print(f"[app.process_query] Đã nhận kết quả từ process(): {type(result)}")
        
        # Biến để theo dõi các phản hồi đã xuất
        yielded_responses = set()
        
        # Thêm hàm an toàn để cập nhật UI Streamlit
        def safe_streamlit_update(func):
            try:
                return func()
            except st.errors.NoSessionContext:
                print("[app.process_query] Bỏ qua cập nhật UI do không có session context")
                return None
        
        # Xử lý kết quả streaming
        async for chunk in result:
            print(f"\n[app.process_query] Nhận chunk: {type(chunk)}")
            
            try:
                # Xử lý AddableUpdatesDict từ LangGraph
                if hasattr(chunk, 'get') and callable(chunk.get):
                    # Trường hợp đặc biệt: AddableUpdatesDict từ LangGraph
                    for agent_name in chunk.keys():
                        print(f"[app.process_query] Xử lý agent: {agent_name}")
                        agent_data = chunk.get(agent_name)
                        
                        if isinstance(agent_data, dict) and 'messages' in agent_data:
                            messages = agent_data['messages']
                            if isinstance(messages, list) and len(messages) > 0:
                                for msg in messages:
                                    if hasattr(msg, 'content'):
                                        content = msg.content
                                        # Kiểm tra nội dung đã xuất chưa
                                        if content not in yielded_responses:
                                            print(f"[app.process_query] Phản hồi mới từ {agent_name}: {content[:100]}...")
                                            yield f"{agent_name}: {content}"
                                            safe_streamlit_update(lambda: yielded_responses.add(content))
                
                # Xử lý các loại chunk khác nhau
                elif isinstance(chunk, dict):
                    # Trường hợp 1: chunk là dict với các agent name là key
                    for agent_name, messages in chunk.items():
                        print(f"[app.process_query] Agent: {agent_name}, Messages: {type(messages)}")
                        
                        if isinstance(messages, list):
                            for message in messages:
                                if hasattr(message, 'content'):
                                    # Nếu message là AIMessage hoặc có thuộc tính content
                                    content = message.content
                                    if content not in yielded_responses:
                                        print(f"Yielding response from {agent_name}: {content[:50]}...")
                                        yield f"{agent_name}: {content}"
                                        safe_streamlit_update(lambda: yielded_responses.add(content))
                                elif isinstance(message, dict) and 'content' in message:
                                    # Nếu message là dict với key 'content'
                                    content = message['content']
                                    if content not in yielded_responses:
                                        print(f"Yielding response from {agent_name} (dict): {content[:50]}...")
                                        yield f"{agent_name}: {content}"
                                        safe_streamlit_update(lambda: yielded_responses.add(content))
                        elif hasattr(messages, 'content'):
                            # Nếu messages là một message duy nhất
                            content = messages.content
                            if content not in yielded_responses:
                                print(f"Yielding response from {agent_name} (single): {content[:50]}...")
                                yield f"{agent_name}: {content}"
                                safe_streamlit_update(lambda: yielded_responses.add(content))
                        elif isinstance(messages, str):
                            # Nếu messages là string
                            if messages not in yielded_responses:
                                print(f"Yielding response from {agent_name} (string): {messages[:50]}...")
                                yield f"{agent_name}: {messages}"
                                safe_streamlit_update(lambda: yielded_responses.add(messages))
                elif hasattr(chunk, 'messages') and isinstance(chunk.messages, list):
                    # Trường hợp 2: chunk có thuộc tính messages là list
                    for message in chunk.messages:
                        if hasattr(message, 'content'):
                            content = message.content
                            agent = getattr(message, 'name', 'Agent')
                            if content not in yielded_responses:
                                print(f"Yielding response from {agent}: {content[:50]}...")
                                yield f"{agent}: {content}"
                                safe_streamlit_update(lambda: yielded_responses.add(content))
                elif isinstance(chunk, str):
                    # Trường hợp 3: chunk là string
                    if chunk not in yielded_responses:
                        print(f"Yielding direct string response: {chunk[:50]}...")
                        yield chunk
                        safe_streamlit_update(lambda: yielded_responses.add(chunk))
                else:
                    # Trường hợp khác: chunk không xử lý được
                    chunk_str = str(chunk)
                    if chunk_str not in yielded_responses:
                        print(f"[app.process_query] Không xử lý được chunk: {type(chunk)}")
                        yield f"Phản hồi từ hệ thống: {chunk_str}"
                        safe_streamlit_update(lambda: yielded_responses.add(chunk_str))
            except Exception as chunk_error:
                print(f"[app.process_query] Lỗi khi xử lý chunk: {str(chunk_error)}")
                import traceback
                traceback.print_exc()
                yield f"Lỗi khi xử lý phản hồi: {str(chunk_error)}"
    
    except Exception as e:
        print(f"[app.process_query] Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"Lỗi khi xử lý yêu cầu: {str(e)}"

# Sidebar
st.sidebar.title("📄 Document Classifier")
st.sidebar.info("Ứng dụng này sử dụng AI để phân loại và trích xuất thông tin từ tài liệu.")

# Thêm phần lịch sử chat
with st.sidebar.expander("Lịch sử chat", expanded=False):
    # Nút tạo chat mới
    if st.button("Tạo chat mới", key="new_chat"):
        # Tạo session ID mới
        st.session_state.current_session_id = f"session_{uuid.uuid4().hex[:8]}"
        # Tạo phiên chat mới trong cơ sở dữ liệu
        db.create_session(st.session_state.current_session_id, title="Chat mới")
        # Xóa tin nhắn hiện tại
        st.session_state.messages = []
        # Cập nhật danh sách phiên chat
        st.session_state.chat_sessions = db.get_all_sessions()
        st.rerun()
    
    # Hiển thị danh sách các phiên chat
    st.subheader("Các phiên chat trước đây")
    
    # Cập nhật danh sách phiên chat
    st.session_state.chat_sessions = db.get_all_sessions()
    
    # Hiển thị từng phiên chat dưới dạng nút
    for session in st.session_state.chat_sessions:
        session_title = session['title'] or f"Chat {session['id']}"
        if st.button(f"{session_title} ({session['created_at'][:10]})", key=f"session_{session['session_id']}"):
            # Đổi sang phiên chat được chọn
            st.session_state.current_session_id = session['session_id']
            # Lấy tin nhắn từ phiên chat này
            messages = db.get_session_messages(session['session_id'])
            # Chuyển đổi định dạng tin nhắn
            st.session_state.messages = [{"role": msg['role'], "content": msg['content']} for msg in messages]
            st.rerun()

# Thêm thông tin về dự án
with st.sidebar.expander("Thông tin dự án"):
    st.write("""
    ### Ứng dụng Chat AI chạy Local LLM tích hợp MCP
    
    **Chức năng chính:**
    - Tìm kiếm nội dung trong thư mục file
    - Phân loại file dựa trên nội dung
    - Gửi metadata file qua API MCP Cloud
    - Hiển thị kết quả trên giao diện chat
    - Lưu trữ lịch sử chat vào SQLite
    """)

# Main content
st.title("📄 AI Document Classifier Chat")

# Initialize button
if not st.session_state.initialized:
    if st.sidebar.button("Khởi tạo Agents"):
        init_message = asyncio.run(initialize_agents())
        st.sidebar.success(init_message)
        # Thêm tin nhắn chào mừng
        welcome_msg = "Xin chào! Tôi là AI Assistant. Tôi có thể giúp bạn tìm kiếm, phân loại và trích xuất thông tin từ tài liệu. Hãy nhập yêu cầu của bạn."
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.initialized:
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Lưu tin nhắn vào lịch sử và cơ sở dữ liệu
        st.session_state.messages.append({"role": "user", "content": prompt})
        db.add_message(st.session_state.current_session_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Xử lý câu hỏi và hiển thị phản hồi
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Xử lý streaming response
            response_text = [""]
            thinking_state = {"active": True}  # Sử dụng dict để có thể thay đổi giá trị từ bên trong hàm
            thinking_dots = ["", ".", "..", "..."]
            
            # Hàm an toàn để cập nhật UI Streamlit cho animation
            def safe_animation_update(content):
                try:
                    message_placeholder.markdown(content)
                    return True
                except st.errors.NoSessionContext:
                    print("[show_thinking_animation] Bỏ qua cập nhật UI do không có session context")
                    return False
                
            # Hàm hiển thị animation "Chain of Thought"
            def show_thinking_animation():
                thinking_idx = 0
                while thinking_state["active"]:
                    safe_animation_update(f"*Đang suy nghĩ{thinking_dots[thinking_idx % len(thinking_dots)]}*")
                    thinking_idx += 1
                    time.sleep(0.3)
            
            # Bắt đầu animation trong thread riêng
            thinking_thread = threading.Thread(target=show_thinking_animation)
            thinking_thread.daemon = True
            thinking_thread.start()
            
            async def process_streaming_response():
                try:
                    # Sử dụng hàm process_query để xử lý câu hỏi
                    print("Starting process_query with supervisor...")
                    
                    # Kiểm tra supervisor đã được khởi tạo
                    if not hasattr(st.session_state, 'supervisor') or st.session_state.supervisor is None:
                        thinking_state["active"] = False
                        time.sleep(0.5)  # Đợi animation dừng
                        response_text[0] = "Hệ thống chưa được khởi tạo. Vui lòng khởi động lại ứng dụng."
                        message_placeholder.markdown(response_text[0])
                        return
                    
                    # Gọi process_query với supervisor và prompt
                    first_chunk = True
                    async for response_chunk in process_query(st.session_state.supervisor, prompt):
                        if first_chunk:
                            # Dừng animation khi có phản hồi đầu tiên
                            thinking_state["active"] = False
                            time.sleep(0.5)  # Đợi animation dừng
                            response_text[0] = ""  # Xóa "Chain of Thought"
                            first_chunk = False
                        
                        response_text[0] += response_chunk + "\n\n"
                        message_placeholder.markdown(response_text[0])
                except Exception as e:
                    thinking_state["active"] = False
                    time.sleep(0.5)  # Đợi animation dừng
                    print(f"Error in process_streaming_response: {e}")
                    import traceback
                    traceback.print_exc()
                    response_text[0] += f"\n\nLỗi: {str(e)}"
                    message_placeholder.markdown(response_text[0])
            
            # Chạy hàm async - sử dụng event loop hiện tại nếu có thể
            try:
                # Thử lấy event loop hiện tại
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Nếu đã đóng, tạo mới
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(process_streaming_response())
            except Exception as e:
                print(f"Lỗi khi chạy event loop: {str(e)}")
                # Fallback nếu có lỗi
                asyncio.run(process_streaming_response())
        
        # Lưu phản hồi vào lịch sử và cơ sở dữ liệu
        st.session_state.messages.append({"role": "assistant", "content": response_text[0]})
        db.add_message(st.session_state.current_session_id, "assistant", response_text[0])
else:
    st.info("Vui lòng khởi tạo agents từ sidebar trước khi bắt đầu chat.")
