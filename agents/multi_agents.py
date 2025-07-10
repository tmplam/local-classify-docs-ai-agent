import os
import sys
import asyncio
import re
from typing import Dict, List, Any, Optional, Annotated, Tuple, Union

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
    chain_of_thought: List[str]  # Detailed execution steps
    agent_results: Dict[str, str]  # Results from each agent
    original_query: str  # Original user query for reflection

class ReflectionAgent:
    """
    Agent chuyên về reflection - tổng hợp và trả lời cuối cùng cho người dùng
    dựa trên kết quả từ các agent khác và query ban đầu.
    """
    
    def __init__(self):
        """Initialize the ReflectionAgent."""
        self.model = gemini
    
    async def reflect_and_respond(self, state: AgentState) -> str:
        """
        Phân tích kết quả từ các agent và tạo câu trả lời cuối cùng có ngữ nghĩa tốt.
        
        Args:
            state: Trạng thái hiện tại của hệ thống
            
        Returns:
            Câu trả lời cuối cùng được tối ưu hóa cho người dùng
        """
        try:
            # Import các module cần thiết
            import re
            import os
            # Lấy query ban đầu
            original_query = state.get("original_query", "")
            if not original_query:
                # Fallback: tìm trong messages
                for message in state["messages"]:
                    if isinstance(message, HumanMessage):
                        original_query = message.content
                        break
            
            # Thu thập kết quả từ các agent
            agent_results = state.get("agent_results", {})
            used_tools = state.get("used_tools", [])
            chain_of_thought = state.get("chain_of_thought", [])
            
            # Tạo tóm tắt các kết quả quan trọng
            key_findings = []
            file_found = None
            extraction_result = None
            classification_result = None
            metadata_ids = []
            
            # Thu thập thông tin chi tiết về file
            # Đảm bảo khởi tạo lại detailed_files mỗi lần gọi
            detailed_files = []
            classification_labels = state.get("classification_labels", {})
            
            # Xóa file_count cũ trong state nếu có
            if "file_count" in state:
                del state["file_count"]
            
            # Phân tích kết quả từ từng agent
            for message in state["messages"]:
                if not isinstance(message, AIMessage):
                    continue
                    
                content = message.content
                
                # Kết quả từ RAG/Filesystem agent - tìm file
                # Kiểm tra nhiều định dạng thông báo tìm thấy file
                file_found_indicators = [
                    "Tôi đã tìm thấy file:", 
                    "Tìm thấy các file sau:", 
                    "Đã tìm thấy các file:",
                    "Tìm thấy nhiều file:",
                    "files found:",
                    "found files:",
                    "kết quả tìm kiếm:",
                    "search results:",
                    "plan-"  # Thêm dấu hiệu tìm kiếm file có chứa "plan"
                ]
                
                # Kiểm tra nếu có bất kỳ indicator nào trong nội dung
                has_file_indicator = any(indicator.lower() in content.lower() for indicator in file_found_indicators)
                
                # Tìm tất cả các đường dẫn file trong nội dung
                file_pattern = r'[A-Z]:\\[^\\/:*?"<>|\r\n]+(?:\\[^\\/:*?"<>|\r\n]+)*\\?'
                file_matches = re.findall(file_pattern, content)
                
                # Lọc ra các đường dẫn hợp lệ
                files_found = [match.strip() for match in file_matches if os.path.exists(match.strip())]
                
                # Nếu tìm thấy file nhưng chưa có trong files_found, thử cách khác
                if has_file_indicator and not files_found:
                    # Thử tìm theo định dạng danh sách đánh số hoặc gạch đầu dòng
                    list_pattern = r'(?:\d+\.\s*|-\s*|\*\s*)([^\n\r]+)'
                    list_matches = re.findall(list_pattern, content)
                    if list_matches:
                        # Lọc ra các mục có vẻ giống đường dẫn file
                        potential_paths = [match.strip() for match in list_matches]
                        files_found = [path for path in potential_paths if os.path.exists(path)]
                
                # Nếu vẫn chưa tìm thấy, thử tìm bất kỳ chuỗi nào giống đường dẫn
                if not files_found and has_file_indicator:
                    # Tìm các chuỗi có chứa dấu chấm (đuôi file) và dấu gạch chéo
                    potential_paths = re.findall(r'[\w\\:]+\.[\w.]+', content)
                    files_found = [path for path in potential_paths if os.path.exists(path)]
                
                # Nếu tìm thấy file, xử lý kết quả
                if files_found:
                    # Lấy danh sách file từ state nếu có (đây là danh sách chính xác từ RAG agent)
                    # Hàm helper để kiểm tra và xử lý kết quả từ agent một cách an toàn
                    def safe_get_agent_result(agent_name):
                        if "agent_results" not in state or agent_name not in state["agent_results"]:
                            return {}
                        
                        agent_result = state["agent_results"][agent_name]
                        if isinstance(agent_result, dict):
                            return agent_result
                        elif isinstance(agent_result, str):
                            try:
                                # Thử parse JSON nếu là string
                                if '{' in agent_result and '}' in agent_result:
                                    import json
                                    parsed_result = json.loads(agent_result)
                                    log(f"ReflectionAgent debug - Parsed {agent_name} result: {parsed_result}")
                                    return parsed_result
                            except Exception as e:
                                log(f"ReflectionAgent debug - Failed to parse {agent_name} result: {str(e)}")
                        
                        # Trả về dict rỗng nếu không phải dict và không thể parse
                        log(f"ReflectionAgent debug - {agent_name} result is not a dict: {agent_result}")
                        return {}
                    
                    # Log state keys for debugging
                    log(f"ReflectionAgent debug - State keys: {list(state.keys())}")
                    if "agent_results" in state:
                        log(f"ReflectionAgent debug - Agent results keys: {list(state['agent_results'].keys())}")
                        
                        # Xử lý an toàn cho tất cả các agent results
                        rag_result = safe_get_agent_result("rag")
                        text_extraction_result = safe_get_agent_result("text_extraction")
                        file_classification_result = safe_get_agent_result("file_classification")
                        metadata_result = safe_get_agent_result("metadata")
                    
                    state_file_paths = []
                    
                    # Thử lấy từ agent_results.rag.file_paths
                    if isinstance(rag_result, dict) and "file_paths" in rag_result:
                        state_file_paths = rag_result["file_paths"]
                        log(f"ReflectionAgent debug - Found file_paths in rag_result: {state_file_paths}")
                    # Thử lấy trực tiếp từ state
                    elif "file_paths" in state:
                        state_file_paths = state["file_paths"]
                        log(f"ReflectionAgent debug - Found file_paths in state: {state_file_paths}")
                    # Thử lấy từ processed_files
                    elif "processed_files" in state:
                        state_file_paths = state["processed_files"]
                        log(f"ReflectionAgent debug - Found processed_files in state: {state_file_paths}")
                    
                    # Sử dụng danh sách file từ state nếu có, nếu không thì dùng files_found
                    actual_files = state_file_paths if state_file_paths else files_found
                    
                    # Đảm bảo actual_files là list và chỉ chứa các đường dẫn duy nhất
                    unique_files = set()
                    
                    if isinstance(actual_files, str):
                        # Nếu actual_files là string, có thể là một file duy nhất hoặc mô tả
                        unique_files.add(actual_files)
                    else:
                        # Thêm tất cả các file vào set để loại bỏ trùng lặp
                        for file_path in actual_files:
                            if file_path and isinstance(file_path, str):
                                unique_files.add(file_path)
                    
                    # Số lượng file chính xác là số lượng phần tử duy nhất
                    file_count = len(unique_files)
                    
                    # Lưu file_count vào state để các phần khác của code có thể truy cập
                    state["file_count"] = file_count
                    state["unique_files"] = list(unique_files)
                    
                    log(f"ReflectionAgent debug - Actual file count: {file_count}, unique_files: {list(unique_files)}")
                    
                    # Thu thập thông tin chi tiết về từng file
                    # Sử dụng unique_files đã được tạo trước đó
                    unique_file_list = state.get("unique_files", [])
                    
                    # Xử lý từng file duy nhất
                    for file_path in unique_file_list:
                        file_name = os.path.basename(file_path)
                        file_info = {
                            "file_path": file_path,
                            "file_name": file_name,
                            "label": classification_labels.get(file_name, ""),
                            "metadata_id": ""
                        }
                        detailed_files.append(file_info)
                    
                    # Tạo tên hiển thị ngắn gọn cho danh sách file
                    if file_count == 1:
                        if isinstance(actual_files, str):
                            file_name = os.path.basename(actual_files)
                            file_names = [file_name]
                        else:
                            file_names = [os.path.basename(actual_files[0])]
                        file_found = f"1 file: {file_names[0]}"
                    elif file_count <= 3 and not isinstance(actual_files, str):
                        file_names = [os.path.basename(f) for f in actual_files]
                        file_found = f"{file_count} files: {', '.join(file_names)}"
                    elif not isinstance(actual_files, str):
                        file_names = [os.path.basename(f) for f in actual_files[:2]]
                        file_found = f"{file_count} files: {', '.join(file_names)} và {file_count - 2} file khác"
                    
                    # Thêm thông tin về số lượng file vào key_findings
                    if file_count == 1:
                        key_findings.append(f"Đã tìm thấy 1 file")
                    else:
                        key_findings.append(f"Đã tìm thấy {file_count} files")
                    
                    # Log thông tin về file
                    if isinstance(actual_files, str):
                        log(f"ReflectionAgent: Đã tìm thấy 1 file: {os.path.basename(actual_files)}")
                    else:
                        file_names = [os.path.basename(f) for f in actual_files[:3]]
                        log(f"ReflectionAgent: Đã tìm thấy {file_count} files: {', '.join(file_names)}...")
                
                # Kết quả từ Text Extraction agent
                elif "📝" in content and ("Kết quả trích xuất từ file" in content or "Kết quả trích xuất từ các file" in content):
                    # Kiểm tra xem có phải là trích xuất nhiều file không
                    is_multi_extraction = "Kết quả trích xuất từ các file" in content or "trích xuất nhiều file" in content.lower()
                    
                    # Lấy nội dung trích xuất từ text_extraction_results trong state nếu có
                    extracted_content = ""
                    if "agent_results" in state and "text_extraction" in state["agent_results"]:
                        extracted_content = state["agent_results"]["text_extraction"]
                        log(f"Found extracted content in agent_results: {len(extracted_content)} characters")
                    
                    # Nếu không có trong agent_results, thử trích xuất từ nội dung message
                    if not extracted_content:
                        # Tách nội dung sau header
                        parts = content.split(":\n\n", 1)
                        if len(parts) > 1:
                            extracted_content = parts[1].strip()
                            log(f"Extracted content from message: {len(extracted_content)} characters")
                    
                    # Nếu vẫn không có, thử cách khác
                    if not extracted_content:
                        content_lines = content.split('\n')
                        extract_lines = []
                        found_header = False
                        
                        for line in content_lines:
                            if not found_header and ("Kết quả trích xuất từ file" in line or "File:" in line):
                                found_header = True
                                continue
                            if found_header and line.strip():
                                extract_lines.append(line.strip())
                        
                        if extract_lines:
                            extracted_content = "\n".join(extract_lines)
                            log(f"Extracted content line by line: {len(extracted_content)} characters")
                    
                    # Tạo preview cho extraction_result
                    if extracted_content:
                        # Lấy tối đa 500 ký tự đầu tiên cho preview
                        preview_length = min(500, len(extracted_content))
                        extraction_result = extracted_content[:preview_length]
                        if len(extracted_content) > preview_length:
                            extraction_result += "..."
                    else:
                        # Fallback: Lấy 3 dòng đầu tiên sau header
                        content_lines = content.split('\n')
                        preview_lines = []
                        found_content = False
                        file_count = 0
                        
                        for line in content_lines:
                            if "Kết quả trích xuất từ file" in line or "File:" in line:
                                file_count += 1
                                found_content = True
                            elif found_content and line.strip() and not line.startswith("File:"):
                                preview_lines.append(line.strip())
                                if len(preview_lines) >= 3:  # Lấy 3 dòng đầu
                                    break
                        
                        if preview_lines:
                            extraction_result = " ".join(preview_lines)[:100] + "..."
                    
                    # Đếm số file đã xử lý
                    if "accessible_files" in state:
                        file_count = len(state["accessible_files"])
                    elif "file_count" in state:
                        file_count = state["file_count"]
                    
                    # Thêm kết quả vào key_findings
                    if is_multi_extraction or file_count > 1:
                        key_findings.append(f"Đã trích xuất nội dung từ {file_count} files")
                    else:
                        key_findings.append(f"Đã trích xuất nội dung từ file")
                        
                    # Lưu nội dung trích xuất vào state để sử dụng trong reflection
                    state["extracted_content_preview"] = extraction_result
                
                # Kết quả từ File Classification agent
                elif "🏷️" in content and "Kết quả phân loại file" in content:
                    # Kiểm tra xem có phải là phân loại nhiều file không
                    is_multi_classification = "nhiều file" in content.lower() or "các file" in content.lower()
                    
                    # Trích xuất số lượng file từ nội dung
                    file_count_pattern = r'(\d+)\s+files?'
                    file_count_matches = re.findall(file_count_pattern, content)
                    file_count = int(file_count_matches[0]) if file_count_matches else 1
                    
                    # Thu thập thông tin phân loại chi tiết
                    file_classifications = {}
                    lines = content.split('\n')
                    
                    # Tìm các dòng chứa thông tin phân loại
                    for line in lines:
                        # Kiểm tra các mẫu phổ biến
                        if "File:" in line or "file:" in line:
                            # Mẫu 1: File: tên_file - phân_loại
                            if "-" in line:
                                parts = line.split("-", 1)
                                file_part = parts[0]
                                class_part = parts[1].strip()
                                
                                # Trích xuất tên file
                                if "File:" in file_part or "file:" in file_part:
                                    file_name = file_part.split(":", 1)[1].strip()
                                    file_classifications[file_name] = class_part
                            
                            # Mẫu 2: File: tên_file: phân_loại
                            elif line.count(":") >= 2:
                                parts = line.split(":", 2)
                                if len(parts) >= 3:
                                    file_name = parts[1].strip()
                                    class_part = parts[2].strip()
                                    file_classifications[file_name] = class_part
                        
                        # Mẫu 3: tên_file - phân_loại
                        elif "-" in line and not line.startswith("#") and not line.startswith("-"):
                            parts = line.split("-", 1)
                            file_name = parts[0].strip()
                            class_part = parts[1].strip()
                            
                            # Kiểm tra xem có phải tên file hợp lệ không
                            if "." in file_name and not " " in file_name:
                                file_classifications[file_name] = class_part
                    
                    # Log kết quả trích xuất để debug
                    log(f"Extracted classifications: {file_classifications}")
                    
                    # Nếu không tìm thấy phân loại, thử tìm kiếm toàn bộ nội dung
                    if not file_classifications:
                        # Tìm các mẫu như "plan2023.pdf - Kế hoạch kinh doanh"
                        pattern = r'([\w.-]+\.\w+)\s*[-:]\s*([^\n\r]+)'
                        matches = re.findall(pattern, content)
                        for file_name, classification in matches:
                            file_classifications[file_name.strip()] = classification.strip()
                    
                    # Cập nhật thông tin phân loại cho các file đã tìm thấy
                    for file_info in detailed_files:
                        if file_info["file_name"] in file_classifications:
                            file_info["label"] = file_classifications[file_info["file_name"]]
                    
                    # Sử dụng số lượng file thực tế từ detailed_files hoặc state["file_count"]
                    actual_file_count = state.get("file_count", len(detailed_files))
                    
                    if is_multi_classification or actual_file_count > 1:
                        # Trích xuất các nhãn phân loại cho nhiều file
                        classifications = list(file_classifications.values())
                        
                        if classifications:
                            # Lấy các phân loại duy nhất
                            unique_classifications = list(set(classifications))
                            classification_result = f"{', '.join(unique_classifications[:3])}"
                            if len(unique_classifications) > 3:
                                classification_result += f" và {len(unique_classifications) - 3} loại khác"
                            key_findings.append(f"Đã phân loại {actual_file_count} files: {classification_result}")
                        else:
                            # Fallback nếu không tìm thấy chi tiết phân loại
                            label_pattern = r'Kết quả phân loại file[^:]*:\s*([^\n\r]+)'
                            label_matches = re.findall(label_pattern, content)
                            if label_matches:
                                classification_result = label_matches[0].strip()
                                key_findings.append(f"Đã phân loại {actual_file_count} files: {classification_result}")
                    else:
                        # Xử lý phân loại đơn file
                        label_pattern = r'Kết quả phân loại file[^:]*:\s*([^\n\r]+)'
                        label_matches = re.findall(label_pattern, content)
                        if label_matches:
                            classification_result = label_matches[0].strip()
                            key_findings.append(f"Đã phân loại file: {classification_result}")
                
                # Kết quả từ Metadata agent
                elif "📋" in content and "Đã lưu metadata thành công" in content:
                    # Kiểm tra xem có phải là lưu metadata cho nhiều file không
                    is_multi_file_metadata = "nhiều file" in content.lower() or "các file" in content.lower()
                    
                    # Trích xuất metadata ID và file paths
                    id_pattern = r'ID:\s*([a-f0-9-]+)'
                    id_matches = re.findall(id_pattern, content)
                    
                    # Trích xuất thông tin chi tiết về metadata
                    metadata_file_info = {}
                    lines = content.split('\n')
                    current_file = None
                    current_id = None
                    
                    for line in lines:
                        # Tìm ID metadata
                        if "ID:" in line:
                            id_match = re.search(r'ID:\s*([a-f0-9-]+)', line)
                            if id_match:
                                current_id = id_match.group(1)
                                if current_id not in metadata_ids:
                                    metadata_ids.append(current_id)
                        
                        # Tìm thông tin file
                        if "File:" in line or "Đường dẫn:" in line:
                            file_path_match = re.search(r'(?:File|Đường dẫn):\s*(.+)', line)
                            if file_path_match:
                                file_path = file_path_match.group(1).strip()
                                file_name = os.path.basename(file_path)
                                
                                # Cập nhật metadata ID cho file trong detailed_files
                                for file_info in detailed_files:
                                    if file_info["file_name"] == file_name or file_info["file_path"] == file_path:
                                        file_info["metadata_id"] = current_id
                    
                    # Trích xuất số lượng file từ nội dung
                    file_count_pattern = r'(\d+)\s+files?'
                    file_count_matches = re.findall(file_count_pattern, content)
                    file_count = int(file_count_matches[0]) if file_count_matches else 1
                    
                    if id_matches:
                        if is_multi_file_metadata or file_count > 1:
                            key_findings.append(f"Đã lưu metadata cho {file_count} files với ID: {', '.join(metadata_ids[:3])}")
                            if len(metadata_ids) > 3:
                                key_findings[-1] += f" và {len(metadata_ids) - 3} ID khác"
                        else:
                            key_findings.append(f"Đã lưu metadata với ID: {metadata_ids[0]}")
                            
                    # Cập nhật thông tin metadata cho các file chưa có metadata_id
                    if len(metadata_ids) == 1 and detailed_files:
                        for file_info in detailed_files:
                            if "metadata_id" not in file_info:
                                file_info["metadata_id"] = metadata_ids[0]
            
            # Tạo prompt cho reflection
            # Tạo phần mô tả chi tiết về các file đã tìm thấy
            file_info = ""
            file_count = 0  # Khởi tạo file_count với giá trị mặc định
            
            if detailed_files:
                file_count = len(detailed_files)  # Cập nhật file_count dựa trên detailed_files
                file_list = []
                for file_info_item in detailed_files[:3]:  # Giới hạn hiển thị chi tiết 3 file đầu tiên
                    file_detail = f"File: {file_info_item['file_name']}"
                    if file_info_item.get("label"):
                        file_detail += f", Phân loại: {file_info_item['label']}"
                    if file_info_item.get("metadata_id"):
                        file_detail += f", Metadata ID: {file_info_item['metadata_id']}"
                    file_list.append(file_detail)
                
                file_info = f"Đã tìm thấy {len(detailed_files)} files:\n- " + "\n- ".join(file_list)
                if len(detailed_files) > 3:
                    file_info += f"\n- và {len(detailed_files) - 3} file khác"
            elif file_found:
                file_count = 1  # Nếu có file_found, có ít nhất 1 file
                if "files:" in file_found.lower() or "file:" in file_found.lower():
                    # Nếu đã có thông tin đầy đủ về file
                    file_info = file_found
                    # Thử đếm số file từ file_found
                    import re
                    file_count_matches = re.findall(r'(\d+)\s+files?', file_found.lower())
                    if file_count_matches:
                        file_count = int(file_count_matches[0])
                else:
                    # Nếu chỉ có tên file đơn lẻ
                    file_info = f"File: {file_found}"
            
            # Đảm bảo file_count được lấy từ state nếu có
            if "file_count" in state:
                file_count = state["file_count"]
                log(f"Using file_count from state: {file_count}")
            elif "accessible_files" in state:
                file_count = len(state["accessible_files"])
                log(f"Using file_count from accessible_files: {file_count}")
            
            # Kiểm tra xem metadata agent có thực sự được sử dụng không
            metadata_agent_used = "metadata" in used_tools
            
            # Lấy nội dung trích xuất từ state nếu có
            extracted_content = ""
            if "agent_results" in state and "text_extraction" in state["agent_results"]:
                extracted_content = state["agent_results"]["text_extraction"]
                # Giới hạn độ dài nội dung trích xuất để tránh prompt quá dài
                if len(extracted_content) > 1000:
                    extracted_content = extracted_content[:1000] + "..."
                log(f"Found extracted content in agent_results: {len(extracted_content)} characters")
            
            # Tạo prompt với thông tin rõ ràng hơn và số lượng file chính xác
            reflection_prompt = f"""
            Bạn là một AI assistant chuyên về tổng hợp kết quả và trả lời người dùng một cách tự nhiên, thân thiện.
            
            YÊU CẦU BAN ĐẦU CỦA NGƯỜI DÙNG:
            "{original_query}"
            
            CÁC CÔNG CỤ ĐÃ SỬ DỤNG:
            {', '.join(used_tools) if used_tools else 'Không có'}
            
            KẾT QUẢ CHÍNH:
            {chr(10).join(f"- {finding}" for finding in key_findings) if key_findings else "- Không có kết quả nào được ghi nhận"}
            
            THÔNG TIN CHI TIẾT VỀ FILE:
            {file_info if file_info else "- Không tìm thấy file nào phù hợp"}
            
            SỐ LƯỢNG FILE CHÍNH XÁC: {file_count}
            
            # Không hiển thị nội dung trích xuất trong prompt
            # {f"NỘI DUNG TRÍCH XUẤT:\n{extracted_content}" if extracted_content and "text_extraction" in used_tools else ""}
            
            LƯU Ý ĐẶC BIỆT:
            {'Đã sử dụng metadata agent để lưu metadata. Hãy đề cập đến việc đã lưu metadata trong phản hồi của bạn.' if metadata_agent_used else 'KHÔNG đề cập đến việc lưu metadata trong phản hồi của bạn vì metadata agent không được sử dụng.'}
            {'KHÔNG đề cập đến việc trích xuất nội dung trong phản hồi của bạn.' if "text_extraction" in used_tools else 'KHÔNG đề cập đến nội dung trích xuất nếu không có.'}
            
            YÊU CẦU:
            Hãy tạo một câu trả lời ngắn gọn, tự nhiên và hữu ích dựa trên thông tin trên.
            
            HƯỚNG DẪN TRẢ LỜI:
            1. Nếu đã tìm thấy file:
               - Xác nhận đã tìm thấy file thành công
               - Liệt kê tên các file chính (nếu ít hơn 5 file) hoặc số lượng file (nếu nhiều hơn 5)
               - QUAN TRỌNG: Nếu có kết quả phân loại, LUÔN đề cập đến việc các file được phân loại là gì
               - Mô tả ngắn gọn về các file đã tìm thấy
                
            2. Nếu không tìm thấy file:
               - Thông báo không tìm thấy file phù hợp
               - Đề xuất các từ khóa tìm kiếm khác nếu có thể
            
            3. Nếu có lỗi hoặc vấn đề:
               - Giải thích ngắn gọn vấn đề
               - Đề xuất hướng khắc phục nếu có
               
            4. Nếu đã trích xuất nội dung:
                - KHÔNG đề cập đến việc trích xuất nội dung trong phản hồi
                - Tập trung vào kết quả phân loại hoặc tìm kiếm hoặc save metadata tùy vào yêu cầu của người dùng
            
            LƯU Ý QUAN TRỌNG:
            - Luôn đề cập đến các file đã tìm thấy nếu có
            - LUÔN đề cập đến kết quả phân loại file nếu có (ví dụ: "Hai file đều được phân loại là tài chính")
            - Sử dụng ngôn ngữ tự nhiên, gần gũi
            - Giới hạn trong 2-3 câu 
            - KHÔNG đề cập đến việc trích xuất nội dung trong phản hồi
            - Không cần giải thích thêm sau câu trả lời
            
            CÂU TRẢ LỜI (chỉ trả về câu trả lời, không có phần giải thích):
            """
            
            # Gọi LLM để tạo reflection response
            response = await self.model.ainvoke(reflection_prompt)
            reflection_response = response.content.strip()
            
            # Đảm bảo số lượng file được báo cáo chính xác
            # Sử dụng state["file_count"] đã được xác định trước đó
            file_count = state.get("file_count", 0)
            
            # Nếu không có file_count trong state, sử dụng số lượng file duy nhất trong detailed_files
            if file_count == 0 and detailed_files:
                # Số lượng file chính xác là số lượng phần tử trong detailed_files
                # Vì detailed_files đã được xử lý để không có trùng lặp
                file_count = len(detailed_files)
                
            # Đảm bảo file_count luôn là số nguyên dương
            file_count = max(0, file_count)
                
            log(f"ReflectionAgent debug - Final file count for response: {file_count}, detailed_files: {len(detailed_files) if detailed_files else 0}")
            
            # Kiểm tra nếu phản hồi không chính xác về số lượng file
            incorrect_file_count = False
            
            # Kiểm tra nếu phản hồi đề cập đến số lượng file khác với số lượng thực tế
            # Tìm các số trong phản hồi
            numbers_in_response = re.findall(r'\b(\d+)\s+files?\b', reflection_response.lower())
            
            # Nếu có số trong phản hồi và khác với file_count
            for num_str in numbers_in_response:
                if int(num_str) != file_count:
                    incorrect_file_count = True
                    break
                    
            # Nếu chỉ có 1 file nhưng phản hồi đề cập đến nhiều file
            if file_count == 1 and ("files" in reflection_response.lower() or re.search(r'\d+\s+files', reflection_response.lower())):
                incorrect_file_count = True
                
            # Nếu có nhiều file nhưng phản hồi không đề cập đến nhiều file
            if file_count > 1 and "files" not in reflection_response.lower():
                incorrect_file_count = True
                
            if incorrect_file_count:
                # Tạo danh sách tên file để hiển thị trong prompt
                file_names = []
                if detailed_files:
                    file_names = [f_info["file_name"] for f_info in detailed_files[:3]]
                    if len(detailed_files) > 3:
                        file_names.append(f"và {len(detailed_files) - 3} file khác")
                
                # Thử lại với prompt rõ ràng hơn
                if file_count == 1:
                    # Nếu chỉ có 1 file
                    file_name = file_names[0] if file_names else "file"
                    enhanced_prompt = f"""
                    {reflection_prompt}
                    
                    LƯU Ý ĐẶC BIỆT: 
                    Bạn đã tìm thấy CHÍNH XÁC 1 FILE, không phải nhiều file.
                    File này là: {file_name}
                    Hãy đảm bảo đề cập đến việc tìm thấy CHỈ MỘT file trong câu trả lời của bạn.
                    KHÔNG được đề cập đến nhiều file trong câu trả lời.
                    """
                else:
                    # Nếu có nhiều file
                    enhanced_prompt = f"""
                    {reflection_prompt}
                    
                    LƯU Ý ĐẶC BIỆT: 
                    Bạn đã tìm thấy CHÍNH XÁC {file_count} FILE, không phải nhiều hơn hay ít hơn.
                    Các file bao gồm: {', '.join(file_names) if file_names else file_found}
                    Hãy đảm bảo đề cập đến việc tìm thấy CHÍNH XÁC {file_count} FILE trong câu trả lời của bạn và liệt kê tên file.
                    KHÔNG được đề cập đến số lượng file khác với {file_count}.
                    """
                
                try:
                    enhanced_response = await self.model.ainvoke(enhanced_prompt)
                    enhanced_reflection = enhanced_response.content.strip()
                    
                    # Kiểm tra xem phản hồi mới có chính xác về số lượng file không
                    is_correct_response = False
                    
                    if file_count == 1:
                        # Nếu chỉ có 1 file, phản hồi không nên đề cập đến nhiều file
                        if "files" not in enhanced_reflection.lower() and "nhiều file" not in enhanced_reflection.lower():
                            is_correct_response = True
                    else:
                        # Nếu có nhiều file, phản hồi phải đề cập đến nhiều file
                        if "files" in enhanced_reflection.lower() or "nhiều file" in enhanced_reflection.lower():
                            is_correct_response = True
                    
                    if is_correct_response:
                        reflection_response = enhanced_reflection
                        log(f"Enhanced reflection response generated with correct file count mention: {file_count}")
                except Exception as e:
                    log(f"Error generating enhanced reflection: {str(e)}", level='warning')
            
            log(f"Reflection response generated: {reflection_response}")
            return reflection_response
            
        except Exception as e:
            log(f"Error in reflection agent: {str(e)}", level='error')
            # Fallback response
            return f"Tôi đã hoàn thành yêu cầu của bạn. Đã sử dụng {len(state.get('used_tools', []))} công cụ để xử lý và đạt được kết quả mong muốn."

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
        self.reflection_agent = ReflectionAgent()  # Thêm reflection agent
        
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
        graph_builder.add_node("reflection", self.run_reflection_agent)  # Thêm reflection node
        
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
        
        # Evaluator routes to reflection instead of directly to END
        graph_builder.add_conditional_edges(
            "evaluator",
            self.route_based_on_evaluation,
            {"complete": "reflection", "continue": "worker"}  # Thay đổi: complete -> reflection
        )
        
        # Reflection agent ends the workflow
        graph_builder.add_edge("reflection", END)
        
        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=memory)
        log("Multi-agent graph built successfully with reflection agent")
    
    async def run_reflection_agent(self, state: AgentState) -> AgentState:
        """
        Run the reflection agent to create final response for user.
        """
        try:
            log("Running ReflectionAgent...")
            
            # Track that we're using reflection agent
            if "used_tools" not in state:
                state["used_tools"] = []
            state["used_tools"].append("reflection")
            
            # Generate reflection response
            reflection_response = await self.reflection_agent.reflect_and_respond(state)
            
            # Add reflection response to messages
            state["messages"].append(AIMessage(content=f"💭 {reflection_response}"))
            
            # Add to chain of thought
            if "chain_of_thought" not in state:
                state["chain_of_thought"] = []
            state["chain_of_thought"].append(f"🤔 Reflection: Tạo câu trả lời cuối cùng cho người dùng")
            
            # Mark task as complete
            state["task_complete"] = True
            state["success_criteria_met"] = True
            
            log(f"ReflectionAgent completed: {reflection_response}")
            return state
            
        except Exception as e:
            log(f"Error in reflection agent: {str(e)}", level='error')
            # Fallback response
            fallback_response = "Tôi đã hoàn thành yêu cầu của bạn thành công."
            state["messages"].append(AIMessage(content=f"💭 {fallback_response}"))
            state["task_complete"] = True
            state["success_criteria_met"] = True
            return state

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
        
        # Store original query if not already stored
        if not state.get("original_query"):
            for message in state["messages"]:
                if isinstance(message, HumanMessage):
                    state["original_query"] = message.content
                    break
        
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
                # state["chain_of_thought"].append(f"Điều chỉnh thứ tự agent để đảm bảo workflow chính xác: {', '.join(new_agents)}")
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
            response_content = f"🗂️ {agent_response.content}"
            print(f"FilesystemAgent response: {response_content}")
            log(f"FilesystemAgent response: {response_content}")
            state["messages"].append(AIMessage(content=response_content))
            
            # Store result in agent_results
            if "agent_results" not in state:
                state["agent_results"] = {}
            state["agent_results"]["filesystem"] = agent_response.content
            
            # Check if filesystem agent found any results
            if "Không tìm thấy" in agent_response.content or "không biết" in agent_response.content.lower() or "không tìm thấy" in agent_response.content.lower():
                print("Filesystem agent didn't find results. Trying RAG agent...")
                
                # Call RAG agent for content-based search
                rag_agent = self.agents["rag"]
                rag_result = await rag_agent.invoke(query, self.session_id)
                
                if isinstance(rag_result, dict) and 'content' in rag_result:
                    # Add RAG response to messages
                    rag_content = f"🔍 Tìm kiếm theo nội dung file:\n\n{rag_result['content']}"
                    state["messages"].append(AIMessage(content=rag_content))
                    
                    # Store RAG result
                    state["agent_results"]["rag"] = rag_result['content']
                    
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
                
                # Nếu RAG trả về chuỗi văn bản có chứa đường dẫn file, trích xuất chúng
                # Mẫu: "Tôi đã tìm thấy các file sau:\n1. C:\path\to\file1.docx\n2. C:\path\to\file2.docx"
                if "\n" in response_content and ("Tôi đã tìm thấy" in response_content or "tìm thấy các file" in response_content):
                    lines = response_content.split("\n")
                    for line in lines:
                        # Tìm các dòng có đường dẫn file
                        if line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip().startswith("3."):
                            # Trích xuất đường dẫn file từ dòng
                            parts = line.strip().split(".", 1)
                            if len(parts) > 1:
                                file_path = parts[1].strip()
                                file_paths.append(file_path)
                    
                    log(f"Extracted {len(file_paths)} file paths from RAG response text: {file_paths}")

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
            
            # Store result in agent_results
            if "agent_results" not in state:
                state["agent_results"] = {}
            state["agent_results"]["rag"] = response_content
            
            # Lưu đường dẫn file vào state để các agent khác có thể sử dụng
            if file_paths:
                state["processed_files"] = file_paths
                
            return state
            
        except Exception as e:
            log(f"Error in RAG agent: {str(e)}", level='error')
            state["messages"].append(AIMessage(
                content=f"Có lỗi xảy ra khi tìm kiếm nội dung: {str(e)}"
            ))
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

            # Extract file paths from previous agent messages
            file_paths = []
            file_content = None
            file_classification = None
            
            # Kiểm tra xem có file paths đã được lưu trong state từ các agent trước đó không
            if "accessible_files" in state:
                file_paths = state["accessible_files"]
                log(f"Found {len(file_paths)} file paths from state: {file_paths}")
            elif "classified_files" in state:
                file_paths = state["classified_files"]
                log(f"Found {len(file_paths)} file paths from classified_files: {file_paths}")
            else:
                # First, look for file paths from RAG or filesystem agents
                for message in reversed(state["messages"]):
                    if isinstance(message, AIMessage):
                        # Check for file_paths in additional_kwargs (from RAG agent)
                        if hasattr(message, '_additional_kwargs') and 'file_paths' in message._additional_kwargs:
                            paths = message._additional_kwargs['file_paths']
                            if paths and isinstance(paths, list) and len(paths) > 0:
                                file_paths = paths
                                log(f"Found {len(file_paths)} file paths from RAG agent: {file_paths}")
                                break
                        
                        # Check for file paths in message content
                        if "Tôi đã tìm thấy file:" in message.content:
                            import re
                            # Tìm một file path
                            file_pattern = r'Tôi đã tìm thấy file:\s*([A-Z]:\\[^\s\n\r]+)'
                            file_matches = re.findall(file_pattern, message.content)
                            if file_matches:
                                file_paths = file_matches
                                log(f"Found {len(file_paths)} file paths from message content: {file_paths}")
                                break
                            
                            # Tìm nhiều file paths từ danh sách đánh số
                            numbered_pattern = r'\d+\. ([A-Z]:\\[^\s\n\r]+)'
                            numbered_matches = re.findall(numbered_pattern, message.content)
                            if numbered_matches:
                                file_paths = numbered_matches
                                log(f"Found {len(file_paths)} file paths from numbered list: {file_paths}")
                                break
            
            # Then, look for extracted content from text extraction agent
            for message in reversed(state["messages"]):
                if not isinstance(message, AIMessage):
                    continue
                    
                # Check if this is a text extraction agent message
                is_extraction_msg = ("📄" in message.content or "[Text Extraction Agent]:" in message.content or 
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
                if isinstance(message, AIMessage) and ("🏷️" in message.content or "[File Classification Agent]:" in message.content or "Kết quả phân loại file" in message.content or "Giáo dục" in message.content):
                    log("Found file classification agent message")
                    # Look for classification label in different possible formats
                    import re
                    
                    # Format 1: "PhĐiều chỉnh thứ tự agent để đảm bảo workflow chính xác:n loại: Giáo dục"
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
            
            # Xử lý nhiều file paths
            if file_paths:
                import os
                # Đảm bảo file_paths chỉ chứa các đường dẫn hợp lệ và duy nhất
                valid_file_paths = []
                for path in file_paths:
                    if path and isinstance(path, str) and os.path.exists(path) and path not in valid_file_paths:
                        valid_file_paths.append(path)
                
                # Log để debug
                log(f"Found {len(file_paths)} file paths, {len(valid_file_paths)} valid paths")
                
                # Nếu có nhiều file, tạo danh sách tên file
                if len(valid_file_paths) > 1:
                    file_names = [os.path.basename(path) for path in valid_file_paths]
                    metadata_params['file_names'] = file_names
                    metadata_params['file_paths'] = valid_file_paths
                    # Sử dụng file đầu tiên làm file chính cho metadata
                    metadata_params['file_name'] = file_names[0] + f" và {len(file_names)-1} file khác"
                    metadata_params['file_path'] = valid_file_paths[0]
                    metadata_params['is_multi_file'] = True
                    metadata_params['file_count'] = len(valid_file_paths)
                    
                    # Log để debug
                    log(f"Processing multiple files: {len(valid_file_paths)} files")
                    log(f"File names: {file_names}")
                elif len(valid_file_paths) == 1:
                    # Nếu chỉ có một file
                    metadata_params['file_name'] = os.path.basename(valid_file_paths[0])
                    metadata_params['file_path'] = valid_file_paths[0]
                    metadata_params['is_multi_file'] = False
                else:
                    # Không có file hợp lệ
                    log("Warning: No valid file paths found")
                    return state
            
            # Always pass classification labels if available
            if "classification_labels" in state:
                metadata_params['classification_labels'] = state.get("classification_labels", {})
                log(f"Passing classification_labels to metadata agent: {metadata_params['classification_labels']}")
                
            # Set classification if available
            if file_classification and file_classification.lower() not in ["không xác định", "chưa phân loại", "không có phân loại"]:
                # Clean up the classification label
                label = file_classification.split(':')[-1].strip()
                metadata_params['label'] = label
            else:
                # Check if we can extract classification from state
                if "classified_files" in state and "classification_labels" in state:
                    # Get labels from state
                    labels = state.get("classification_labels", {})
                    if labels:
                        # Use the first label as default
                        first_label = next(iter(labels.values()))
                        metadata_params['label'] = first_label
                        log(f"Using label from state: {first_label}")
            
            # Check if we have individual file extraction results in the state
            individual_contents = {}
            if "text_extraction_results" in state:
                extraction_results = state["text_extraction_results"]
                log(f"Found individual text extraction results for {len(extraction_results)} files")
                
                # Convert file paths to file names for easier lookup
                for file_path, content in extraction_results.items():
                    file_name = os.path.basename(file_path)
                    individual_contents[file_name] = content
                    log(f"Found content for {file_name}: {len(content)} characters")
            
            # Set content if available
            if 'is_multi_file' in metadata_params and metadata_params['is_multi_file']:
                # For multi-file case, we'll still use the combined content for the preview
                if file_content:
                    # Ensure content is properly formatted and not too long
                    content = file_content.strip()
                    if len(content) > 4000:  # Truncate if too long for the model
                        content = content[:4000] + "... [nội dung bị cắt bớt]"
                    metadata_params['content'] = content
                    log(f"Content length for metadata preview: {len(content)} characters")
                    
                    # Also store individual contents if we have them
                    if individual_contents:
                        metadata_params['individual_contents'] = individual_contents
                        log(f"Added individual contents for {len(individual_contents)} files")
                else:
                    # Provide a placeholder when no content is available
                    log("Warning: No content found for metadata. Using placeholder.")
                    file_names = metadata_params.get('file_names', [])
                    placeholder = f"Multiple files: {', '.join(file_names[:3])}{'...' if len(file_names) > 3 else ''} (no content extracted)"
                    metadata_params['content'] = placeholder
                    log(f"Using placeholder content: {placeholder}")
            else:
                # Single file case
                file_name = metadata_params.get('file_name', 'unknown_file')
                
                # Try to get content from individual extraction results first
                if file_name in individual_contents:
                    content = individual_contents[file_name]
                    log(f"Using individual content for {file_name}: {len(content)} characters")
                    metadata_params['content'] = content
                elif file_content:
                    # Fall back to general content if available
                    content = file_content.strip()
                    if len(content) > 4000:  # Truncate if too long for the model
                        content = content[:4000] + "... [nội dung bị cắt bớt]"
                    metadata_params['content'] = content
                    log(f"Using general content for {file_name}: {len(content)} characters")
                else:
                    # No content available, use placeholder
                    placeholder = f"File: {file_name} (no content extracted)"
                    metadata_params['content'] = placeholder
                    log(f"Using placeholder content: {placeholder}")
            
            # Build the enhanced query for the metadata agent
            enhanced_query = "Tôi cần tạo và lưu metadata với các thông tin sau:\n\n"
            
            # Add file information if available
            if 'is_multi_file' in metadata_params and metadata_params['is_multi_file']:
                # Thông tin cho nhiều file
                enhanced_query += f"- NHÓM FILE: {metadata_params['file_count']} files\n"
                enhanced_query += f"- TÊN FILE CHÍNH: {metadata_params['file_name']}\n"
                
                # Thêm danh sách tất cả các file
                file_list = "\n".join([f"  + {i+1}. {name}" for i, name in enumerate(metadata_params['file_names'])])
                enhanced_query += f"- DANH SÁCH FILES:\n{file_list}\n"
                
                # Thêm đường dẫn file chính
                enhanced_query += f"- ĐƯỜNG DẪN CHÍNH: {metadata_params['file_path']}\n"
            else:
                # Thông tin cho một file
                if 'file_name' in metadata_params:
                    enhanced_query += f"- TÊN FILE: {metadata_params['file_name']}\n"
                if 'file_path' in metadata_params:
                    enhanced_query += f"- ĐƯỜNG DẪN: {metadata_params['file_path']}\n"
            
            # Thêm phân loại nếu có
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
                
                # Chuẩn bị metadata cho agent
                metadata_for_agent = {
                    'file_name': metadata_params.get('file_name'),
                    'file_path': metadata_params.get('file_path'),
                    'label': metadata_params.get('label'),
                    'content': metadata_params.get('content', '')
                }
                
                # Thêm thông tin về nhiều file nếu có
                if 'is_multi_file' in metadata_params and metadata_params['is_multi_file']:
                    metadata_for_agent['is_multi_file'] = True
                    metadata_for_agent['file_count'] = metadata_params.get('file_count')
                    metadata_for_agent['file_names'] = metadata_params.get('file_names', [])
                    metadata_for_agent['file_paths'] = metadata_params.get('file_paths', [])
                
                response = metadata_agent.invoke(
                    query=enhanced_query,
                    sessionId=self.session_id,
                    metadata=metadata_for_agent
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
                        r'id[\s:]*([a-f0-9]{8,})',            # id: xxxxxxxx-xxxx-...
                        r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'  # UUID format
                    ]
                    
                    for pattern in id_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            metadata_id = match.group(1)
                            log(f"Found metadata ID using pattern {pattern}: {metadata_id}")
                            break
                    
                    if metadata_id:
                        # Tạo phản hồi dựa trên số lượng file
                        if 'is_multi_file' in metadata_params and metadata_params['is_multi_file']:
                            file_count = metadata_params.get('file_count', 0)
                            file_names = metadata_params.get('file_names', [])
                            
                            # Tạo danh sách tên file ngắn gọn
                            if len(file_names) > 3:
                                file_list = f"{', '.join(file_names[:2])} và {len(file_names)-2} file khác"
                            else:
                                file_list = ", ".join(file_names)
                                
                            response_content = f"✅ Đã lưu metadata cho {file_count} files ({file_list}). ID: {metadata_id}"
                        else:
                            file_name = metadata_params.get('file_name', 'không xác định')
                            response_content = f"✅ Đã lưu metadata cho file {file_name} thành công. ID: {metadata_id}"
                            
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
                        
                        # Tạo phản hồi dựa trên số lượng file khi có lỗi
                        if 'is_multi_file' in metadata_params and metadata_params['is_multi_file']:
                            file_count = metadata_params.get('file_count', 0)
                            file_names = metadata_params.get('file_names', [])
                            
                            # Tạo danh sách tên file ngắn gọn
                            if len(file_names) > 3:
                                file_list = f"{', '.join(file_names[:2])} và {len(file_names)-2} file khác"
                            else:
                                file_list = ", ".join(file_names)
                                
                            if any(keyword in response.lower() for keyword in error_keywords):
                                response_content = f"❌ Lỗi khi xử lý metadata cho {file_count} files ({file_list}): {response}"
                            else:
                                response_content = f"ℹ️ Đã xử lý metadata cho {file_count} files ({file_list}), nhưng không tìm thấy ID: {response}"
                        else:
                            file_name = metadata_params.get('file_name', 'không xác định')
                            if any(keyword in response.lower() for keyword in error_keywords):
                                response_content = f"❌ Lỗi khi xử lý metadata cho file {file_name}: {response}"
                            else:
                                response_content = f"ℹ️ Đã xử lý metadata cho file {file_name}, nhưng không tìm thấy ID: {response}"
                        
                        log(f"Metadata agent response (no ID found): {response}")
                        
                # Add the response to the conversation
                formatted_response = f"📋 {response_content}"
                log(f"MetadataAgent response: {formatted_response[:200]}...")
                state["messages"].append(AIMessage(content=formatted_response))
                
                # Store result in agent_results even if no metadata ID was found
                if "agent_results" not in state:
                    state["agent_results"] = {}
                    
                state["agent_results"]["metadata"] = {
                    "metadata_id": metadata_id,
                    "response": response_content,
                    "is_multi_file": metadata_params.get('is_multi_file', False),
                    "file_count": metadata_params.get('file_count', 1) if metadata_params.get('file_name') else 0,
                    "success": metadata_id is not None
                }
                
                # Store metadata info in the state for future reference
                if 'metadata' not in state:
                    state['metadata'] = {}
                
                if 'metadata_ids' not in state['metadata']:
                    state['metadata']['metadata_ids'] = []
                    
                if metadata_id:  # Use the metadata_id variable that we defined earlier
                    state['metadata']['metadata_ids'].append(metadata_id)
                    log(f"Added metadata ID to state: {metadata_id}")
                    
                    # Lưu thông tin về các file đã xử lý metadata
                    if 'is_multi_file' in metadata_params and metadata_params['is_multi_file']:
                        # Lưu danh sách các file đã xử lý
                        if 'processed_files' not in state['metadata']:
                            state['metadata']['processed_files'] = []
                            
                        # Thêm các file vào danh sách đã xử lý
                        file_paths = metadata_params.get('file_paths', [])
                        state['metadata']['processed_files'].extend(file_paths)
                        log(f"Added {len(file_paths)} files to processed_files in state")
                        
                        # Lưu thông tin về nhóm file
                        if 'file_groups' not in state['metadata']:
                            state['metadata']['file_groups'] = {}
                            
                        # Tạo nhóm file với metadata_id làm key
                        state['metadata']['file_groups'][metadata_id] = {
                            'file_count': metadata_params.get('file_count', 0),
                            'file_paths': file_paths,
                            'file_names': metadata_params.get('file_names', []),
                            'label': metadata_params.get('label', 'không xác định')
                        }
                    else:
                        # Lưu thông tin cho một file
                        if 'processed_files' not in state['metadata']:
                            state['metadata']['processed_files'] = []
                            
                        file_path = metadata_params.get('file_path')
                        if file_path:
                            state['metadata']['processed_files'].append(file_path)
                            log(f"Added file {file_path} to processed_files in state")
                else:
                    log("No metadata ID to add to state", level='warning')
                
                # Lưu thông tin về agent_results
                if "agent_results" not in state:
                    state["agent_results"] = {}
                    
                state["agent_results"]["metadata"] = {
                    "metadata_id": metadata_id,
                    "response": response_content,
                    "is_multi_file": metadata_params.get('is_multi_file', False),
                    "file_count": metadata_params.get('file_count', 1) if metadata_params.get('file_name') else 0
                }
                
                return state
                
            except Exception as e:
                error_msg = f"Lỗi khi chạy MetadataAgent: {str(e)}"
                log(error_msg, level='error')
                state["messages"].append(AIMessage(
                    content=f"[Lỗi] {error_msg}. Vui lòng thử lại hoặc kiểm tra kết nối MCP server."
                ))
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
        def clean_invalid_unicode(text):
            """Xử lý và loại bỏ các ký tự Unicode không hợp lệ"""
            if not text:
                return ""
            # Thay thế các ký tự surrogate đơn lẻ và ký tự không hợp lệ khác
            return text.encode('utf-8', errors='ignore').decode('utf-8')
            
        try:
            # Tìm file paths từ state trước tiên nếu có
            file_paths = []
            if "processed_files" in state and state["processed_files"]:
                file_paths = state["processed_files"]
                log(f"Using {len(file_paths)} file paths from state[processed_files]: {file_paths}")
            
            # Nếu không có trong state, tìm từ các tin nhắn trước đó
            if not file_paths:
                for message in reversed(state["messages"]):
                    if isinstance(message, AIMessage):
                        log(f"Checking message for file paths: {message.content[:100]}...")
                        
                        # Kiểm tra xem tin nhắn có phải là từ RAG agent không (có thể có trường file_paths)
                        if hasattr(message, '_additional_kwargs') and 'file_paths' in message._additional_kwargs:
                            log(f"Found RAG agent message with file_paths field")
                            paths = message._additional_kwargs['file_paths']
                            if paths and isinstance(paths, list) and len(paths) > 0:
                                file_paths.extend(paths)
                                log(f"Extracted {len(paths)} file paths from RAG agent")
                                break
                    
                    # Tìm kiếm câu "Tôi đã tìm thấy file:" hoặc "Tôi đã tìm thấy {n} files:" trong tin nhắn
                    if "Tôi đã tìm thấy file:" in message.content:
                        log(f"Found agent message with standard format for single file")
                        
                        # Tìm đường dẫn file sau câu "Tôi đã tìm thấy file:"
                        import re
                        
                        # Tìm sau "Tôi đã tìm thấy file:" - kiểm tra cả đường dẫn đầy đủ
                        full_path_pattern = r'Tôi đã tìm thấy file:\s*([A-Z]:\\[^\s\n\r]+)'
                        full_path_matches = re.findall(full_path_pattern, message.content)
                        
                        if full_path_matches:
                            # Lấy đường dẫn và loại bỏ các ký tự không mong muốn ở cuối
                            raw_path = full_path_matches[0]
                            # Loại bỏ các ký tự không mong muốn có thể có ở cuối đường dẫn
                            if raw_path.endswith("'}"):
                                file_paths.append(raw_path[:-2])
                            else:
                                file_paths.append(raw_path)
                            log(f"Extracted full file path: {file_paths[-1]}")
                            break
                    
                    # Tìm nhiều file từ định dạng "Tôi đã tìm thấy {n} files:"
                    elif "files:" in message.content and "Tôi đã tìm thấy" in message.content:
                        log(f"Found agent message with multiple files format")
                        import re
                        
                        # Tìm đường dẫn file từ danh sách đánh số
                        multi_file_pattern = r'\d+\.\s*([A-Z]:\\[^\s\n\r]+)'
                        multi_file_matches = re.findall(multi_file_pattern, message.content)
                        
                        if multi_file_matches:
                            for raw_path in multi_file_matches:
                                if raw_path.endswith("'}"):
                                    file_paths.append(raw_path[:-2])
                                else:
                                    file_paths.append(raw_path)
                            log(f"Extracted {len(multi_file_matches)} file paths from numbered list")
                            break
                    
                    # Dự phòng: Nếu không tìm thấy câu chuẩn, thử tìm bất kỳ đường dẫn Windows nào
                    elif any(indicator in message.content for indicator in ["Đã tìm thấy file:", "tìm thấy file", "[Filesystem Agent]:", "[RAG Agent]:", "🗂️", "🔍"]):
                        log(f"Found agent message with non-standard format")
                        
                        # Tìm bất kỳ đường dẫn nào trong tin nhắn
                        import re
                        file_pattern = r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.[a-zA-Z0-9]+'
                        file_matches = re.findall(file_pattern, message.content)
                        
                        if file_matches:
                            file_paths.extend(file_matches)
                            log(f"Extracted {len(file_matches)} file paths using general pattern")
                            break

            # Get the last message that's not from an agent
            query = ""
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    query = message.content
                    break

            # Kiểm tra quyền truy cập file nếu tìm thấy file paths
            if file_paths:
                # Import AccessControlManager
                import sys
                import os
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                from utils.access_control import AccessControlManager
                
                # Lấy vai trò người dùng từ state hoặc session
                user_role = state.get("user_role", "user")  # Mặc định là "user" nếu không có
                
                # Khởi tạo AccessControlManager
                access_control_file = os.path.join(current_dir, "config", "access_control.json")
                access_manager = AccessControlManager(access_control_file)
                
                # Kiểm tra quyền truy cập cho từng file
                accessible_files = []
                for file_path in file_paths:
                    has_access, reason = access_manager.check_file_access(file_path, user_role)
                    access_manager.log_access_attempt(file_path, user_role, has_access, reason)
                    
                    if has_access:
                        log(f"Access granted for user role '{user_role}' to file '{file_path}'")
                        accessible_files.append(file_path)
                    else:
                        log(f"Access denied for file '{file_path}': {reason}", level='warning')
                
                if not accessible_files:
                    # Không có quyền truy cập vào bất kỳ file nào, thông báo cho người dùng
                    error_message = f"⚠️ Không thể trích xuất nội dung: Bạn không có quyền truy cập vào các file này"
                    state["messages"].append(AIMessage(content=error_message))
                    log(f"Access denied to all files", level='warning')
                    return state
                
                # Có quyền truy cập ít nhất một file, tiếp tục với trích xuất
                if len(accessible_files) == 1:
                    enhanced_query = f"Extract text from the file at path {accessible_files[0]}"
                else:
                    # Nếu có nhiều file, tạo danh sách đường dẫn
                    file_paths_str = "\n".join([f"- {path}" for path in accessible_files])
                    enhanced_query = f"Extract text from the following files:\n{file_paths_str}"
                
                log(f"Enhanced query with file paths: {enhanced_query[:100]}...")
                
                # Lưu danh sách file có quyền truy cập vào state để sử dụng sau này
                state["accessible_files"] = accessible_files
            else:
                # Không tìm thấy file path, sử dụng query gốc
                enhanced_query = query
                log(f"WARNING: No file paths found! Running TextExtractionAgent with original query: {query}")

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
                content = clean_invalid_unicode(response['content'])
                log(f"Extracted content from response dict with key 'content': {content[:100]}...")
            
            # Trường hợp 2: Response là string
            elif isinstance(response, str):
                content = clean_invalid_unicode(response)
                log(f"Response is already a string: {content[:100]}...")
            
            # Trường hợp 3: Response là dict nhưng không có 'content', thử tìm các key khác
            elif isinstance(response, dict):
                log(f"Response keys: {list(response.keys())}")
                
                # Thử lấy từ key 'response_type' trước
                if 'response_type' in response and 'content' in response:
                    content = clean_invalid_unicode(response['content'])
                    log(f"Using content from standard response format: {content[:100]}...")
                
                # Thử lấy giá trị từ các key khác nếu là string dài
                else:
                    for key in response.keys():
                        if isinstance(response[key], str) and len(response[key]) > 20:
                            content = clean_invalid_unicode(response[key])
                            log(f"Using content from key '{key}': {content[:100]}...")
                            break
            
            # Trường hợp 4: Các trường hợp khác, chuyển về string
            else:
                content = clean_invalid_unicode(str(response))
                log(f"Converted response to string: {content[:100]}...")
                
            # Kiểm tra nếu content chứa kết quả trích xuất
            if "I'll extract the text from" in content or "Here's the extracted text" in content:
                # Tìm phần nội dung trích xuất sau các câu mở đầu
                import re
                extracted_text = re.split(r"Here's the extracted text[:\s]*|I'll extract the text[^:]*:\s*", content, 1)
                if len(extracted_text) > 1:
                    content = extracted_text[1].strip()
                    log(f"Extracted the actual content after introduction: {content[:100]}...")
            
            # Kiểm tra nếu content chứa "I'll use the extract_text_from" (câu trả lời của agent) hoặc nếu nội dung trích xuất trùng với query
            if ("I'll use the extract_text_from" in content or content.strip() == query.strip()) and "accessible_files" in state:
                log("Agent response contains tool usage description but no actual extraction result or returned the query")
                # Thử trực tiếp các hàm trích xuất dựa vào định dạng file
                from agents.text_extraction_agent import extract_text_from_pdf, extract_text_from_word, extract_text_from_powerpoint
                
                # Trích xuất nội dung từ từng file có quyền truy cập
                extracted_contents = []
                extraction_results = {}
                for file_path in state["accessible_files"]:
                    try:
                        if file_path.lower().endswith('.pdf'):
                            extracted_text = extract_text_from_pdf(file_path)
                            extraction_results[file_path] = clean_invalid_unicode(extracted_text)
                            log(f"Directly extracted text from PDF: {file_path}")
                        elif file_path.lower().endswith('.docx'):
                            extracted_text = extract_text_from_word(file_path)
                            extraction_results[file_path] = clean_invalid_unicode(extracted_text)
                            log(f"Directly extracted text from Word document: {file_path}")
                        elif file_path.lower().endswith(('.ppt', '.pptx')):
                            extracted_text = extract_text_from_powerpoint(file_path)
                            extraction_results[file_path] = clean_invalid_unicode(extracted_text)
                            log(f"Directly extracted text from PowerPoint: {file_path}")
                    except Exception as e:
                        log(f"Error in direct extraction for {file_path}: {e}", level='error')
                        extraction_results[file_path] = f"Lỗi khi trích xuất: {str(e)}"
                
                # Nếu có kết quả trích xuất, gộp lại
                if extraction_results:
                    content = ""
                    for file_path, extracted_text in extraction_results.items():
                        content += f"\n\n--- Từ file {os.path.basename(file_path)} ---\n{clean_invalid_unicode(extracted_text)}\n"
                    content = content.strip()
                    
                    # Store individual file extraction results in the state
                    state["text_extraction_results"] = extraction_results
                    log(f"Stored individual text extraction results for {len(extraction_results)} files in state")

            if not content.strip():
                if "accessible_files" in state and state["accessible_files"]:
                    if len(state["accessible_files"]) == 1:
                        content = f"Tôi đã cố gắng trích xuất nội dung từ file {state['accessible_files'][0]} nhưng không có kết quả đáng chú ý."
                    else:
                        content = f"Tôi đã cố gắng trích xuất nội dung từ {len(state['accessible_files'])} files nhưng không có kết quả đáng chú ý."
                else:
                    content = "Tôi đã cố gắng trích xuất nội dung nhưng không có kết quả đáng chú ý."

            # Add the agent's response to the state with clear indication of extraction results
            if "accessible_files" in state and state["accessible_files"]:
                if len(state["accessible_files"]) == 1:
                    response_content = f"📝 Kết quả trích xuất từ file {state['accessible_files'][0]}:\n\n{content}"
                else:
                    file_list = "\n".join([f"- {os.path.basename(f)}" for f in state["accessible_files"]])
                    response_content = f"📝 Kết quả trích xuất từ {len(state['accessible_files'])} files:\n{file_list}\n\n{content}"
            else:
                response_content = f"📝 {content}"
                
            log(f"TextExtractionAgent response: {response_content[:100]}...")
            state["messages"].append(AIMessage(content=response_content))
            
            # Store result in agent_results
            if "agent_results" not in state:
                state["agent_results"] = {}
            state["agent_results"]["text_extraction"] = content
            
            # Đảm bảo file_count được lưu vào state
            if "accessible_files" in state:
                state["file_count"] = len(state["accessible_files"])
                log(f"Set file_count in state to {state['file_count']}")
                
                # Lưu danh sách file đã xử lý vào state nếu chưa có
                if "processed_files" not in state:
                    state["processed_files"] = state["accessible_files"]
                    log(f"Set processed_files in state with {len(state['processed_files'])} files")
            
            # If we have accessible_files but no text_extraction_results yet, create it
            if "accessible_files" in state and "text_extraction_results" not in state:
                # Create a mapping of file paths to extracted content
                # For now, we'll assign the same content to all files if we can't distinguish
                # This is better than having no content at all
                extraction_results = {}
                
                # Check if content contains file-specific sections
                import re
                file_sections = re.split(r"\n\n---\s+Từ file\s+([^\n]+)\s+---\n", content)
                
                if len(file_sections) > 1:
                    # Content is divided by file sections
                    # First element is any text before the first file section
                    # Then alternating file names and content
                    for i in range(1, len(file_sections), 2):
                        if i+1 < len(file_sections):
                            file_name = file_sections[i]
                            file_content = file_sections[i+1]
                            
                            # Find the matching file path
                            for file_path in state["accessible_files"]:
                                if os.path.basename(file_path) == file_name:
                                    extraction_results[file_path] = file_content
                                    log(f"Mapped content section to file: {file_name}")
                                    break
                else:
                    # Content is not divided, assign to all files
                    for file_path in state["accessible_files"]:
                        extraction_results[file_path] = content
                        log(f"Assigned full content to file: {os.path.basename(file_path)}")
                
                state["text_extraction_results"] = extraction_results
                log(f"Created text_extraction_results for {len(extraction_results)} files from general content")
            
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
            
    async def run_file_classification_agent(self, state: AgentState):
        """
        Run the file classification agent on the current query.
        """
        try:
            # Tìm file paths từ state trước tiên nếu có
            file_paths = []
            if "processed_files" in state and state["processed_files"]:
                file_paths = state["processed_files"]
                log(f"Using {len(file_paths)} file paths from state[processed_files]: {file_paths}")
            
            # Tìm nội dung cần phân loại từ TextExtractionAgent
            content_to_classify = None
            
            # Nếu không có file paths trong state, tìm từ các tin nhắn
            if not file_paths:
                # Tìm kết quả từ TextExtractionAgent
                for message in reversed(state["messages"]):
                    if isinstance(message, AIMessage) and ("📝" in message.content or "[Text Extraction Agent]:" in message.content):
                        # Trích xuất nội dung sau phần giới thiệu
                        text_parts = message.content.split(":\n\n", 1)
                        if len(text_parts) > 1:
                            content_to_classify = text_parts[1].strip()
                            log(f"Found content to classify from TextExtractionAgent: {content_to_classify[:100]}...")
                            
                            # Kiểm tra nếu là nhiều file
                            
                            # Tìm kiếm chuỗi "Kết quả trích xuất từ X files:"
                            multi_file_pattern = r'Kết quả trích xuất từ (\d+) files:'
                            multi_file_match = re.search(multi_file_pattern, text_parts[0])
                            
                            if multi_file_match:
                                # Đây là kết quả từ nhiều file
                                file_list_pattern = r'- ([^\n]+)'
                                file_names = re.findall(file_list_pattern, text_parts[0])
                                log(f"Found file names in extraction: {file_names}")
                                
                                # Nếu có accessible_files trong state, lấy đường dẫn đầy đủ
                                if "accessible_files" in state and state["accessible_files"]:
                                    # Lọc các file paths dựa trên tên file đã tìm thấy
                                    for file_path in state["accessible_files"]:
                                        file_name = os.path.basename(file_path)
                                        # Kiểm tra xem file_name có trong danh sách file_names không
                                        if any(name.strip() == file_name for name in file_names):
                                            file_paths.append(file_path)
                                    log(f"Found {len(file_paths)} matching file paths from accessible_files")
                            else:
                                # Tìm file path đơn
                                file_pattern = r'từ file ([A-Z]:\\[^\s\n\r]+)'
                                file_matches = re.findall(file_pattern, text_parts[0])
                                if file_matches:
                                    file_paths.append(file_matches[0])
                                    log(f"Found file path: {file_paths[0]}")
                            break
            
            # Nếu không tìm thấy nội dung từ TextExtractionAgent hoặc không có file paths, tìm từ các nguồn khác
            if not content_to_classify or not file_paths:
                # Kiểm tra nếu có accessible_files trong state
                if "accessible_files" in state and state["accessible_files"]:
                    file_paths = state["accessible_files"]
                    log(f"Using accessible_files from state: {len(file_paths)} files")
                else:
                    # Tìm file path từ các tin nhắn
                    for message in reversed(state["messages"]):
                        if isinstance(message, AIMessage):
                            # Kiểm tra nếu là tin nhắn từ RAG agent với file_paths
                            if hasattr(message, '_additional_kwargs') and 'file_paths' in message._additional_kwargs:
                                paths = message._additional_kwargs['file_paths']
                                if paths and isinstance(paths, list):
                                    file_paths.extend(paths)
                                    log(f"Found {len(paths)} file paths from RAG agent")
                                    break
                            
                            # Tìm kiếm câu "Tôi đã tìm thấy file:" trong tin nhắn
                            elif "Tôi đã tìm thấy file:" in message.content:
                                # Trích xuất đường dẫn file từ tin nhắn
                                file_pattern = r'Tôi đã tìm thấy file:\s*([A-Z]:\\[^\s\n\r]+)'
                                file_matches = re.findall(file_pattern, message.content)
                                if file_matches:
                                    raw_path = file_matches[0]
                                    # Loại bỏ các ký tự không mong muốn có thể có ở cuối đường dẫn
                                    if raw_path.endswith("'}"):
                                        file_paths.append(raw_path[:-2])
                                    else:
                                        file_paths.append(raw_path)
                                    log(f"Found file path from FilesystemAgent: {file_paths[-1]}")
                                    break
                            
                            # Tìm nhiều file từ định dạng "Tôi đã tìm thấy {n} files:"
                            elif "files:" in message.content and "Tôi đã tìm thấy" in message.content:
                                multi_file_pattern = r'\d+\.\s*([A-Z]:\\[^\s\n\r]+)'
                                multi_file_matches = re.findall(multi_file_pattern, message.content)
                                
                                if multi_file_matches:
                                    for raw_path in multi_file_matches:
                                        if raw_path.endswith("'}"):
                                            file_paths.append(raw_path[:-2])
                                        else:
                                            file_paths.append(raw_path)
                                    log(f"Found {len(multi_file_matches)} file paths from numbered list")
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
            elif file_paths:
                # Nếu có đường dẫn file
                if len(file_paths) == 1:
                    # Nếu chỉ có một file, yêu cầu agent phân loại dựa trên file path
                    file_name = os.path.basename(file_paths[0])
                    classification_query = f"Hãy phân loại file: {file_name} (path: {file_paths[0]})"
                    log(f"Using file path for classification: {file_paths[0]}")
                else:
                    # Nếu có nhiều file, tạo danh sách đường dẫn và tên file
                    file_items = []
                    for path in file_paths:
                        file_name = os.path.basename(path)
                        file_items.append(f"- {file_name} (path: {path})")
                    
                    file_paths_str = "\n".join(file_items)
                    classification_query = f"Hãy phân loại từng file sau và trả về kết quả theo định dạng 'tên_file - phân_loại':\n{file_paths_str}"
                    log(f"Using multiple file paths for classification: {len(file_paths)} files")
            else:
                # Không có cả nội dung và đường dẫn, sử dụng query gốc
                classification_query = query
                log(f"No content or file paths found. Using original query: {query}")
            
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
                
            # Trích xuất kết quả phân loại thực sự từ phản hồi
            # Tìm kiếm mẫu "Kết quả phân loại file" hoặc các mẫu tương tự
            if classification_result:
                import re
                # Mẫu 1: Kết quả phân loại file: XYZ
                pattern1 = r'Kết quả phân loại file[^:]*:\s*([^\n\r]+)'
                # Mẫu 2: XYZ (nếu phản hồi chỉ là kết quả phân loại)
                pattern2 = r'^([^\n\r:]+)$'
                
                match = re.search(pattern1, classification_result)
                if match:
                    classification_result = match.group(1).strip()
                    log(f"Extracted classification using pattern1: {classification_result}")
                else:
                    match = re.search(pattern2, classification_result)
                    if match and len(classification_result.split()) <= 5:  # Nếu ngắn gọn (≤ 5 từ)
                        classification_result = match.group(1).strip()
                        log(f"Extracted classification using pattern2: {classification_result}")
                    else:
                        log(f"Could not extract classification pattern from result, using as-is")
                        
                # Loại bỏ các phần thừa như "Phân loại:", "Kết quả:", v.v.
                prefixes_to_remove = ["phân loại:", "kết quả:", "nhãn:"]
                lower_result = classification_result.lower()
                for prefix in prefixes_to_remove:
                    if lower_result.startswith(prefix):
                        classification_result = classification_result[len(prefix):].strip()
                        log(f"Removed prefix '{prefix}' from result: {classification_result}")
                        break
            
            # Trường hợp 3: Các trường hợp khác, chuyển về string
            else:
                classification_result = str(response)
                log(f"Converted response to string: {classification_result}")

            # Kiểm tra kết quả phân loại và xử lý trùng lặp
            if not classification_result.strip():
                classification_result = "Không thể phân loại"
            else:
                # Xử lý kết quả phân loại
                lines = classification_result.strip().split('\n')
                
                # Kiểm tra số lượng file và số lượng dòng phân loại
                if len(file_paths) > 1 and len(lines) > 1:
                    # Nếu có nhiều file và nhiều dòng phân loại, giữ nguyên kết quả
                    # vì mỗi dòng có thể là phân loại cho một file
                    log(f"Keeping multiple classification results for {len(file_paths)} files")
                elif len(lines) > 1 and len(set(lines)) == 1:
                    # Nếu tất cả các dòng giống nhau, chỉ giữ lại một dòng
                    classification_result = lines[0]
                    log(f"Removed duplicate classification results, using: {classification_result}")

            # Add the agent's response to the state with clear indication of classification results
            if file_paths:
                if len(file_paths) == 1:
                    response_content = f"🏷️ Kết quả phân loại file {file_paths[0]}: {classification_result}"
                else:
                    # Tạo danh sách tên file
                    file_names = [os.path.basename(path) for path in file_paths]
                    file_list = ", ".join(file_names[:3])
                    if len(file_names) > 3:
                        file_list += f" và {len(file_names) - 3} file khác"
                    
                    # Kiểm tra nếu có nhiều kết quả phân loại cho nhiều file
                    lines = classification_result.strip().split('\n')
                    if len(lines) > 1 and len(lines) == len(file_paths):
                        # Tạo danh sách phân loại theo từng file
                        classifications = []
                        for i, (file_name, classification) in enumerate(zip(file_names, lines)):
                            classifications.append(f"{file_name}: {classification}")
                        
                        formatted_classifications = "\n- ".join(classifications)
                        response_content = f"🏷️ Kết quả phân loại {len(file_paths)} files:\n- {formatted_classifications}"
                    else:
                        # Nếu số lượng phân loại không khớp với số file, sử dụng định dạng cũ
                        response_content = f"🏷️ Kết quả phân loại {len(file_paths)} files ({file_list}): {classification_result}"
            else:
                response_content = f"🏷️ Kết quả phân loại: {classification_result}"
                
            log(f"FileClassificationAgent response: {response_content}")
            
            # Thêm kết quả phân loại vào state
            state["messages"].append(AIMessage(content=response_content))
            
            # Store result in agent_results
            if "agent_results" not in state:
                state["agent_results"] = {}
            state["agent_results"]["file_classification"] = classification_result
            
            # Lưu thông tin file đã phân loại vào state
            if file_paths:
                state["classified_files"] = file_paths
                
                # Extract and store classification labels
                classification_labels = {}
                
                # Parse classification results
                if len(file_paths) == 1:
                    # Single file case
                    classification_labels[os.path.basename(file_paths[0])] = classification_result.strip()
                else:
                    # Multi-file case
                    lines = classification_result.strip().split('\n')
                    
                    # Check if we have one classification per file
                    if len(lines) == len(file_paths):
                        for i, path in enumerate(file_paths):
                            classification_labels[os.path.basename(path)] = lines[i].strip()
                    else:
                        # If we don't have one classification per file, use the same classification for all files
                        for path in file_paths:
                            classification_labels[os.path.basename(path)] = classification_result.strip()
                
                # Store labels in state
                state["classification_labels"] = classification_labels
                log(f"Stored classification labels in state: {classification_labels}")
            
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
            - Nếu yêu cầu chỉ liên quan đến tìm kiếm file thì chỉ sử dụng filesystem agent hoặc rag agent không sử dụng thêm các agent khác
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
        
    async def run(self, query: str, session_id: str = None, user_role: str = "user") -> Dict[str, Any]:
        """
        Run the multi-agent system with the given query.
        
        Args:
            query: Câu truy vấn của người dùng
            session_id: ID phiên làm việc, được tạo tự động nếu không cung cấp
            user_role: Vai trò của người dùng, mặc định là "user"
        
        Returns:
            Dict chứa kết quả và trạng thái của hệ thống
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
                "feedback_on_work": None,
                "success_criteria_met": False,
                "completed": False,
                "used_tools": [],
                "chain_of_thought": ["🔍1. Bắt đầu xử lý yêu cầu: " + query],
                "agent_results": {},
                "original_query": query,
                "user_role": user_role  # Thêm vai trò người dùng vào state
            }
            
            # Plan which agents to use
            state = await self.plan_agents(state)
            log(f"Kế hoạch agent ban đầu: {state['current_agents']}")
            
            # Validate and fix agent sequence if needed
            state = await self._validate_agent_sequence(state)
            log(f"Kế hoạch agent sau khi kiểm tra: {state['current_agents']}")
            state["chain_of_thought"].append(f"🧠2. Lập kế hoạch sử dụng các agent: {', '.join(state['current_agents'])}")
            
            # Run the agents in the planned order
            step_count = 3
            agent_execution_order = []
            
            while not state.get("completed", False) and state["current_agents"]:
                agent_name = state["current_agents"].pop(0)
                agent_execution_order.append(agent_name)
                log(f"Running agent: {agent_name} (Thứ tự thực thi: {agent_execution_order})")
                state["chain_of_thought"].append(f"⚡{step_count}. Đang chạy agent: {agent_name}")
                
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
                    state["chain_of_thought"].append(f"✨{step_count}a. Kết quả từ {agent_name}: {summary}")
                
                step_count += 1
            
            # Run reflection agent to create final response
            log("Running reflection agent for final response...")
            state["chain_of_thought"].append(f"🤔{step_count}. Đang tạo câu trả lời cuối cùng...")
            state = await self.run_reflection_agent(state)
            
            # Mark as completed
            state["completed"] = True
            state["chain_of_thought"].append(f"🚀{step_count + 1}. Hoàn thành xử lý")
            
            # # Generate execution summary
            # agent_summary = ""
            # if "agent_execution_order" in state:
            #     agent_summary = f"Thứ tự thực thi các agent: {', '.join(state['agent_execution_order'])}"
            #     log(f"Agent execution summary: {agent_summary}")
            #     state["chain_of_thought"].append(f"🔍Tóm tắt thực thi: {agent_summary}")
            
            # Add used tools to the summary
            log(f"Used tools: {state.get('used_tools', [])}")
            
            # Get the final reflection response for the main content
            final_reflection_content = ""
            for message in reversed(state["messages"]):
                if isinstance(message, AIMessage) and message.content.startswith("💭"):
                    final_reflection_content = message.content[2:].strip()  # Remove 💭 emoji
                    break
            
            # Return the final state with reflection as main content
            return {
                "response_type": "data",
                "is_task_complete": True,
                "require_user_input": False,
                "content": final_reflection_content if final_reflection_content else state["messages"][-1].content,
                "state": state,
                "chain_of_thought": state["chain_of_thought"],
                "agent_execution_order": state.get("agent_execution_order", []),
                "used_tools": state.get("used_tools", []),
                "agent_results": state.get("agent_results", {})
            }
        except Exception as e:
            log(f"Error running multi-agent system: {e}", level='error')
            return {
                "response_type": "error",
                "content": f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
                "is_task_complete": False,
                "require_user_input": False,
                "chain_of_thought": [f"❌Lỗi: {str(e)}"]
            }
             
    async def stream(self, query: str, session_id: str = "default", user_role: str = "user") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the multi-agent system's response.
        
        Args:
            query: The user's query
            session_id: Session ID for memory management
            user_role: Vai trò của người dùng, mặc định là "user"
            
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
                "used_tools": [],
                "agent_results": {},
                "original_query": query,
                "user_role": user_role  # Thêm vai trò người dùng vào state
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
                        # Check if this is the final reflection message
                        is_reflection = latest_message.content.startswith("💭")
                        content = latest_message.content[2:].strip() if is_reflection else latest_message.content
                        
                        # Yield the partial response
                        yield {
                            "response_type": "text",
                            "content": content,
                            "is_task_complete": chunk.get("success_criteria_met", False),
                            "require_user_input": chunk.get("require_user_input", False),
                            "is_partial": not is_reflection,  # Reflection is the final response
                            "used_tools": chunk.get("used_tools", []),
                            "is_reflection": is_reflection
                        }
            
            # Yield the final complete response
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # Find the final reflection message
            final_message = None
            for message in reversed(final_state["messages"]):
                if isinstance(message, AIMessage) and message.content.startswith("💭"):
                    final_message = message
                    break
            
            # If no reflection found, use the last non-evaluator message
            if not final_message:
                for message in reversed(final_state["messages"]):
                    if isinstance(message, AIMessage) and not message.content.startswith("[Đánh giá nội bộ:"):
                        final_message = message
                        break
            
            if not final_message:
                final_message = final_state["messages"][-1] if final_state["messages"] else AIMessage(content="Không có phản hồi từ hệ thống.")
            
            # Extract content, removing emoji if it's a reflection
            content = final_message.content
            if content.startswith("💭"):
                content = content[2:].strip()
            
            yield {
                "response_type": "text",
                "content": content,
                "is_task_complete": final_state.get("success_criteria_met", False),
                "require_user_input": final_state.get("require_user_input", False),
                "is_partial": False,
                "used_tools": final_state.get("used_tools", []),
                "is_reflection": True
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
    Test the enhanced worker-evaluator multi-agent system with reflection.
    """
    try:
        # Initialize the multi-agent system
        multi_agent = await MultiAgentSystem().initialize()
        session_id = "test_session_reflection_123"
        
        # Test với câu truy vấn metadata và vai trò người dùng
        query1 = "Tìm file có nội dung Kế hoạch sau đó lưu metadata"
        print(f"\nTest Query 1: {query1}")
        print("Running with reflection agent...")
        
        # Test với vai trò admin
        result1 = await multi_agent.run(query1, session_id=f"{session_id}_admin", user_role="admin")
        print(f"\nMain Response: {result1.get('content', 'No content')}")
        print(f"Used tools: {result1.get('used_tools', [])}")
        print(f"Agent execution order: {result1.get('agent_execution_order', [])}")
        
        if result1.get('chain_of_thought'):
            print("\nChain of Thought:")
            for i, thought in enumerate(result1['chain_of_thought'], 1):
                print(f"{i}. {thought}")
        
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import uuid
    asyncio.run(main())