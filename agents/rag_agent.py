import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, AsyncIterable, Dict, List, Optional, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from config.llm import gemini
from agents.base import BaseAgent
import asyncio
import time
from pathlib import Path
import hashlib


class RAGAgent(BaseAgent):
    """RAG Agent for content-based file search."""
    
    def __init__(self):
        super().__init__(
            agent_name='RAGAgent',
            description='Search for files based on their content using RAG',
            content_types=['text', 'text/plain']
        )
        
        self.model = gemini
        # Using 'sentence-transformers/all-MiniLM-L6-v2' - a lightweight but powerful model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU is available
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.index_built = False
        self.last_index_time = 0
        self.index_ttl = 3600  # Rebuild index after 1 hour
        self.data_dir = None
        
    @property
    def name(self):
        return self.agent_name
        
    def _get_cache_dir(self) -> str:
        """Get the cache directory for storing the vector store."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc
        cache_dir = os.path.join(base_dir, 'vector_store')  # lưu vào vector_store
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _get_data_dir_hash(self, data_dir: str) -> str:
        """Generate a hash of the data directory contents for change detection."""
        hasher = hashlib.md5()
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                except Exception as e:
                    log(f"Không thể đọc file {file_path} để tạo hash: {e}", level='warning')
        return hasher.hexdigest()
    
    async def _is_index_up_to_date(self, data_dir: str) -> bool:
        """Check if the existing index is up to date with the data directory."""
        # Check if index exists
        if not os.path.exists(os.path.join(self._get_cache_dir(), 'chroma-collection.parquet')):
            return False
            
        # Check if data directory has changed
        current_hash = self._get_data_dir_hash(data_dir)
        hash_file = os.path.join(self._get_cache_dir(), 'data_hash.txt')
        
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                saved_hash = f.read().strip()
            return saved_hash == current_hash
        return False
    
    async def _load_document(self, file_path: str) -> Optional[Document]:
        """Load document from file path based on file extension."""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                return loader.load()
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
                return loader.load()
            elif file_ext == '.pptx':
                loader = UnstructuredPowerPointLoader(file_path)
                return loader.load()
            else:
                print(f"Unsupported file type: {file_ext}")
                return None
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return None
    
    async def build_index(self, directory_path: str, force_rebuild: bool = False) -> bool:
        """
        Build or load the vector index from documents in the specified directory.
        
        Args:
            directory_path: Path to the directory containing documents
            force_rebuild: If True, force rebuild the index even if it exists
            
        Returns:
            bool: True if index was built/loaded successfully, False otherwise
        """
        try:
            # Check if we can use the existing index
            if not force_rebuild and await self._is_index_up_to_date(directory_path):
                print("Loading existing RAG index from cache...")
                self.vector_store = Chroma(
                    persist_directory=self._get_cache_dir(),
                    embedding_function=self.embeddings
                )
                self.index_built = True
                self.last_index_time = time.time()
                return True
                
            print(f"Building new RAG index from {directory_path}...")
            self.data_dir = directory_path
            
            # Collect all documents
            documents = []
            file_extensions = [".pdf", ".docx", ".pptx", ".txt"]
            
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in file_extensions):
                        file_path = os.path.join(root, file)
                        print(f"Processing: {file_path}")
                        
                        docs = await self._load_document(file_path)
                        if docs:
                            # Add source metadata
                            for doc in docs:
                                doc.metadata["source"] = file_path
                                doc.metadata["filename"] = os.path.basename(file_path)
                            documents.extend(docs)
            
            if not documents:
                print("No documents found to index")
                return False
                
            print(f"Loaded {len(documents)} documents")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            print(f"Split into {len(splits)} chunks")
            
            # Create vector store with cache
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self._get_cache_dir()
            )
            
            # Save the data directory hash
            current_hash = self._get_data_dir_hash(directory_path)
            with open(os.path.join(self._get_cache_dir(), 'data_hash.txt'), 'w') as f:
                f.write(current_hash)
            
            # Create compressor for better retrieval
            compressor = LLMChainExtractor.from_llm(self.model)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
            )
            
            self.index_built = True
            self.last_index_time = time.time()
            print(f"RAG index built successfully with {len(splits)} chunks")
            return True
            
        except Exception as e:
            print(f"Error building RAG index: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _check_index(self) -> bool:
        """Check if index needs to be rebuilt."""
        if not self.index_built or not self.vector_store:
            if self.data_dir:
                return await self.build_index(self.data_dir)
            return False
            
        # Check if index is too old
        if time.time() - self.last_index_time > self.index_ttl:
            print("Index TTL expired, rebuilding...")
            if self.data_dir:
                return await self.build_index(self.data_dir)
            return False
            
        return True
    
    async def search_by_content(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for documents based on content similarity to query."""
        if not await self._check_index():
            return [{"error": "Index not built or failed to build."}]
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Process results
            results = []
            seen_files = set()  # Track unique files
            
            for doc in docs:
                file_path = doc.metadata.get("source")
                if file_path and file_path not in seen_files:
                    seen_files.add(file_path)
                    
                    results.append({
                        "file_path": file_path,
                        "filename": doc.metadata.get("filename", os.path.basename(file_path)),
                        "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "page": doc.metadata.get("page", 1) if "page" in doc.metadata else None
                    })
                    
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            print(f"Error in search_by_content: {e}")
            import traceback
            traceback.print_exc()
            return [{"error": str(e)}]
    
    async def invoke(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Search for documents based on content similarity to the query.
        
        Args:
            query: The search query
            session_id: Optional session ID for tracking
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Ensure index is built
            if not self.index_built or not self.retriever:
                if not self.data_dir:
                    return {
                        'response_type': 'error',
                        'content': 'Thư mục dữ liệu chưa được thiết lập. Vui lòng gọi build_index() trước.',
                        'is_task_complete': True,
                        'require_user_input': False
                    }
                
                # Try to build index if not built yet
                success = await self.build_index(self.data_dir)
                if not success:
                    return {
                        'response_type': 'error',
                        'content': 'Không thể xây dựng chỉ mục tìm kiếm.',
                        'is_task_complete': True,
                        'require_user_input': False
                    }
            
            # Perform similarity search
            relevant_docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.retriever.get_relevant_documents(query)
            )
            
            if not relevant_docs:
                return {
                    'response_type': 'message',
                    'content': 'Không tìm thấy tài liệu nào phù hợp với yêu cầu của bạn.',
                    'is_task_complete': True,
                    'require_user_input': False
                }
            
            # Format results as file paths for consistency with filesystem agent
            unique_files = set()
            for doc in relevant_docs:
                source = doc.metadata.get('source', '')
                if source and os.path.exists(source):
                    unique_files.add(source)
            
            if not unique_files:
                return {
                    'response_type': 'message',
                    'content': 'Không tìm thấy tài liệu nào phù hợp với yêu cầu của bạn.',
                    'is_task_complete': True,
                    'require_user_input': False
                }
            
            # Format results as newline-separated file paths with full paths for compatibility with text extraction
            results = [f"Tôi đã tìm thấy file: {file}" for file in unique_files]
            
            return {
                'response_type': 'message',
                'content': '\n'.join(results),
                'file_paths': list(unique_files),  # Include full paths in a separate field
                'is_task_complete': True,
                'require_user_input': False
            }
            
        except Exception as e:
            log(f"Lỗi khi tìm kiếm nội dung: {str(e)}", level='error')
            return {
                'response_type': 'error',
                'content': f'Có lỗi xảy ra khi tìm kiếm nội dung: {str(e)}',
                'is_task_complete': True,
                'require_user_input': False
            }
    
    async def ainvoke(self, input_data: Union[str, Dict], config: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Asynchronous version of invoke for LangChain compatibility.
        
        Args:
            input_data: Either a string query or a dictionary with an 'input' key
            config: Optional configuration
            **kwargs: Additional arguments
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Extract query from input_data
            if isinstance(input_data, dict):
                if 'input' in input_data:
                    query = input_data['input']
                elif 'query' in input_data:
                    query = input_data['query']
                elif 'messages' in input_data and input_data['messages']:
                    # Handle LangChain message format
                    last_message = input_data['messages'][-1]
                    if hasattr(last_message, 'content'):
                        query = last_message.content
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        query = last_message['content']
                    else:
                        query = str(last_message)
                else:
                    query = str(input_data)
            else:
                query = str(input_data)
            
            # Get session_id from config or kwargs if available
            session_id = None
            if config and 'configurable' in config and 'session_id' in config['configurable']:
                session_id = config['configurable']['session_id']
            elif config and 'configurable' in config and 'thread_id' in config['configurable']:
                session_id = config['configurable']['thread_id']
            elif 'session_id' in kwargs:
                session_id = kwargs['session_id']
            
            # Call the main invoke method
            result = await self.invoke(query, session_id)
            
            # Format response for LangChain
            if 'messages' in input_data:
                if 'error' in result:
                    return {"messages": [AIMessage(content=result.get('content', 'Có lỗi xảy ra'))]}
                return {"messages": [AIMessage(content=result.get('content', ''))]}
            
            # Return in the format expected by the multi-agent system
            return result
            
        except Exception as e:
            log(f"Lỗi trong ainvoke: {str(e)}", level='error')
            return {
                'response_type': 'error',
                'content': f'Có lỗi xảy ra: {str(e)}',
                'is_task_complete': True,
                'require_user_input': False
            }

# For testing
async def main():
    agent = RAGAgent()
    await agent.build_index("C:\\Users\\dhuu3\\Desktop\\local-classify-docs-ai-agent\\data")
    result = await agent.invoke("Tìm kiếm tài liệu về trực quan hóa dữ liệu", "test_session")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
