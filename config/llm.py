from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM 
from langchain_google_genai import ChatGoogleGenerativeAI

ollama_chat_model = ChatOllama(model="mistral")
ollama_model = OllamaLLM(model="mistral")
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key="AIzaSyBL7ZpuDz0CpYMQux0b-Dlyw89_-VNM5Ms"
)