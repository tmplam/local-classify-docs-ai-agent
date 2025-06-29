from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM 
from langchain_google_genai import ChatGoogleGenerativeAI

ollama_chat_model = ChatOllama(model="llama3.2")
ollama_model = OllamaLLM(model="llama3.2")
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key="AIzaSyDO2joR2ymxrurBhCFlL1JpQYKSsrqwAUc"
)