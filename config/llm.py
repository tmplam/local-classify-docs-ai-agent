from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from dotenv import find_dotenv
load_dotenv(find_dotenv(), override=True)

ollama_chat_model = ChatOllama(model="mist")
ollama_model = OllamaLLM(model="mistral")
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

DATA_DIR = os.getenv("DATA_DIR")