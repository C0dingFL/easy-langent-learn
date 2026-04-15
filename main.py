import langchain
import langgraph
import openai
import importlib
from dotenv import load_dotenv
load_dotenv()

print("LangChain版本：", langchain.__version__)
print("LangGraph版本：", importlib.metadata.version("langgraph"))
print("OpenAI版本：", openai.__version__)