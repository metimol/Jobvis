from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from const import GROQ_API_KEY

model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    api_key=GROQ_API_KEY,
    max_tokens=8192,
)

agent = create_agent(
    model,
    system_prompt=(
        "You are Jobvis, an AI assistant for searching for job "
        "opportunities. Your job is to help users find the best jobs they "
        "are looking for by using the provided search tools. Be concise, friendly."
    ),
    name="Jobvis",
)
