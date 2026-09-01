from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from const import GOOGLE_API_KEY

model = init_chat_model(
    model="google_genai:gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
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
