import logging
from typing import Any

from ai.config import agent

logger = logging.getLogger(__name__)


async def ask_agent(text: str) -> str:
    try:
        msg = {"role": "user", "content": text}
        response = await agent.ainvoke({"messages": [msg]})
    except Exception as e:
        logger.warning("Error calling AI in ask_agent: %s", e)
        return "ai_error"

    ai_message = response["messages"][-1]
    content = getattr(ai_message, "content", "")

    if isinstance(content, str) and content.strip():
        return content

    if isinstance(content, list):
        for item in reversed(content):
            if isinstance(item, dict) and "text" in item and item["text"]:
                return str(item["text"])
            if isinstance(item, str) and item.strip():
                return item

    # Fallback to reasoning_content if content is empty (common in reasoning models)
    additional_kwargs: dict[str, Any] = getattr(ai_message, "additional_kwargs", {})
    reasoning = additional_kwargs.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    return str(content)
