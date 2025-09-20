from typing import List, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel


class ChatInputType(BaseModel):
    input: List[Union[HumanMessage, AIMessage, SystemMessage]]
