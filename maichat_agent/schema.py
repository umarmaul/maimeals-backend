from typing import List
from pydantic import BaseModel


class ChatInputType(BaseModel):
    input: List[dict]
