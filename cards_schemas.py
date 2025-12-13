from pydantic import BaseModel, Field
from typing import List, Optional

class Card(BaseModel):
    question: str = Field(..., min_length=3)
    answer: str = Field(..., min_length=1)
    source_chunk: int
    tags: Optional[List[str]] = None
