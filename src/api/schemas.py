from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="ID do modelo registrado na API",
        examples=["causal-gpt-neo-125m", "causal-opt-1.3b", "seq2seq-mt5-small", "seq2seq-bart-large"],
    )
    instruction: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Pergunta ou instrução para o modelo",
    )
    max_new_tokens: int = Field(
        default=200,
        ge=10,
        le=500,
        description="Número máximo de tokens a gerar",
    )


class ChatResponse(BaseModel):
    model_id: str
    instruction: str
    response: str


class ModelInfo(BaseModel):
    id: str
    nome: str
    descricao: str
    tipo: str


class HealthResponse(BaseModel):
    status: str
    modelos_carregados: list[str]
    device: str
