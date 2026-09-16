"""Small OpenAI-compatible MiniLM embedding service for the LoCoMo clients."""
from __future__ import annotations

import argparse
import base64
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastembed import TextEmbedding
from pydantic import BaseModel
import uvicorn


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoding_format: str = "float"
    dimensions: int | None = None


def create_app(model_name: str) -> FastAPI:
    app = FastAPI()
    encoder = TextEmbedding(model_name=model_name)

    @app.get("/health")
    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        texts = [request.input] if isinstance(request.input, str) else list(request.input)
        vectors = [np.asarray(vector, dtype=np.float32) for vector in encoder.embed(texts)]
        if request.dimensions not in (None, 384):
            raise ValueError(f"MiniLM output dimension is 384, requested {request.dimensions}")
        data = []
        for index, vector in enumerate(vectors):
            if request.encoding_format == "base64":
                value: list[float] | str = base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")
            else:
                value = vector.tolist()
            data.append({"object": "embedding", "index": index, "embedding": value})
        return {
            "object": "list",
            "model": model_name,
            "data": data,
            "usage": {"prompt_tokens": sum(len(text.split()) for text in texts), "total_tokens": sum(len(text.split()) for text in texts)},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()