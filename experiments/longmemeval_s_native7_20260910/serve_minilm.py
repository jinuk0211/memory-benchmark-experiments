"""Serve pinned MiniLM embeddings through a local OpenAI-compatible endpoint."""

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"
    dimensions: int | None = None


def create_app(model_path: str, journal: Path) -> FastAPI:
    model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    if model.max_seq_length != 256 or model.get_sentence_embedding_dimension() != 384:
        raise RuntimeError("Unexpected MiniLM window or dimensions")
    journal.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict:
        if request.model != MODEL_NAME:
            raise HTTPException(400, "Unexpected embedding model")
        if request.encoding_format not in ("float", "base64") or request.dimensions not in (None, 384):
            raise HTTPException(400, "Only 384-dimensional float/base64 embeddings are supported")
        inputs = [request.input] if isinstance(request.input, str) else request.input
        if not inputs:
            raise HTTPException(400, "Input cannot be empty")
        started = time.time()
        features = model.tokenize(inputs)
        token_count = int(features["attention_mask"].sum())
        vectors = model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": started, "model": MODEL_NAME,
                "input_count": len(inputs), "prompt_tokens": token_count,
                "elapsed_seconds": time.time() - started,
                "device": "cpu", "dtype": "float32", "max_seq_length": 256}) + "\n")
        values = [vector.tolist() if request.encoding_format == "float" else
                  base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")
                  for vector in vectors]
        return {"object": "list", "model": MODEL_NAME,
            "data": [{"object": "embedding", "index": index, "embedding": vector}
                     for index, vector in enumerate(values)],
            "usage": {"prompt_tokens": token_count, "total_tokens": token_count}}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model_path, args.journal), host="127.0.0.1", port=args.port)