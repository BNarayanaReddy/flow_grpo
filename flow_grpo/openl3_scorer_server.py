import numpy as np
import openl3

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

SAMPLE_RATE = 48000

print("Loading OpenL3 model...")
model = openl3.models.load_audio_embedding_model(
    input_repr="linear",
    content_type="env",
    embedding_size=512
)
print("OpenL3 model loaded.")


from typing import List

class RewardRequest(BaseModel):
    predictions: List[List[float]]
    references: List[List[float]]


@app.post("/openl3_reward")
def reward(req: RewardRequest):

    rewards = []

    for pred, ref in zip(req.predictions, req.references):

        pred = np.asarray(pred, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)

        emb1, _ = openl3.get_audio_embedding(
            pred,
            SAMPLE_RATE,
            model=model
        )

        emb2, _ = openl3.get_audio_embedding(
            ref,
            SAMPLE_RATE,
            model=model
        )

        emb1 = emb1.mean(axis=0)
        emb2 = emb2.mean(axis=0)

        score = (
            np.dot(emb1, emb2)
            / (
                np.linalg.norm(emb1)
                * np.linalg.norm(emb2)
                + 1e-8
            )
        )

        rewards.append(float(score))

    return {"rewards": rewards}