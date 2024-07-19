import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from typing import Union, Dict

import time
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings


from .models import (
    DuelCreateResponse,
    DuelPredictRequest,
    DuelPredictResponse,
    DuelPredictErrorResponse,
)
from .features import predict, init_code_list, PredictState, Predictor


class Settings(BaseSettings):
    code_list: str = "code_list.txt"
    checkpoint: str = "latest.flax_model"
    enable_cors: bool = Field(default=True, description="Enable CORS")
    state_expire: int = Field(default=3600, description="Duel state expire time in seconds")
    test_duel_id: str = Field(default="9654823a-23fd-4850-bb-6fec241740b0", description="Test duel id")
    ygo_num_threads: int = Field(default=1, description="Number of threads to use for YGO prediction")

settings = Settings()

all_models = {}
duel_states: Dict[str, PredictState] = {}

def delete_outdated_states():
    while True:
        current_time = time.time()
        for k, v in list(duel_states.items()):
            if k == settings.test_duel_id:
                continue
            if current_time - v._timestamp > settings.state_expire:
                del duel_states[k]
        time.sleep(600)

# Start the thread to delete outdated states
thread = threading.Thread(target=delete_outdated_states)
thread.daemon = True
thread.start()

@asynccontextmanager
async def lifespan(app: FastAPI):    
    init_code_list(settings.code_list)

    checkpoint = settings.checkpoint
    predictor = Predictor.load(checkpoint, settings.ygo_num_threads)
    all_models["default"] = predictor
    print(f"loaded checkpoint from {checkpoint}")

    state = new_state()
    test_duel_id = settings.test_duel_id
    duel_states[test_duel_id] = state

    yield
    # Clean up the ML models and release the resources
    all_models.clear()


app = FastAPI(
    lifespan=lifespan,
)

if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def new_state():
    return PredictState()

@app.get('/', status_code=200, response_class=PlainTextResponse)
async def root():
    return "OK"


@app.post('/v0/duels', response_model=DuelCreateResponse)
async def create_duel() -> DuelCreateResponse:
    """
    Create duel
    """
    duel_id = str(uuid.uuid4())
    state = new_state()
    duel_states[duel_id] = state
    return DuelCreateResponse(duelId=duel_id, index=state.index)


@app.delete('/v0/duels/{duelId}', status_code=204)
async def delete_duel(
    duel_id: str = Path(..., alias='duelId')
) -> None:
    """
    Delete duel
    """
    if duel_id in duel_states:
        duel_states.pop(duel_id)


@app.post(
    '/v0/duels/{duelId}/predict',
)
async def duel_predict(
    duel_id: str = Path(..., alias='duelId'), body: DuelPredictRequest = None
) -> Union[DuelPredictResponse, DuelPredictErrorResponse]:
    index = body.index
    if duel_id not in duel_states:
        return DuelPredictErrorResponse(
            error=f"duel {duel_id} not found"
        )
    duel_state = duel_states[duel_id]
    if index != duel_state.index:
        return DuelPredictErrorResponse(
            error=f"index mismatch: expected {duel_state.index}, got {index}"
        )

    predictor = all_models["default"]
    model_fn = predictor.predict

    _start = time.time()    
    try:
        predict_results = predict(model_fn, body.input, body.prev_action_idx, duel_state)
    except (KeyError, NotImplementedError) as e:
        return DuelPredictErrorResponse(
            error=f"{e}"
        )
    predict_time = time.time() - _start

    print(f"predict time: {predict_time:.3f}")
    return DuelPredictResponse(
        index=duel_state.index,
        predict_results=predict_results,
    )
