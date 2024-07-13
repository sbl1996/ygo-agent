import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from typing import Union, Dict

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings

import numpy as np
import jax
import jax.numpy as jnp
import flax
from ygoai.rl.jax.agent import RNNAgent

from .models import (
    DuelCreateResponse,
    DuelPredictRequest,
    DuelPredictResponse,
    DuelPredictErrorResponse,
)
from .features import predict, sample_input, init_code_list, PredictState


class Settings(BaseSettings):
    code_list: str = "code_list.txt"
    checkpoint: str = "latest.flax_model"

settings = Settings()

def create_agent():
    return RNNAgent(
        num_layers=2,
        rnn_channels=512,
        use_history=True,
        rnn_type='lstm',
        num_channels=128,
        film=True,
        noam=True,
        version=2,
    )


@jax.jit
def get_probs_and_value(params, rstate, obs):
    agent = create_agent()
    next_rstate, logits, value = agent.apply(params, obs, rstate)[:3]
    probs = jax.nn.softmax(logits, axis=-1)
    return next_rstate, probs, value


def predict_fn(params, rstate, obs):
    obs = jax.tree.map(lambda x: jnp.array([x]), obs)
    rstate, probs, value = get_probs_and_value(params, rstate, obs)
    return rstate, np.array(probs)[0].tolist(), float(np.array(value)[0])


all_models = {}
duel_states: Dict[str, PredictState] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))
    
    init_code_list(settings.code_list)

    agent = create_agent()
    key = jax.random.PRNGKey(0)
    key, agent_key = jax.random.split(key, 2)
    sample_obs = sample_input()
    sample_obs_ = jax.tree.map(lambda x: jnp.array([x]), sample_obs)

    rstate = agent.init_rnn_state(1)
    params = jax.jit(agent.init)(agent_key, sample_obs_, rstate)

    checkpoint = settings.checkpoint
    with open(checkpoint, "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())

    params = jax.device_put(params)
    all_models["param"] = params

    all_models["agent"] = agent

    predict_fn(params, rstate, sample_obs)

    print(f"loaded checkpoint from {checkpoint}")

    state = new_state()
    test_duel_id = "9654823a-23fd-4850-bb-6fec241740b0"
    duel_states[test_duel_id] = state

    yield
    # Clean up the ML models and release the resources
    all_models.clear()


app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def new_state():
    return PredictState(all_models["agent"].init_rnn_state(1))

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

    params = all_models["param"]

    _start = time.time()    
    model_fn = lambda r, x: predict_fn(params, r, x)
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
