import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import logging

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.server import PredictServer
from src.task import PredictTask


config_path = os.path.join(current_dir, "server.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])
console_handler = logging.StreamHandler()
console_handler.setLevel(config["logging_level"])
logger.addHandler(console_handler)


class PredictRequest(BaseModel):
    seq: str
    name: str
    type: str


class PredictResponse(BaseModel):
    job_id: str
    prediction: str


class ResultResponse(BaseModel):
    job_id: str
    prediction: float


class StatusResponse(BaseModel):
    busy_models: int
    processed_tasks: int
    remaining_tasks: int


# app = FastAPI()
app = FastAPI(debug=True)

import to_pdb

to_pdb.process()
predict_server = PredictServer(config_path, logger=logger)


@app.get("/status", response_model=StatusResponse)
async def get_status():
    status = predict_server.get_status()
    return StatusResponse(**status)


@app.post("/predict/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if predict_server.task_queue.full():
        raise HTTPException(status_code=429, detail="Job queue is full")
    task = PredictTask(
        seq=request.seq,
        name=request.name,
        task_type=request.type,
    )
    predict_server.task_queue.put(task)

    return PredictResponse(job_id=task.id, prediction="Processing...")


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str):
    if job_id in predict_server.result_pool:
        result = predict_server.result_pool[job_id]
        return ResultResponse(job_id=job_id, prediction=result)
    elif job_id in predict_server.working_pool:
        raise HTTPException(status_code=202, detail="Task is still processing")
    else:
        raise HTTPException(status_code=404, detail="Task not found")


# when app is interrupted, stop the predict server
@app.on_event("shutdown")
async def shutdown_event():
    predict_server.stop_server()
