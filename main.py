import asyncio
import logging
import os
import sys
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from regex import P

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from server import PredictServer, PredictTask

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a StreamHandler and set its level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Add the StreamHandler to the logger
logger.addHandler(console_handler)


class PredictRequest(BaseModel):
    data: str
    type: str


class PredictResponse(BaseModel):
    job_id: str
    prediction: str


class ResultResponse(BaseModel):
    job_id: str
    prediction: float


# app = FastAPI()
app = FastAPI(debug=True)

config_path = os.path.join(current_dir, "server.yaml")
predict_server = PredictServer(config_path, logger=logger)


@app.post("/predict/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    logger.debug(request)
    if predict_server.task_queue.full():
        raise HTTPException(status_code=429, detail="Job queue is full")
    task = PredictTask(
        id=uuid4().hex,
        data=request.data,
        type=request.type,
    )
    predict_server.task_queue.put(task)

    return PredictResponse(job_id=task.id, prediction="Processing...")


async def check_job_status(job_id: str):
    while job_id in predict_server.working_pool:
        await asyncio.sleep(1)  # Sleep for 1 second before checking again

    if job_id not in predict_server.result_pool:
        logger.info(f"job_id is {job_id}")
        logger.info("job_id not in ", predict_server.result_pool)
        raise HTTPException(status_code=404, detail="Job not found")

    result = predict_server.result_pool[job_id]
    return result


async def wait_for_result(job_id: str, timeout: float):
    try:
        result = await asyncio.wait_for(
            asyncio.shield(check_job_status(job_id)), timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str, timeout: float = 10.0):
    try:
        result = await wait_for_result(job_id, timeout)
        return ResultResponse(job_id=job_id, prediction=result)
    except HTTPException as e:
        raise e


# when app is interrupted, stop the predict server
@app.on_event("shutdown")
async def shutdown_event():
    predict_server.stop_server()
