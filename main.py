import os
import sys
from typing import Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import logging

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import to_pdb
from src.server import PredictServer
from src.task import PredictTask
from src.scheduler import TASK_STATES

config_path = os.path.join(current_dir, "server.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])

# Create a file handler for INFO and DEBUG messages
info_handler = logging.FileHandler(config["info_log_file"])
info_handler.setLevel(logging.INFO)  # Set the handler level to INFO

# Create a file handler for ERROR and higher messages
error_handler = logging.FileHandler(config["error_log_file"])
error_handler.setLevel(logging.ERROR)  # Set the handler level to ERROR

# Create a console handler for all levels (if needed)
console_handler = logging.StreamHandler()
console_handler.setLevel(config["logging_level"])

# Create formatters and add them to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)


class PredictRequest(BaseModel):
    seq: str
    name: Optional[str] = None
    seq2: Optional[str] = None
    type: str


class PredictResponse(BaseModel):
    job_id: str
    prediction: str


class ResultResponse(BaseModel):
    job_id: str
    prediction: float | str


class StatusResponse(BaseModel):
    busy_models: int
    busy_models_details: Dict[
        str, bool
    ]  # This will directly return a dictionary of string to bool mapping
    processed_tasks: int
    remaining_tasks: int
    working_tasks: List[str]  # This will directly return a list of strings


# app = FastAPI()
app = FastAPI(debug=True)


to_pdb.process()
predict_server = PredictServer(config_path, logger=logger)


@app.get("/status", response_model=StatusResponse)
async def get_status():
    status = predict_server.get_status()
    return StatusResponse(**status)


@app.post("/predict/", response_model=PredictResponse)
async def predict(request: PredictRequest):

    task = PredictTask(
        seq=request.seq,
        name=request.name,
        task_type=request.type,
        seq2=request.seq2,
    )

    validate_value = task.validate()
    if validate_value == 0:
        predict_server.task_scheduler.add_task(task)
    else:
        error_messages = {
            1: "Sequence (seq) is required but missing",
            2: "Second sequence (seq2) or name is required for tmscore task type",
            3: "Second sequence (seq2) or name is required for sc-tmscore task type",
            4: f"Unknown task type: {request.type}",
        }
        raise HTTPException(
            status_code=400, detail=error_messages.get(validate_value, "Invalid task")
        )
    return PredictResponse(job_id=task.id, prediction="Processing...")


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str):
    logger.info(f"Checking status for job_id: {job_id}")

    # Check task status using scheduler
    task_data = predict_server.task_scheduler.task_db.get(job_id)
    if task_data is None:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Task not found")

    if task_data["status"] == TASK_STATES["PENDING"]:
        logger.info(f"Job {job_id} is pending")
        raise HTTPException(
            status_code=202, detail="Task is queued, waiting to be processed."
        )
    elif task_data["status"] == TASK_STATES["PROCESSING"]:
        logger.info(f"Job {job_id} is processing")
        raise HTTPException(status_code=202, detail="Task is being processed.")
    elif task_data["status"] == TASK_STATES["COMPLETED"]:
        logger.info(f"Job {job_id} is completed")
        result = task_data["result"]
        return ResultResponse(job_id=job_id, prediction=result)

    # If not found anywhere, return 404
    logger.warning(f"Job {job_id} not found in any pool")
    raise HTTPException(status_code=404, detail="Task not found")


# when app is interrupted, stop the predict server
@app.on_event("shutdown")
async def shutdown_event():
    predict_server.stop_server()
