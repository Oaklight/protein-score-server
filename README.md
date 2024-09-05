# Instruction for ESM3 Protein Structure Prediction Server

## 1. Introduction

This server is a protein structure prediction tool based on the ESM3 model. It processes prediction requests from users and returns the predicted scores (pLDDT) for protein structures.

## 2. Usage

Users can send POST requests to `http://140.221.79.21:8000/predict/` to get predictions. The request body should contain two fields: `data` and `type` .

* `data`: String, representing the protein sequence.
* `type`: String, representing the task type, currently only supports "structure".

The header of the request should contain `Content-Type: application/json` .

For example, a POST request example is as follows:

```json
{
    "data": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "type": "structure"
}
```

## 3. Response

The server will return a JSON response containing two fields: `job_id` and `prediction` .

* `job_id`: String, representing the task ID.
* `prediction`: String, representing the prediction result, currently only returning pLDDT scores.

For example, a response example is as follows:

```json
{
    "job_id": "your_job_id", # this is a uuid4 hex string
    "prediction": "your_prediction_score"
}
```

## 4. Error Handling

* If the task queue is full (currently 50 tasks), the server will return a 429 status code with an error message "Job queue is full".
* If the request times out, the server will return a 408 status code with an error message "Request timeout".
* If the requested `type` is not supported, the server will return a 400 status code with an error message "Task type not supported".

## 5. Result Retrieval

Users can send GET requests to `http://140.221.79.21:8000/result/{job_id}` to get prediction results. The header of the request should contain `Content-Type: application/json` .

For example, a GET request example is as follows:

```bash
http://140.221.79.21:8000/result/your_job_id
```

The server will return a JSON response containing two fields: `job_id` and `prediction` .

## 6. Server Configuration

The server uses the `server.yaml` file for configuration. Currently configurable items include:

* `api_key`: API key for Hugging Face Hub login.
* `history_path`: History result storage path.
* `intermediate_pdb_path`: Intermediate PDB file storage path.
* `model`: Model configuration, including model name and number of replicas.
* `task_queue_size`: Task queue size.

## 7. Server Shutdown

To stop the server, simply ctrl+c in the terminal where the server is running.

## 8. License

This server is licensed under the Apache License 2.0.
