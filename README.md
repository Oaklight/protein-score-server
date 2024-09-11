# Instruction for ESM3 Protein Structure Prediction Server

## 1. Introduction

This server is a protein structure prediction tool based on the ESM3 model. It processes prediction requests from users and returns the predicted scores (pLDDT) for protein structures.

## 2. Installation

To install the environment, you need to clone this repository and install the required dependencies.

```bash
git clone https://github.com/Oaklight/esm3-plddt-server.git
cd esm3-plddt-server
conda env create -f env/environment.yml
conda activate esm
pip install -r env/requirements.txt
```

## 3. Server Configuration

The server uses the `server.yaml` file for configuration. Currently configurable items include:

* `api_key`: API key for Hugging Face Hub login.
* `history_path`: History result storage path.
* `intermediate_pdb_path`: Intermediate PDB file storage path.
* `model`: Model configuration
    - `name`: model name, `esm3_sm_open_v1` for now and please don't change
    - `replica`: optional GPU device and replications mapping
* `task_queue_size`: Task queue size, default to 50.

After the config are set, run these commands inside the project folder:

```shell
conda activate esm
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4. Usage

For details, please refer to [test.py](./test.py)

### 4.1. Request Prediction

Users can send `POST` requests to `http://your-host:8000/predict/` to get predictions. The request body should contain two fields: `data` and `type` .

* `data`: String, representing the protein sequence.
* `type`: String, representing the task type, currently only supports "structure".

The server will return a JSON response containing two fields: `job_id` and `prediction` .

* `job_id`: String, representing the task ID.
* `prediction`: String, currently only indicating the prediction is in processing.

```json
{
    "job_id": "0a98a981748c4b7eacfd5e0957905ced", # this is a uuid4 hex string
    "prediction": ... # not very useful at this moment
}
```

### 4.2. Result Retrieval

Users can send `GET` requests to `http://your-host:8000/result/{job_id}` to get prediction results. The header of the request should contain `Content-Type: application/json` .

The server will return a JSON response containing two fields: `job_id` and `prediction` .

```json
{
    "job_id": "0a98a981748c4b7eacfd5e0957905ced", # this is a uuid4 hex string
    "prediction": 0.983124
}
```

### 4.3. Error Handling

* If the task queue is full (currently 50 tasks), the server will return a `429` status code with an error message "Job queue is full".
* If the request times out, the server will return a `408` status code with an error message "Request timeout".
* If the requested `type` is not supported, the server will return a `400` status code with an error message "Task type not supported".

| Status Code | Error Message |
| --- | --- |
| 429 | Job queue is full |
| 408 | Request timeout |
| 400 | Task type not supported |

## 5. Server Shutdown

To stop the server, simply `ctrl+c` in the terminal where the server is running.

## 6. License

This server is licensed under the Apache License 2.0.
