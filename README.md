# Protein Structure Score Server

## 1. Introduction

This server is a protein structure prediction tool that processes prediction requests from users and capable of returning various scores for protein sequences.

## 2. Installation

To install the environment, follow these steps:

```bash
git clone https://github.com/Oaklight/protein-score-server.git
cd protein-score-server
conda env create -f environment.yaml
conda activate esm
pip install -r requirements.txt
```

## 3. Server Configuration

**Configuration File**: 
   - Copy `server.yaml.sample` to `server.yaml` :

     

```bash
cp server.yaml.sample server.yaml
```

   - Edit `server.yaml` with your settings.

The server uses the `server.yaml` file for configuration. Currently configurable items include:

* `api_key`: API key for Hugging Face Hub login.
* `history_path`: History result storage path.
* `intermediate_pdb_path`: Intermediate PDB file storage path.
* `model`: Model configuration
    - `name`: model name, `esmfold` or `protenix (bytedances' alphafold3 implementation)`
    - `replica`: GPU device and replications mapping, should be in `<device>: <num_replica>` format. For `esmfold` case, use `_: <num_replica>` instead.
* `task_queue_size`: Task queue size, default to 50.
* `timeout`: Timeout for async prediction result retrieval, default to 15 seconds.
* `backbone_pdb`:
    - `reversed_index`: path for reverse index from pdb id to pdb file path
    - `parquet_prefix`: path prefix for parquet files
    - `pdb_prefix`: path prefix for pdb files

For example, see [ `server.yaml` ](./server.yaml)

After the config are set, run these commands inside the project folder:

```bash
conda activate esm
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4. Usage

### 4.1. Request Prediction

Users can send `POST` requests to `http://your-host:8000/predict/` to get predictions. The request body comprises of these fields: `seq` , `name` , `type` , `seq2` .

* `seq`: String, representing the protein sequence.
* `name`: String, representing the name of the reference protein.
* `type`: String, representing the task type, currently supports **"plddt", "tmscore", "sc-tmscore", "pdb"**.
* `seq2`: String, representing the sequence of the reference protein. **Used only for `sc-tmscore` task. You may choose to provide either `seq2` or `name`**

1. **pLDDT**

```json
{
    "seq": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST",
    "type": "plddt"
}
```

2. **TMscore**

```json
{
    "seq": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST",
    "name": "1a0a.A", # must provide for tasks that require a reference structure
    "type": "tmscore"
}
```

3. **sc-TMscore**

```json
{
    "seq": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST",
    "seq2": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST", # choose to provide either seq2 or name
    "type": "sc-tmscore"
}
```

or

```json
{
    "seq": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST",
    "name": "1a0a.A", # choose to provide either seq2 or name
    "type": "sc-tmscore"
}
```

4. **pdb**

```json
{
    "seq": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST",
    "type": "pdb"
}
```

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

When querying for results, use the following guidelines based on the status code:
- **102 Processing**: The task is queued. Wait a few seconds before checking again.
- **202 Accepted**: The task is being processed. Wait a few seconds before checking again.
- **200 OK**: The task is complete. The result is available in the response.
- **404 Not Found**: The task ID is invalid. Check the ID or resubmit the task.
- **429 Too Many Requests**: The server is busy. Wait and try again later.

### 4.4. Retry Strategy

* Recommend to use an exponential backoff strategy with a base of 3 when querying for results.
* Example of querying is available in [`test.py`](test.py).

## 5. Server Shutdown

To stop the server, use `Ctrl+C` in the terminal where the server is running.

## 6. License

This server is licensed under the Apache License 2.0.
