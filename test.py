from time import sleep, time
import requests
import json
from tqdm import tqdm

# 定义测试数据
test_data = {
    "data": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "type": "structure",
}

bulk_test = 100

# 将测试数据转换为JSON
test_data_json = json.dumps(test_data)

job_ids = {}
t_run = time()
# 发送POST请求
for i in tqdm(range(bulk_test)):
    response = requests.post(
        "http://140.221.79.21:8000/predict/",
        data=test_data_json,
        headers={"Content-Type": "application/json"},
    )
    while response.status_code != 200:
        if response.status_code == 429:
            sleep(10)  # job queue is full, wait for 10 seconds
        if response.status_code == 408:
            sleep(10)  # request timeout, wait for 10 seconds
        response = requests.post(
            "http://140.221.79.21:8000/predict/",
            data=test_data_json,
            headers={"Content-Type": "application/json"},
        )
    job_ids[response.json()["job_id"]] = -1

plddt_scores = {}
for id in tqdm(job_ids):
    response = requests.get(
        f"http://140.221.79.21:8000/result/{id}",
        headers={"Content-Type": "application/json"},
    )
    job_ids[id] = response.json()["prediction"]
t_run_done = time() - t_run

# 打印响应内容
print(json.dumps(job_ids, indent=4))
print(f"Time taken: {t_run_done:.4f} seconds")
