import json
from time import sleep, time

import requests

# from tqdm import tqdm

# 定义测试数据
test_data = {
    "seq": "GSAGEDVGAPPDHLWVHQEGIYRDEYQRTWVAVVEEETSFLRARVQQIQVPLGDAARPSHLLTSQLPLMWQLYPEERYMDNNSRLWQIQHHLMVRGVQELLLKLLPDD",
    "name": "1a1x.A",
    "type": "tmscore",
}

bulk_test = 2

config_file = "./client.json"
with open(config_file, "r") as f:
    config = json.load(f)

# 将测试数据转换为JSON
test_data_json = json.dumps(test_data)

job_ids = {}
t_run = time()
# 发送POST请求
for i in range(bulk_test):
    print(i)
    response = requests.post(
        f"{config['server']}/predict/",
        data=test_data_json,
        headers={"Content-Type": "application/json"},
    )
    while response.status_code != 200:
        if response.status_code == 429:
            sleep(10)  # job queue is full, wait for 10 seconds
        if response.status_code == 408:
            sleep(10)  # request timeout, wait for 10 seconds
        response = requests.post(
            f"{config['server']}/predict/",
            data=test_data_json,
            headers={"Content-Type": "application/json"},
        )
    job_ids[response.json()["job_id"]] = -1

plddt_scores = {}
for id in job_ids:
    print(id)
    response = requests.get(
        f"{config['server']}/result/{id}?timeout=20.0",
        headers={"Content-Type": "application/json"},
    )
    #    job_ids[id] = response.json()["prediction"]
    job_ids[id] = response.json()
t_run_done = time() - t_run

# 打印响应内容
print(json.dumps(job_ids, indent=4))
print(f"Time taken: {t_run_done:.4f} seconds")
