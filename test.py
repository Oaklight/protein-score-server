import json
import random
import time

import requests

bulk_test = 20

# 定义测试数据
sequences = {
    1: {
        "name": "1a1x.A",
        "seq": "GSAGEDVGAPPDHLWVHQEGIYRDEYQRTWVAVVEEETSFLRARVQQIQVPLGDAARPSHLLTSQLPLMWQLYPEERYMDNNSRLWQIQHHLMVRGVQELLLKLLPDD",
    },
    2: {
        "name": "1hh5.A",
        "seq": "MPKKIILICSPHIDDAASIFLAKGDPKINLLAVLTVVGGRSLDTNTKNALLVTDIFGIEGVPVAAGEEEPLVEGRKPKKDEPGEKGIGSIEYPPEFKNKLHGKHAVDLLIELILKYEPKTIILCPVGSLTNLATAIKEAPEIVERIKEIVFSGGGYTSGDATPVAEYTVYFDPEAAAIVFNTKLKVTMVGLDATAQALVTPEIKARIAAVGTRPAAFLLEVLEYYAKLKPAKKDEYGYLSDPLAVAYIIDPDVMTTRKAPASVDLDGEETVGTVVVDFEEPIPEECKTRVAVKVDYEKFWNMIVAALKRIGDPA",
    },
    3: {
        "name": "1bkf.A",
        "seq": "GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDKNKPFKFMLGKQEVIRGWEEGVAQMSVGQRAKLTISPDYAYGATGVPGIIPPHATLVFDVELLKLE",
    },
    4: {
        "name": "1ezs.A",
        "seq": "AESVQPLEKIAPYPQAEKGMKRQVIQLTPQEDESTLKVELLIGQTLEVDCNLHRLGGKLENKTLEGAAAAYYVFDKVSSPVSTRMACPDGKKEKKFVTAYLGDAGMLRYNSKLPIVVYTPDNVDVKYRVWKAEEKIDNAVVR",
    },
}


# Assuming you have a function to generate test data
def generate_test_data():
    choice = random.choice(list(sequences.keys()))

    # Generate random sequence, name, and type
    return {
        "seq": sequences[choice]["seq"],
        "name": sequences[choice]["name"],
        "type": random.choice(["plddt", "tmscore"]),
    }


config_file = "./client.json"
with open(config_file, "r") as f:
    config = json.load(f)

# Send POST requests with delay and exponential backoff
job_ids = {}
t_run = time.time()
for i in range(bulk_test):
    print(f"Sending request {i+1}/{bulk_test}")
    test_data = generate_test_data()
    test_data_json = json.dumps(test_data)

    retry_count = 0
    while retry_count < 5:  # Maximum of 5 retries
        try:
            response = requests.post(
                f"{config['server']}/predict/",
                data=test_data_json,
                headers={"Content-Type": "application/json"},
                timeout=60,  # Increase client-side timeout
            )
            if response.status_code == 200:
                job_ids[response.json()["job_id"]] = -1
                break
            elif response.status_code == 429:
                print("Job queue is full, retrying after delay...")
                time.sleep(2**retry_count)  # Exponential backoff
            elif response.status_code == 408:
                print("Request timeout, retrying after delay...")
                time.sleep(2**retry_count)  # Exponential backoff
            retry_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(2**retry_count)  # Exponential backoff

print(job_ids)
plddt_scores = {}
for id in job_ids:
    print(f"Fetching result for job ID: {id}")
    retry_count = 0
    while retry_count < 5:  # Maximum of 5 retries
        try:
            response = requests.get(
                f"{config['server']}/result/{id}",
                headers={"Content-Type": "application/json"},
                timeout=60,  # Increase client-side timeout
            )
            if response.status_code == 200:
                job_ids[id] = response.json()
                break
            elif response.status_code == 202:
                print("Request timeout, retrying after delay...")
                time.sleep(2**retry_count)  # Exponential backoff
            retry_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(2**retry_count)  # Exponential backoff

t_run_done = time.time() - t_run

# Print response content
print(json.dumps(job_ids, indent=4))
print(f"Time taken: {t_run_done:.4f} seconds")
