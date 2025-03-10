import json
import random
import time

import requests

bulk_test = 1

# 定义测试数据
sequences = {
    # 1: {
    #     "name": "1a1x.A",
    #     "seq": "GSAGEDVGAPPDHLWVHQEGIYRDEYQRTWVAVVEEETSFLRARVQQIQVPLGDAARPSHLLTSQLPLMWQLYPEERYMDNNSRLWQIQHHLMVRGVQELLLKLLPDD",
    # },
    # 2: {
    #     "name": "1hh5.A",
    #     "seq": "MPKKIILICSPHIDDAASIFLAKGDPKINLLAVLTVVGGRSLDTNTKNALLVTDIFGIEGVPVAAGEEEPLVEGRKPKKDEPGEKGIGSIEYPPEFKNKLHGKHAVDLLIELILKYEPKTIILCPVGSLTNLATAIKEAPEIVERIKEIVFSGGGYTSGDATPVAEYTVYFDPEAAAIVFNTKLKVTMVGLDATAQALVTPEIKARIAAVGTRPAAFLLEVLEYYAKLKPAKKDEYGYLSDPLAVAYIIDPDVMTTRKAPASVDLDGEETVGTVVVDFEEPIPEECKTRVAVKVDYEKFWNMIVAALKRIGDPA",
    # },
    # 3: {
    #     "name": "1bkf.A",
    #     "seq": "GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDKNKPFKFMLGKQEVIRGWEEGVAQMSVGQRAKLTISPDYAYGATGVPGIIPPHATLVFDVELLKLE",
    # },
    # 4: {
    #     "name": "1ezs.A",
    #     "seq": "AESVQPLEKIAPYPQAEKGMKRQVIQLTPQEDESTLKVELLIGQTLEVDCNLHRLGGKLENKTLEGAAAAYYVFDKVSSPVSTRMACPDGKKEKKFVTAYLGDAGMLRYNSKLPIVVYTPDNVDVKYRVWKAEEKIDNAVVR",
    # },
    5: {
        "seq": "MAEVIRSSAFWRSFPIFEEFDSETLCELSGIASYRKWSAGTVIFQRGDQGDYMIVVVSGRIKLSLFTPQGRELMLRQHEAGALFGEMALLDGQPRSADATAVTAAEGYVIGKKDFLALITQRPKTAEAVIRFLCAQLRDTTDRLETIALYDLNARVARFFLATLRQIHGSEMPQSANLRLTLSQTDIASILGASRPKVNRAILSLEESGAIKRADGIICCNVGRLLSIADPEEDLEHHHHHHHH",
        "seq2": "MAEVIRSSAFWRSFPIFEEFDSETLCELSGIASYRKWSAGTVIFQRGDQGDYMIVVVSGRIKLSLFTPQGRELMLRQHEAGALFGEMALLDGQPRSADATAVTAAEGYVIGKKDFLALITQRPKTAEAVIRFLCAQLRDTTDRLETIALYDLNARVARFFLATLRQIHGSEMPQSANLRLTLSQTDIASILGASRPKVNRAILSLEESGAIKRADGIICCNVGRLLSIADPEEDLEHHHHHHHH",
    },
}


# Assuming you have a function to generate test data
def generate_test_data():
    choice = random.choice(list(sequences.keys()))

    # Generate random sequence, name, and type
    test_data = {
        "seq": sequences[choice]["seq"],
        "type": random.choice(
            [
                # "plddt",
                # "tmscore",
                "sc-tmscore",
                # "pdb",
            ]
        ),
        "seq2": sequences[choice].get("seq2", None),  # Include seq2 if it exists
        "name": sequences[choice].get("name", None),  # Default name if not specified
    }

    # Include name if it exists in the sequences dictionary
    if "name" in sequences[choice]:
        test_data["name"] = sequences[choice]["name"]

    return test_data


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
    sleep_base = 3
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
                sleep_interval = sleep_base**retry_count
                print(f"Job queue is full, retrying after delay... {sleep_interval}s")
                time.sleep(sleep_interval)  # Exponential backoff
            elif response.status_code == 408:
                sleep_interval = sleep_base**retry_count
                print(f"Request timeout, retrying after delay... {sleep_interval}s")
                time.sleep(sleep_interval)  # Exponential backoff
            elif response.status_code == 400:
                print(f"Bad request: {response.text}")
                break
            elif response.status_code == 500:
                print(f"Server error: {response.text}")
                break
            else:
                print(f"Unexpected status code: {response.status_code}")
                sleep_interval = sleep_base**retry_count
                print(f"Request failed, retrying after delay... {sleep_interval}s")
                time.sleep(sleep_interval)  # Exponential backoff
            retry_count += 1
        except requests.exceptions.RequestException as e:
            sleep_interval = sleep_base**retry_count
            print(f"Request failed: {e}")
            time.sleep(sleep_interval)  # Exponential backoff

print(job_ids)
plddt_scores = {}
for id in job_ids:
    print(f"Fetching result for job ID: {id}")
    retry_count = 0
    sleep_base = 3
    while retry_count <= 5:  # Maximum of 5 retries
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
                sleep_interval = sleep_base**retry_count
                print(
                    f"Job is processing, be patient, retrying after delay... {sleep_interval}s"
                )
                time.sleep(sleep_interval)  # Exponential backoff
            retry_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(2**retry_count)  # Exponential backoff

t_run_done = time.time() - t_run

# Print response content
print(json.dumps(job_ids, indent=4))
print(f"Time taken: {t_run_done:.4f} seconds")
