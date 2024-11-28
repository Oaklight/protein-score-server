import json
import time
import requests

config_file = "./client.json"
with open(config_file, "r") as f:
    config = json.load(f)
list_404 = [
    "92b81a3b16554aa8ac40cc714b077d3d",
    "92b81a3b16554aa8ac40cc714b077d3d",
    "92b81a3b16554aa8ac40cc714b077d3d",
    "92b81a3b16554aa8ac40cc714b077d3d",
    "92b81a3b16554aa8ac40cc714b077d3d",
    "e34d13acb4b948deb54b694d08813857",
    "e34d13acb4b948deb54b694d08813857",
    "e34d13acb4b948deb54b694d08813857",
    "e34d13acb4b948deb54b694d08813857",
    "e34d13acb4b948deb54b694d08813857",
    "a98fed66b8164ffba9b0b97fe4717b83",
    "a98fed66b8164ffba9b0b97fe4717b83",
    "a98fed66b8164ffba9b0b97fe4717b83",
    "a98fed66b8164ffba9b0b97fe4717b83",
    "a98fed66b8164ffba9b0b97fe4717b83",
    "be90f113f73549b08ca4fd187580e3f2",
    "be90f113f73549b08ca4fd187580e3f2",
    "be90f113f73549b08ca4fd187580e3f2",
    "be90f113f73549b08ca4fd187580e3f2",
    "be90f113f73549b08ca4fd187580e3f2",
    "a3638b4d51474dd783bdc8bf4c591b65",
    "a3638b4d51474dd783bdc8bf4c591b65",
    "a3638b4d51474dd783bdc8bf4c591b65",
    "a3638b4d51474dd783bdc8bf4c591b65",
    "a3638b4d51474dd783bdc8bf4c591b65",
    "cf98dd20b1f149d8839816057aca4a69",
    "cf98dd20b1f149d8839816057aca4a69",
    "cf98dd20b1f149d8839816057aca4a69",
    "cf98dd20b1f149d8839816057aca4a69",
    "cf98dd20b1f149d8839816057aca4a69",
    "4aaab30b62af4616a74c26e7c3fe45d4",
    "4aaab30b62af4616a74c26e7c3fe45d4",
    "4aaab30b62af4616a74c26e7c3fe45d4",
    "4aaab30b62af4616a74c26e7c3fe45d4",
    "4aaab30b62af4616a74c26e7c3fe45d4",
    "dacb0e10be04405f8d577d3f12319e20",
    "dacb0e10be04405f8d577d3f12319e20",
    "dacb0e10be04405f8d577d3f12319e20",
    "dacb0e10be04405f8d577d3f12319e20",
    "dacb0e10be04405f8d577d3f12319e20",
    "466e037e3ebd4c89beae52f380632538",
    "466e037e3ebd4c89beae52f380632538",
    "466e037e3ebd4c89beae52f380632538",
    "466e037e3ebd4c89beae52f380632538",
    "466e037e3ebd4c89beae52f380632538",
    "1af6df9479e5432a9cca5fefe1d77929",
    "1af6df9479e5432a9cca5fefe1d77929",
    "1af6df9479e5432a9cca5fefe1d77929",
    "1af6df9479e5432a9cca5fefe1d77929",
    "1af6df9479e5432a9cca5fefe1d77929",
    "cc2b59319a7a4da697dd94e58ce69d66",
    "cc2b59319a7a4da697dd94e58ce69d66",
    "cc2b59319a7a4da697dd94e58ce69d66",
    "cc2b59319a7a4da697dd94e58ce69d66",
    "cc2b59319a7a4da697dd94e58ce69d66",
    "89e222eb54894c30afc59492bc8c9b87",
    "89e222eb54894c30afc59492bc8c9b87",
    "89e222eb54894c30afc59492bc8c9b87",
    "89e222eb54894c30afc59492bc8c9b87",
    "89e222eb54894c30afc59492bc8c9b87",
    "3d1b69a7f611491893e615785e308ea4",
    "3d1b69a7f611491893e615785e308ea4",
    "3d1b69a7f611491893e615785e308ea4",
    "3d1b69a7f611491893e615785e308ea4",
    "3d1b69a7f611491893e615785e308ea4",
    "4e057ca48e394e6f889353a29970664f",
    "4e057ca48e394e6f889353a29970664f",
    "4e057ca48e394e6f889353a29970664f",
    "4e057ca48e394e6f889353a29970664f",
    "4e057ca48e394e6f889353a29970664f",
    "6620c7a0818c441196845b2078c2e2e1",
    "6620c7a0818c441196845b2078c2e2e1",
    "6620c7a0818c441196845b2078c2e2e1",
    "6620c7a0818c441196845b2078c2e2e1",
    "6620c7a0818c441196845b2078c2e2e1",
    "db69e9150faf43d5af6b29fcbf80683e",
    "db69e9150faf43d5af6b29fcbf80683e",
    "db69e9150faf43d5af6b29fcbf80683e",
    "db69e9150faf43d5af6b29fcbf80683e",
    "db69e9150faf43d5af6b29fcbf80683e",
    "95f6616d12ae409d951a5137aefc7086",
    "95f6616d12ae409d951a5137aefc7086",
    "95f6616d12ae409d951a5137aefc7086",
    "95f6616d12ae409d951a5137aefc7086",
    "95f6616d12ae409d951a5137aefc7086",
    "63dd9dafaa164060aa595c492a3babd6",
    "63dd9dafaa164060aa595c492a3babd6",
    "63dd9dafaa164060aa595c492a3babd6",
    "63dd9dafaa164060aa595c492a3babd6",
    "63dd9dafaa164060aa595c492a3babd6",
    "6b990eb955274f949413e32dc73ac4a8",
    "6b990eb955274f949413e32dc73ac4a8",
    "6b990eb955274f949413e32dc73ac4a8",
    "6b990eb955274f949413e32dc73ac4a8",
    "6b990eb955274f949413e32dc73ac4a8",
    "3d37a1ffe96a462685d89a34817a1d15",
    "3d37a1ffe96a462685d89a34817a1d15",
    "3d37a1ffe96a462685d89a34817a1d15",
    "3d37a1ffe96a462685d89a34817a1d15",
    "3d37a1ffe96a462685d89a34817a1d15",
    "07954f24fb94479f931108fdd3999b12",
    "07954f24fb94479f931108fdd3999b12",
    "07954f24fb94479f931108fdd3999b12",
    "07954f24fb94479f931108fdd3999b12",
    "07954f24fb94479f931108fdd3999b12",
    "4a3f17e5bd394621a0c09b062459af27",
    "4a3f17e5bd394621a0c09b062459af27",
    "4a3f17e5bd394621a0c09b062459af27",
    "4a3f17e5bd394621a0c09b062459af27",
    "4a3f17e5bd394621a0c09b062459af27",
    "04727d87e1d24723a4c35a1deb26420e",
    "04727d87e1d24723a4c35a1deb26420e",
    "04727d87e1d24723a4c35a1deb26420e",
    "04727d87e1d24723a4c35a1deb26420e",
    "04727d87e1d24723a4c35a1deb26420e",
    "e7b04afd69034a27aa7c004426d2d66f",
    "e7b04afd69034a27aa7c004426d2d66f",
    "e7b04afd69034a27aa7c004426d2d66f",
    "e7b04afd69034a27aa7c004426d2d66f",
    "e7b04afd69034a27aa7c004426d2d66f",
    "29a0648b6e0141ae95b256acf15c4179",
    "29a0648b6e0141ae95b256acf15c4179",
    "29a0648b6e0141ae95b256acf15c4179",
    "29a0648b6e0141ae95b256acf15c4179",
    "29a0648b6e0141ae95b256acf15c4179",
    "6a455fae5bb24518a3149b2f27f99a8a",
    "6a455fae5bb24518a3149b2f27f99a8a",
    "6a455fae5bb24518a3149b2f27f99a8a",
    "6a455fae5bb24518a3149b2f27f99a8a",
    "6a455fae5bb24518a3149b2f27f99a8a",
    "5eb4677a973a43e494fd9df06d8bb8e3",
    "5eb4677a973a43e494fd9df06d8bb8e3",
    "5eb4677a973a43e494fd9df06d8bb8e3",
    "5eb4677a973a43e494fd9df06d8bb8e3",
    "5eb4677a973a43e494fd9df06d8bb8e3",
    "16e1235939674cc6b8c016db68ceee53",
    "16e1235939674cc6b8c016db68ceee53",
    "16e1235939674cc6b8c016db68ceee53",
    "16e1235939674cc6b8c016db68ceee53",
    "16e1235939674cc6b8c016db68ceee53",
    "e64b45f6d2e24be981f727851a845a2b",
    "e64b45f6d2e24be981f727851a845a2b",
    "e64b45f6d2e24be981f727851a845a2b",
    "e64b45f6d2e24be981f727851a845a2b",
    "e64b45f6d2e24be981f727851a845a2b",
    "d76a7c324c2947c3866a782f03cf09d1",
    "d76a7c324c2947c3866a782f03cf09d1",
    "d76a7c324c2947c3866a782f03cf09d1",
    "d76a7c324c2947c3866a782f03cf09d1",
    "d76a7c324c2947c3866a782f03cf09d1",
]


list_404_dedup = list(set(list_404))

job_ids = {each: None for each in list_404_dedup}

print(len(list_404))
print(len(job_ids))

t_run = time.time()

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
