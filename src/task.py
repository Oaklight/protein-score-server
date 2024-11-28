import hashlib
import json
import time
from uuid import uuid4


class PredictTask:
    def __init__(self, seq, name, task_type, priority=0):
        self.id = uuid4().hex
        self.seq = seq
        self.name = name
        self.type = task_type
        self.hash = hashlib.md5(
            json.dumps({"seq": seq, "name": name, "type": task_type}).encode("utf-8")
        ).hexdigest()
        self.priority = priority  # smaller number, higher priority
        self.create_time = time.time()
        self.require_gpu = False

    def to_dict(self):
        return {
            "id": self.id,
            "seq": self.seq,
            "name": self.name,
            "type": self.type,
            "hash": self.hash,
        }
