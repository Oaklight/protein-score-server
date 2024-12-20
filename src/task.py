import hashlib
import json
import time
from uuid import uuid4


class PredictTask:
    def __init__(self, seq, name, task_type, seq2=None, priority=0):
        self.id = uuid4().hex
        self.seq = seq
        self.name = name  # name of reference sequence
        self.seq2 = seq2  # seq of reference sequence
        self.type = task_type
        self.hash = hashlib.md5(
            json.dumps(
                {"seq": seq, "name": name, "seq2": seq2, "type": task_type}
            ).encode("utf-8")
        ).hexdigest()
        self.priority = priority  # smaller number, higher priority
        self.create_time = time.time()
        self.require_gpu = False

    def to_dict(self):
        return {
            "id": self.id,
            "seq": self.seq,
            "name": self.name,
            "seq2": self.seq2,
            "type": self.type,
            "hash": self.hash,
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)
