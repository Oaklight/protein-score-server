import hashlib
import json
from uuid import uuid4


class PredictTask:
    def __init__(self, seq, name, task_type):
        self.id = uuid4().hex
        self.seq = seq
        self.name = name
        self.type = task_type
        self.hash = hashlib.md5(
            json.dumps({"seq": seq, "name": name, "type": task_type}).encode("utf-8")
        ).hexdigest()

    def to_dict(self):
        return {
            "id": self.id,
            "seq": self.seq,
            "name": self.name,
            "type": self.type,
            "hash": self.hash,
        }
