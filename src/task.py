import hashlib
import json
import logging
import time
from typing import Literal
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PredictTask:

    def __init__(
        self,
        seq: str,
        name: str = None,
        task_type: Literal["pdb", "plddt", "tmscore", "sc-tmscore"] = "plddt",
        seq2: str = None,
        priority: int = 0,
        id: str = None,
        require_gpu: bool = None,
    ):
        self.id = uuid4().hex if id is None else id
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
        self.require_gpu = False if require_gpu is None else require_gpu

    def to_dict(self):
        return {
            "id": self.id,
            "seq": self.seq,
            "name": self.name,
            "seq2": self.seq2,
            "type": self.type,
            "hash": self.hash,
            "priority": self.priority,
            "require_gpu": self.require_gpu,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            seq=data["seq"],
            name=data["name"],
            task_type=data["type"],
            seq2=data["seq2"],
            priority=data["priority"],
            id=data["id"],
            require_gpu=data["require_gpu"],
        )

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

    def validate(self) -> int:
        """
        Validates the task based on its type and required attributes.

        Returns:
            int: A status code indicating the validation result.
                 - 0: Validation successful.
                 - 1: `seq` is required but missing.
                 - 2: `seq2` or `name` is required but missing for `tmscore` task type.
                 - 3: `seq2` or `name` is required but missing for `sc-tmscore` task type.
                 - 4: Unknown task type.

        Raises:
            None: This method does not raise exceptions but logs errors.
        """
        if self.seq is None:
            logging.error(f"[{self.id}] `seq` is required for [{self.type}] task type.")
            return 1  # Indicates that `seq` is required but missing
        match self.type:
            case "pdb" | "plddt":
                return 0  # Indicates successful validation for these task types
            case "tmscore" | "sc-tmscore":
                if self.seq2 is None and self.name is None:
                    logging.error(
                        f"[{self.id}] `seq2` or `name` is required for [{self.type}] task type"
                    )
                    if self.type == "tmscore":
                        return 2
                    else:
                        return 3
                return 0  # Indicates successful validation for these task types
            case _:
                logging.error(f"[{self.id}] Unknown task type: {self.type}")
                return 4  # Indicates an unknown task type
