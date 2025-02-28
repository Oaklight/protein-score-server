from sqlitedict import SqliteDict
from typing import Literal


class ProtCache:

    def __init__(self, db_path="prot_cache.db"):
        self.db_path = db_path
        self.prot_cache = SqliteDict(db_path, tablename="prot_cache", autocommit=True)
        self.task_cache = SqliteDict(db_path, tablename="task_cache", autocommit=True)

    def set(self, key, value, table: Literal["prot_cache", "task_cache"]):
        if table == "prot_cache":
            self.prot_cache[key] = value
        elif table == "task_cache":
            self.task_cache[key] = value

    def get(self, key, table: Literal["prot_cache", "task_cache"]):
        if table == "prot_cache":
            return self.prot_cache.get(key)
        elif table == "task_cache":
            return self.task_cache.get(key)

    def delete(self, key, table: Literal["prot_cache", "task_cache"]):
        if table == "prot_cache":
            del self.prot_cache[key]
        elif table == "task_cache":
            del self.task_cache[key]

    def close(self):
        self.prot_cache.close()
        self.task_cache.close()

if __name__ == "__main__":
    prot_cache = ProtCache()

    prot_cache.set("ATGCGTACGT", "1ABC", table="prot_cache")
    pdb = prot_cache.get("ATGCGTACGT", table="prot_cache")
    print(f"PDB for ATGCGTACGT: {pdb}")

    prot_cache.set("ATGCGTACGT", "2DEF", table="prot_cache")
    pdb = prot_cache.get("ATGCGTACGT", table="prot_cache")
    print(f"Updated PDB for ATGCGTACGT: {pdb}")

    prot_cache.delete("ATGCGTACGT", table="prot_cache")
    pdb = prot_cache.get("ATGCGTACGT", table="prot_cache")
    print(f"PDB for ATGCGTACGT after deletion: {pdb}")

    prot_cache.set("task123", 42.0, table="task_cache")
    result = prot_cache.get("task123", table="task_cache")
    print(f"Result for task123: {result}")

    prot_cache.set("task123", 43.0, table="task_cache")
    result = prot_cache.get("task123", table="task_cache")
    print(f"Updated result for task123: {result}")

    prot_cache.delete("task123", table="task_cache")
    result = prot_cache.get("task123", table="task_cache")
    print(f"Result for task123 after deletion: {result}")

    prot_cache.close()
