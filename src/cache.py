import sqlite3
from contextlib import contextmanager
from threading import Lock
from typing import Literal


class ProtCache:
    def __init__(self, db_path="prot_cache.db", pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connection_pool = [self._create_connection() for _ in range(pool_size)]
        self.lock = Lock()
        self._initialize_db()

    def _create_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _initialize_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prot_cache (
                    bio_sequence TEXT PRIMARY KEY,
                    pdb TEXT
                )
            """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS task_cache (
                    task_hash TEXT PRIMARY KEY,
                    result REAL
                )
            """
            )
            conn.commit()

    @contextmanager
    def _get_connection(self):
        with self.lock:
            conn = self.connection_pool.pop()
        try:
            yield conn
        finally:
            self.connection_pool.append(conn)

    def set(self, key, value, table: Literal["prot_cache", "task_cache"]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if table == "prot_cache":
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO prot_cache (bio_sequence, pdb)
                    VALUES (?, ?)
                """,
                    (key, value),
                )
            elif table == "task_cache":
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO task_cache (task_hash, result)
                    VALUES (?, ?)
                """,
                    (key, value),
                )
            conn.commit()

    def get(self, key, table: Literal["prot_cache", "task_cache"]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if table == "prot_cache":
                cursor.execute(
                    "SELECT pdb FROM prot_cache WHERE bio_sequence = ?", (key,)
                )
            elif table == "task_cache":
                cursor.execute(
                    "SELECT result FROM task_cache WHERE task_hash = ?", (key,)
                )
            result = cursor.fetchone()
            return result[0] if result else None

    def delete(self, key, table: Literal["prot_cache", "task_cache"]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if table == "prot_cache":
                cursor.execute("DELETE FROM prot_cache WHERE bio_sequence = ?", (key,))
            elif table == "task_cache":
                cursor.execute("DELETE FROM task_cache WHERE task_hash = ?", (key,))
            conn.commit()

    def close(self):
        for conn in self.connection_pool:
            conn.close()


# Example usage:
if __name__ == "__main__":
    prot_cache = ProtCache()

    # Set a new entry in prot_cache
    prot_cache.set("ATGCGTACGT", "1ABC", table="prot_cache")

    # Get an entry from prot_cache
    pdb = prot_cache.get("ATGCGTACGT", table="prot_cache")
    print(f"PDB for ATGCGTACGT: {pdb}")

    # Update the entry in prot_cache
    prot_cache.set("ATGCGTACGT", "2DEF", table="prot_cache")

    # Get the updated entry from prot_cache
    pdb = prot_cache.get("ATGCGTACGT", table="prot_cache")
    print(f"Updated PDB for ATGCGTACGT: {pdb}")

    # Delete an entry from prot_cache
    prot_cache.delete("ATGCGTACGT", table="prot_cache")

    # Verify deletion from prot_cache
    pdb = prot_cache.get("ATGCGTACGT", table="prot_cache")
    print(f"PDB for ATGCGTACGT after deletion: {pdb}")

    # Set a new entry in task_cache
    prot_cache.set("task123", 42.0, table="task_cache")

    # Get an entry from task_cache
    result = prot_cache.get("task123", table="task_cache")
    print(f"Result for task123: {result}")

    # Update the entry in task_cache
    prot_cache.set("task123", 43.0, table="task_cache")

    # Get the updated entry from task_cache
    result = prot_cache.get("task123", table="task_cache")
    print(f"Updated result for task123: {result}")

    # Delete an entry from task_cache
    prot_cache.delete("task123", table="task_cache")

    # Verify deletion from task_cache
    result = prot_cache.get("task123", table="task_cache")
    print(f"Result for task123 after deletion: {result}")

    # Close the cache
    prot_cache.close()
