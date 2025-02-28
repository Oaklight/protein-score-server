from sqlitedict import SqliteDict
from typing import Optional, Dict, Any, Literal
from datetime import datetime, timedelta
from task import PredictTask  # Import the PredictTask class
import threading

TASK_STATES = {"PENDING": 0, "PROCESSING": 1, "COMPLETED": 2, "FAILED": 3}


class TaskScheduler:

    def __init__(self, db_path="task_scheduler.db"):
        self.db_path = db_path
        self.task_db = SqliteDict(db_path, tablename="tasks", autocommit=True)
        self.lock = threading.Lock()

    def add_task(self, task: PredictTask) -> None:
        """Add a new task to the scheduler."""
        with self.lock:
            task_data = {
                "task": task.to_dict(),  # Store the PredictTask as a dictionary
                "status": TASK_STATES["PENDING"],
                "priority": task.priority,
                "created_at": datetime.now().isoformat(),
                "scheduled_at": datetime.now().isoformat(),
                "result": None,
            }
            self.task_db[task.id] = task_data

    def change_task_status(
        self,
        task_id: str,
        status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"],
    ) -> None:
        """Change the status of a task."""
        with self.lock:
            if task_id in self.task_db:
                task_data = self.task_db[task_id]
                task_data["status"] = TASK_STATES[status]
                self.task_db[task_id] = task_data

    def expedite_task(self, task_id: str) -> None:
        """Expedite a task by increasing its priority number by 1."""
        with self.lock:
            if task_id in self.task_db:
                task_data = self.task_db[task_id]
                task_data["priority"] += 1
                self.task_db[task_id] = task_data

    def reschedule_task(self, task_id: str, new_time: datetime) -> None:
        """Reschedule a task to a new time."""
        with self.lock:
            if task_id in self.task_db:
                task_data = self.task_db[task_id]
                task_data["scheduled_at"] = new_time.isoformat()
                self.task_db[task_id] = task_data

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a task."""
        with self.lock:
            if task_id in self.task_db:
                task_data = self.task_db[task_id]
                return task_data["result"]
            return None

    def get_task(self, task_id: str) -> Optional[PredictTask]:
        """Retrieve the PredictTask object for a given task ID."""
        with self.lock:
            if task_id in self.task_db:
                task_data = self.task_db[task_id]
                return PredictTask.from_dict(task_data["task"])
            return None

    def get_next_task_with_max_priority(self) -> Optional[str]:
        """
        Retrieve the ID of the next task with the highest priority number that is in the PENDING state.
        Uses creation time as a tie-breaker (older tasks have higher priority).

        Returns:
            Optional[str]: The ID of the next task with the highest priority,
                          or None if no such task exists.
        """
        with self.lock:
            next_task_id = None
            max_priority = -1
            oldest_creation = None

            for task_id, task_data in self.task_db.items():
                if task_data["status"] == TASK_STATES["PENDING"]:
                    current_priority = task_data["priority"]
                    current_created_at = task_data["created_at"]

                    if current_priority > max_priority or (
                        current_priority == max_priority
                        and current_created_at < oldest_creation
                    ):
                        next_task_id = task_id
                        max_priority = current_priority
                        oldest_creation = current_created_at

            return next_task_id

    def close(self) -> None:
        """Close the database connection."""
        self.task_db.close()


if __name__ == "__main__":
    import multiprocessing
    import threading
    import time
    import random

    def worker_process(scheduler, worker_id):
        """Worker process that processes tasks"""
        while True:
            task_id = scheduler.get_next_task_with_max_priority()
            if task_id is None:
                break
            
            # Process the task
            scheduler.change_task_status(task_id, "PROCESSING")
            print(f"Worker {worker_id} processing task {task_id}")
            time.sleep(random.uniform(0.5, 1.5))  # Simulate task processing
            scheduler.change_task_status(task_id, "COMPLETED")
            print(f"Worker {worker_id} completed task {task_id}")

    def worker_thread(scheduler, worker_id):
        """Worker thread that processes tasks"""
        while True:
            task_id = scheduler.get_next_task_with_max_priority()
            if task_id is None:
                break
            
            # Process the task
            scheduler.change_task_status(task_id, "PROCESSING")
            print(f"Thread {worker_id} processing task {task_id}")
            time.sleep(random.uniform(0.5, 1.5))  # Simulate task processing
            scheduler.change_task_status(task_id, "COMPLETED")
            print(f"Thread {worker_id} completed task {task_id}")

    # Create scheduler instance
    scheduler = TaskScheduler()

    # Add sample tasks
    for i in range(10):
        task = PredictTask(seq=f"SEQ{i}", task_type="plddt", priority=random.randint(1, 5))
        scheduler.add_task(task)
        print(f"Added task {task.id} with priority {task.priority}")

    # Multiprocess example
    print("\nRunning multiprocess example:")
    processes = []
    for i in range(3):  # Create 3 worker processes
        p = multiprocessing.Process(target=worker_process, args=(scheduler, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Add more tasks for thread example
    for i in range(10, 20):
        task = PredictTask(seq=f"SEQ{i}", task_type="pdb", priority=random.randint(1, 5))
        scheduler.add_task(task)
        print(f"Added task {task.id} with priority {task.priority}")

    # Multithread example
    print("\nRunning multithread example:")
    threads = []
    for i in range(3):  # Create 3 worker threads
        t = threading.Thread(target=worker_thread, args=(scheduler, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Close the scheduler
    scheduler.close()
