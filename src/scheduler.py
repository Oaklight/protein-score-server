import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from filelock import FileLock
from sqlitedict import SqliteDict

from task import PredictTask  # Import the PredictTask class

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import cprint

TASK_STATES = {
    "PENDING": 0,
    "PROCESSING": 1,
    "COMPLETED": 2,
    "FAILED": 3,
}


class TaskScheduler:

    def __init__(self, db_path="task_scheduler.db"):
        self.db_path = db_path
        self.task_db = SqliteDict(db_path, tablename="tasks", autocommit=True)
        self.lock = FileLock(f"{db_path}.lock")

    def add_task(self, task: PredictTask) -> None:
        """Add a new task to the scheduler."""
        with self.lock:
            task_data = {
                "task": (
                    task.to_dict() if task is not None else None
                ),  # Store the PredictTask as a dictionary
                "status": TASK_STATES["PENDING"],
                "priority": (
                    task.priority if task is not None else sys.maxsize
                ),  # max priority if None, meaning to stop server
                "created_at": datetime.now().isoformat(),
                "scheduled_at": datetime.now().isoformat(),
                "result": None,
                "worker_id": None,
            }
            if task is not None:
                self.task_db[task.id] = task_data
            else:  # shutting down special case
                self.task_db["shutdown"] = task_data

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

    def reschedule_task(
        self, task_id: str, new_time: datetime, need_expedite=False
    ) -> None:
        """Reschedule a task to a new time."""
        with self.lock:
            if task_id in self.task_db:
                task_data = self.task_db[task_id]
                task_data["status"] = TASK_STATES["PENDING"]
                task_data["worker_id"] = None
                task_data["scheduled_at"] = new_time.isoformat()
                if need_expedite:
                    task_data["priority"] += 1
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
        Atomically marks the task as IN_PROGRESS to prevent other workers from taking it.

        Returns:
            Optional[str]: The ID of the next task with the highest priority,
                          or None if no such task exists.
        """
        with self.lock:
            next_task_id = None
            max_priority = -1
            oldest_creation = None

            # First pass: Find the highest priority task
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

            # Second pass: Atomically assign task to this worker
            if next_task_id is not None:
                task_data = self.task_db[next_task_id]
                if (
                    task_data["status"] == TASK_STATES["PENDING"]
                    and task_data["worker_id"] is None
                ):
                    task_data["status"] = TASK_STATES["PROCESSING"]
                    task_data["worker_id"] = (
                        os.getpid()
                    )  # Use process ID for both multiprocess and multithread
                    self.task_db[next_task_id] = task_data
                else:
                    # Task was taken by another worker, try again
                    next_task_id = None

            return next_task_id

    def close(self) -> None:
        """Close the database connection."""
        self.task_db.close()


if __name__ == "__main__":
    import argparse
    import random
    import signal
    import time
    import threading
    import multiprocessing

    argparser = argparse.ArgumentParser(description="Task Scheduler Example")
    argparser.add_argument(
        "-m",
        "--mode",
        default="multithread",
        help="test which case, multithread or multiprocess",
    )
    args = argparser.parse_args()

    def worker_process(worker_id, db_path, stop_event):
        """Worker process that processes tasks with timeout and failure handling"""
        scheduler = TaskScheduler(db_path)
        try:
            while not stop_event.is_set():
                task_id = scheduler.get_next_task_with_max_priority()
                if task_id is None:
                    time.sleep(0.5)  # Short delay before checking again
                    continue

                # Process the task (task is already assigned to this worker)
                task_data = scheduler.task_db[task_id]
                cprint(
                    f"Worker {worker_id} processing task {task_id} with priority {task_data['priority']} created at {task_data['created_at']}",
                    "cyan",
                    flush=True,
                )

                try:
                    # Simulate task processing with potential timeout/failure
                    process_time = random.uniform(0.5, 2.5)
                    if process_time > 2.0:  # Simulate timeout
                        raise TimeoutError(
                            f"Task {task_id} timed out after {process_time:.2f}s"
                        )
                    if random.random() < 0.2:  # 20% chance of failure
                        raise RuntimeError(f"Task {task_id} failed randomly")

                    time.sleep(process_time)
                    scheduler.change_task_status(task_id, "COMPLETED")
                    cprint(
                        f"Worker {worker_id} completed task {task_id} with priority {task_data['priority']} created at {task_data['created_at']}",
                        "magenta",
                        flush=True,
                    )

                except (TimeoutError, RuntimeError) as e:
                    cprint(
                        f"Worker {worker_id} encountered error: {str(e)}",
                        "red",
                        flush=True,
                    )
                    scheduler.change_task_status(task_id, "FAILED")

                    # Reschedule the task with increased priority
                    new_time = datetime.now()
                    scheduler.reschedule_task(task_id, new_time, need_expedite=True)
                    cprint(
                        f"Rescheduled task {task_id} with new priority {task_data["priority"] + 1} at {new_time}",
                        "yellow",
                        flush=True,
                    )

        finally:
            cprint(f"Worker {worker_id} is shutting down", "yellow", flush=True)
            scheduler.close()

    def worker_thread(scheduler, worker_id, stop_event):
        """Worker thread that processes tasks with timeout and failure handling"""
        try:
            while not stop_event.is_set():
                try:
                    task_id = scheduler.get_next_task_with_max_priority()
                    if task_id is None:
                        time.sleep(0.5)  # Short delay before checking again
                        continue

                    # Process the task (task is already assigned to this worker)
                    task_data = scheduler.task_db[task_id]
                    cprint(
                        f"Thread {worker_id} processing task {task_id} with priority {task_data['priority']} created at {task_data['created_at']}",
                        "cyan",
                        flush=True,
                    )

                    try:
                        # Simulate task processing with potential timeout/failure
                        process_time = random.uniform(0.5, 2.5)
                        if process_time > 2.0:  # Simulate timeout
                            raise TimeoutError(
                                f"Task {task_id} timed out after {process_time:.2f}s"
                            )
                        if random.random() < 0.2:  # 20% chance of failure
                            raise RuntimeError(f"Task {task_id} failed randomly")

                        time.sleep(process_time)
                        scheduler.change_task_status(task_id, "COMPLETED")
                        cprint(
                            f"Thread {worker_id} completed task {task_id} with priority {task_data['priority']} created at {task_data['created_at']}",
                            "magenta",
                            flush=True,
                        )

                    except (TimeoutError, RuntimeError) as e:
                        cprint(
                            f"Thread {worker_id} encountered error: {str(e)}",
                            "red",
                            flush=True,
                        )
                        scheduler.change_task_status(task_id, "FAILED")

                        # Reschedule the task with increased priority
                        new_time = datetime.now()
                        scheduler.reschedule_task(task_id, new_time, need_expedite=True)
                        cprint(
                            f"Rescheduled task {task_id} with new priority {task_data['priority'] + 1} at {new_time}",
                            "yellow",
                            flush=True,
                        )
                except Exception as e:
                    time.sleep(1)  # Prevent tight error loop
        finally:
            cprint(
                f"[{datetime.now()}] Thread {worker_id} is shutting down",
                "yellow",
                flush=True,
            )

    #  ================= STOP mechanism =================
    stop_event_mp = multiprocessing.Event()  # STOP flag for multiprocess
    stop_event_th = threading.Event()  # STOP flag for multithread

    # Signal handler to set the STOP flag on Ctrl+C
    def signal_handler(signum, frame):
        cprint("\nCtrl+C detected. Stopping workers...", "red")
        stop_event_mp.set()
        stop_event_th.set()

    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # ================= MULTIPROCESS =================
    if args.mode == "multiprocess":
        # Multiprocess example
        cprint("\nRunning multiprocess example:", "white")
        db_path = "task_scheduler_mp.db"
        scheduler = TaskScheduler(db_path)

        try:
            # Add initial tasks
            cprint("Adding tasks for multiprocess example:", "white")
            for i in range(10):
                task = PredictTask(
                    seq=f"SEQ{i}", task_type="plddt", priority=random.randint(1, 5)
                )
                scheduler.add_task(task)
                cprint(f"Added task {task.id} with priority {task.priority}", "yellow")

            # Create worker processes
            processes = []
            for i in range(3):  # Create 3 worker processes
                p = multiprocessing.Process(
                    target=worker_process, args=(i, db_path, stop_event_mp)
                )
                processes.append(p)
                p.start()

            # Wait for processes to complete
            for p in processes:
                p.join()
        finally:
            cprint(f"Main worker is shutting down", "yellow")
            scheduler.close()

    # ================= MULTITHREAD =================
    elif args.mode == "multithread":
        # Multithread example
        cprint("\nRunning multithread example:", "white")
        db_path = "task_scheduler_th.db"
        scheduler = TaskScheduler(db_path)

        try:
            # Add tasks
            cprint("Adding tasks for multithread example:", "white")
            for i in range(10, 20):
                task = PredictTask(
                    seq=f"SEQ{i}", task_type="pdb", priority=random.randint(1, 5)
                )
                scheduler.add_task(task)
                cprint(f"Added task {task.id} with priority {task.priority}", "yellow")

            # Create worker threads
            threads = []
            for i in range(3):  # Create 3 worker threads
                t = threading.Thread(
                    target=worker_thread, args=(scheduler, i, stop_event_th)
                )
                threads.append(t)
                t.start()

            # Wait for threads to complete
            for t in threads:
                t.join()
        finally:
            scheduler.close()
