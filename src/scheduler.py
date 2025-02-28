from sqlitedict import SqliteDict
from typing import Optional, Dict, Any, Literal
from datetime import datetime, timedelta
from task import PredictTask  # Import the PredictTask class

TASK_STATES = {"PENDING": 0, "PROCESSING": 1, "COMPLETED": 2, "FAILED": 3}


class TaskScheduler:

    def __init__(self, db_path="task_scheduler.db"):
        self.db_path = db_path
        self.task_db = SqliteDict(db_path, tablename="tasks", autocommit=True)

    def add_task(self, task: PredictTask) -> None:
        """Add a new task to the scheduler."""
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
        if task_id in self.task_db:
            task_data = self.task_db[task_id]
            task_data["status"] = TASK_STATES[status]
            self.task_db[task_id] = task_data

    def expedite_task(self, task_id: str) -> None:
        """Expedite a task by increasing its priority number by 1."""
        if task_id in self.task_db:
            task_data = self.task_db[task_id]
            task_data["priority"] += 1
            self.task_db[task_id] = task_data

    def reschedule_task(self, task_id: str, new_time: datetime) -> None:
        """Reschedule a task to a new time."""
        if task_id in self.task_db:
            task_data = self.task_db[task_id]
            task_data["scheduled_at"] = new_time.isoformat()
            self.task_db[task_id] = task_data

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a task."""
        if task_id in self.task_db:
            task_data = self.task_db[task_id]
            return task_data["result"]
        return None

    def get_task(self, task_id: str) -> Optional[PredictTask]:
        """Retrieve the PredictTask object for a given task ID."""
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
    scheduler = TaskScheduler()

    # Create and add tasks with different priorities
    task1 = PredictTask(seq="ATGCGTACGT", task_type="plddt", priority=2)
    task2 = PredictTask(seq="CGTAATGCG", task_type="pdb", priority=1)
    task3 = PredictTask(seq="TGCGATACG", task_type="tmscore", priority=3)

    scheduler.add_task(task1)
    scheduler.add_task(task2)
    scheduler.add_task(task3)

    # Get the next task with the highest priority
    next_task = scheduler.get_next_task_with_max_priority()
    if next_task:
        print(f"Next task with max priority: {next_task}")
        task_id = next_task["task"]["id"]
        scheduler.change_task_status(task_id, "PROCESSING")
        print(f"Task '{task_id}' status changed to PROCESSING.")
    else:
        print("No pending tasks found.")

    # Close the scheduler
    scheduler.close()
