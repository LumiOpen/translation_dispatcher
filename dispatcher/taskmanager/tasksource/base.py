from abc import ABC, abstractmethod
from typing import List, Any

from ..task.base import Task

class TaskSource(ABC):
    """
    Source of tasks and destination for results.
    """
    
    @abstractmethod
    def get_next_tasks(self) -> List[Task]:
        """
        Get new tasks to process based on the source's internal batch size.
        Returns an empty list if no tasks are currently available.
        Each task must have at least one request immediately available.
        """
        pass
    
    @abstractmethod
    def save_task_result(self, task: Task) -> None:
        """
        Save the result of a completed task.
        The result and context are retrieved from the task.
        """
        pass
    
    @property
    @abstractmethod
    def is_exhausted(self) -> bool:
        """
        Returns True if the source is permanently exhausted.
        This means no new tasks will ever become available.
        """
        pass
