"""
UUID-based tracking system for test runs and individual evaluations.

Provides unique identifiers for:
- Test runs (entire evaluation session)
- Individual test case evaluations
- Variant executions
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class TestRunMetadata:
    """Metadata for a complete test run with UUID tracking."""
    
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluator_type: str = ""
    model_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    total_test_cases: int = 0
    status: str = "running"  # running, completed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "evaluator_type": self.evaluator_type,
            "model_name": self.model_name,
            "config": self.config,
            "total_test_cases": self.total_test_cases,
            "status": self.status
        }


@dataclass
class TestCaseExecution:
    """Tracks a single test case execution with UUID."""
    
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""  # Links to parent test run
    test_case_id: str = ""  # Original test case ID from input
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "test_case_id": self.test_case_id,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds
        }


class TestRunTracker:
    """
    Tracker for test runs with UUID-based identification.
    
    Usage:
        tracker = TestRunTracker()
        run_id = tracker.start_run(evaluator_type="invariance", model_name="gpt-4o-mini")
        
        # For each test case
        exec_id = tracker.start_execution(run_id, test_case_id="test1")
        # ... run test ...
        tracker.end_execution(exec_id, duration=2.5)
        
        tracker.end_run(run_id, status="completed")
        tracker.save_tracking_data("results/tracking.json")
    """
    
    def __init__(self):
        self.runs: Dict[str, TestRunMetadata] = {}
        self.executions: Dict[str, TestCaseExecution] = {}
    
    def start_run(
        self,
        evaluator_type: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new test run and return its UUID.
        
        Args:
            evaluator_type: Type of evaluator (invariance/perturbation)
            model_name: Name of the model being tested
            config: Configuration dictionary
            
        Returns:
            UUID string for the test run
        """
        run_metadata = TestRunMetadata(
            evaluator_type=evaluator_type,
            model_name=model_name,
            config=config or {}
        )
        self.runs[run_metadata.run_id] = run_metadata
        return run_metadata.run_id
    
    def end_run(self, run_id: str, status: str = "completed", total_test_cases: int = 0):
        """
        Mark a test run as completed.
        
        Args:
            run_id: UUID of the test run
            status: Final status (completed/failed)
            total_test_cases: Total number of test cases executed
        """
        if run_id in self.runs:
            self.runs[run_id].status = status
            self.runs[run_id].total_test_cases = total_test_cases
    
    def start_execution(self, run_id: str, test_case_id: str) -> str:
        """
        Start tracking a test case execution.
        
        Args:
            run_id: UUID of the parent test run
            test_case_id: ID of the test case being executed
            
        Returns:
            UUID string for the execution
        """
        execution = TestCaseExecution(
            run_id=run_id,
            test_case_id=test_case_id
        )
        self.executions[execution.execution_id] = execution
        return execution.execution_id
    
    def end_execution(self, execution_id: str, duration: float):
        """
        Mark an execution as completed and record duration.
        
        Args:
            execution_id: UUID of the execution
            duration: Duration in seconds
        """
        if execution_id in self.executions:
            self.executions[execution_id].duration_seconds = duration
    
    def get_run_metadata(self, run_id: str) -> Optional[TestRunMetadata]:
        """Get metadata for a specific run."""
        return self.runs.get(run_id)
    
    def get_execution_metadata(self, execution_id: str) -> Optional[TestCaseExecution]:
        """Get metadata for a specific execution."""
        return self.executions.get(execution_id)
    
    def save_tracking_data(self, output_path: str):
        """
        Save tracking data to JSON file.
        
        Args:
            output_path: Path to save tracking JSON
        """
        tracking_data = {
            "runs": {run_id: run.to_dict() for run_id, run in self.runs.items()},
            "executions": {exec_id: exec.to_dict() for exec_id, exec in self.executions.items()}
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_tracking_data(input_path: str) -> 'TestRunTracker':
        """
        Load tracking data from JSON file.
        
        Args:
            input_path: Path to tracking JSON file
            
        Returns:
            TestRunTracker instance with loaded data
        """
        tracker = TestRunTracker()
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct runs
        for run_id, run_data in data.get("runs", {}).items():
            run = TestRunMetadata(**run_data)
            tracker.runs[run_id] = run
        
        # Reconstruct executions
        for exec_id, exec_data in data.get("executions", {}).items():
            execution = TestCaseExecution(**exec_data)
            tracker.executions[exec_id] = execution
        
        return tracker


def generate_test_id() -> str:
    """Generate a unique test ID using UUID."""
    return str(uuid.uuid4())


def generate_run_id() -> str:
    """Generate a unique run ID using UUID."""
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """
    Generate a short unique ID (first N characters of UUID).
    
    Args:
        length: Number of characters (default 8)
        
    Returns:
        Short ID string
    """
    return str(uuid.uuid4())[:length]
