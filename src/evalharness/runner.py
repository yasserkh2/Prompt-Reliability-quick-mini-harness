"""
Test runner orchestration.

Loads test data, configurations, runs evaluations, and writes results.
"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import yaml

from .evaluators.base import TestCase, EvaluationResult
from .evaluators.invariance import InvarianceEvaluator
from .evaluators.perturbation import PerturbationEvaluator
from .model_clients.base import ModelClient
from .model_clients.openai_client import OpenAIClient
from .tracking import TestRunTracker


class Runner:
    """
    Orchestrates evaluation runs.
    
    Handles:
    - Loading test data
    - Loading configuration
    - Initializing model client
    - Running evaluator
    - Writing results
    """
    
    def __init__(
        self,
        config_path: str,
        data_path: str,
        output_dir: str = "results",
        evaluator_type: str = "invariance"
    ):
        """
        Initialize the runner.
        
        Args:
            config_path: Path to YAML config file
            data_path: Path to test data JSON file
            output_dir: Directory to write results
            evaluator_type: Type of evaluator ("invariance" or "perturbation")
        """
        self.config_path = Path(config_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.evaluator_type = evaluator_type
        
        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize UUID tracker
        self.tracker = TestRunTracker()
        self.run_id = None
        
        # Load configuration and data
        self.config = self._load_config()
        self.test_cases = self._load_test_data()
        
        # Initialize components
        self.model_client = self._init_model_client()
        self.evaluator = self._init_evaluator()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _load_test_data(self) -> List[TestCase]:
        """Load test cases from JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Test data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to TestCase objects
        test_cases = []
        if isinstance(data, list):
            # Simple list format: [{"input": "...", "output": "..."}, ...]
            for idx, item in enumerate(data):
                test_cases.append(TestCase(
                    id=f"test_{idx}",
                    input=item["input"],
                    output=item["output"],
                    metadata=item.get("metadata", {})
                ))
        elif isinstance(data, dict) and "test_cases" in data:
            # Structured format: {"test_cases": [...], "metadata": {...}}
            for item in data["test_cases"]:
                test_cases.append(TestCase(
                    id=item.get("id", f"test_{len(test_cases)}"),
                    input=item["input"],
                    output=item["output"],
                    metadata=item.get("metadata", {})
                ))
        else:
            raise ValueError("Invalid test data format")
        
        return test_cases
    
    def _init_model_client(self) -> ModelClient:
        """Initialize the model client from configuration."""
        client_config = self.config.get("model", {})
        client_type = client_config.get("type", "openai")
        
        if client_type == "openai":
            return OpenAIClient(
                model_name=client_config.get("name", "gpt-3.5-turbo"),
                api_key=client_config.get("api_key"),
                api_base=client_config.get("api_base"),
                temperature=client_config.get("temperature", 0.0),
                max_tokens=client_config.get("max_tokens", 100),
                timeout=client_config.get("timeout", 30)
            )
        else:
            raise ValueError(f"Unknown client type: {client_type}")
    
    def _init_evaluator(self):
        """Initialize the evaluator from configuration."""
        eval_config = self.config.get("evaluator", {})
        
        if self.evaluator_type == "invariance":
            return InvarianceEvaluator(
                transformations=eval_config.get("transformations"),
                scoring_method=eval_config.get("scoring_method", "normalized_equal"),
                scoring_threshold=eval_config.get("scoring_threshold", 0.8),
                similarity_method=eval_config.get("similarity_method", "overlap")
            )
        elif self.evaluator_type == "perturbation":
            return PerturbationEvaluator(
                perturbation_types=eval_config.get("perturbation_types"),
                scoring_method=eval_config.get("scoring_method", "normalized_equal"),
                scoring_threshold=eval_config.get("scoring_threshold", 0.8),
                similarity_method=eval_config.get("similarity_method", "overlap")
            )
        else:
            raise ValueError(f"Unknown evaluator type: {self.evaluator_type}")
    
    def run(self) -> List[EvaluationResult]:
        """
        Run the evaluation with UUID tracking.
        
        Returns:
            List of evaluation results
        """
        # Start tracking the test run
        self.run_id = self.tracker.start_run(
            evaluator_type=self.evaluator_type,
            model_name=self.model_client.model_name,
            config=self.config
        )
        
        print(f"Running {self.evaluator_type} evaluation...")
        print(f"Run ID: {self.run_id}")
        print(f"Model: {self.model_client.model_name}")
        print(f"Test cases: {len(self.test_cases)}")
        print()
        
        results = []
        for idx, test_case in enumerate(self.test_cases, 1):
            print(f"[{idx}/{len(self.test_cases)}] Evaluating: {test_case.input[:50]}...")
            
            # Start tracking this execution
            start_time = time.time()
            execution_id = self.tracker.start_execution(self.run_id, test_case.id)
            
            # Run evaluation
            result = self.evaluator.evaluate(test_case, self.model_client)
            results.append(result)
            
            # End tracking with duration
            duration = time.time() - start_time
            self.tracker.end_execution(execution_id, duration)
            
            # Add execution metadata to result
            result.metadata['execution_id'] = execution_id
            result.metadata['run_id'] = self.run_id
            result.metadata['duration_seconds'] = duration
            
            print(f"  → Consistency: {result.consistency_score:.2%} "
                  f"({result.passed}/{result.variants_tested} passed) "
                  f"[{duration:.2f}s]")
        
        # Mark run as completed
        self.tracker.end_run(self.run_id, status="completed", total_test_cases=len(results))
        
        return results
    
    def write_results(self, results: List[EvaluationResult], format: str = "json"):
        """
        Write results to file with UUID tracking.
        
        Args:
            results: Evaluation results
            format: Output format ("json" or "csv")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            output_path = self.output_dir / f"{self.evaluator_type}_{timestamp}.json"
            self._write_json_results(results, output_path)
        elif format == "csv":
            output_path = self.output_dir / f"{self.evaluator_type}_{timestamp}.csv"
            self._write_csv_results(results, output_path)
        else:
            raise ValueError(f"Unknown output format: {format}")
        
        # Save tracking data
        tracking_path = self.output_dir / f"tracking_{timestamp}.json"
        self.tracker.save_tracking_data(str(tracking_path))
        
        print(f"\nResults written to: {output_path}")
        print(f"Tracking data saved to: {tracking_path}")
        return output_path
    
    def _write_json_results(self, results: List[EvaluationResult], output_path: Path):
        """Write results as JSON with UUID tracking."""
        output_data = {
            "metadata": {
                "run_id": self.run_id,  # UUID for this test run
                "timestamp": datetime.now().isoformat(),
                "evaluator_type": self.evaluator_type,
                "model": self.model_client.model_name,
                "config": self.config,
                "total_test_cases": len(results)
            },
            "summary": {
                "average_consistency": sum(r.consistency_score for r in results) / len(results) if results else 0,
                "total_variants_tested": sum(r.variants_tested for r in results),
                "total_passed": sum(r.passed for r in results),
                "total_failed": sum(r.failed for r in results),
            },
            "results": [
                {
                    "test_case_id": r.test_case_id,
                    "original_prompt": r.original_prompt,
                    "expected_output": r.expected_output,
                    "variants_tested": r.variants_tested,
                    "passed": r.passed,
                    "failed": r.failed,
                    "consistency_score": r.consistency_score,
                    "success_rate": r.success_rate,
                    "variant_results": [
                        {
                            "variant": vr.variant,
                            "actual_output": vr.actual_output,
                            "expected_output": vr.expected_output,
                            "match": vr.match,
                            "match_score": vr.match_score,
                            "error": vr.error,
                            "metadata": vr.metadata
                        }
                        for vr in r.variant_results
                    ],
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def _write_csv_results(self, results: List[EvaluationResult], output_path: Path):
        """Write results as CSV (flattened variant results)."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "test_case_id",
                "original_prompt",
                "expected_output",
                "variant",
                "actual_output",
                "match",
                "match_score",
                "transformation_type"
            ])
            
            # Data rows
            for result in results:
                for vr in result.variant_results:
                    writer.writerow([
                        result.test_case_id,
                        result.original_prompt,
                        result.expected_output,
                        vr.variant,
                        vr.actual_output,
                        vr.match,
                        vr.match_score,
                        vr.metadata.get("transformation_type", "")
                    ])
    
    def run_and_save(self, output_format: str = "json") -> Path:
        """
        Run evaluation and save results.
        
        Args:
            output_format: Output format ("json" or "csv")
            
        Returns:
            Path to output file
        """
        results = self.run()
        output_path = self.write_results(results, format=output_format)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        avg_consistency = sum(r.consistency_score for r in results) / len(results)
        print(f"Average Consistency Score: {avg_consistency:.2%}")
        print(f"Total Variants Tested: {sum(r.variants_tested for r in results)}")
        print(f"Total Passed: {sum(r.passed for r in results)}")
        print(f"Total Failed: {sum(r.failed for r in results)}")
        print("="*60)
        
        return output_path
