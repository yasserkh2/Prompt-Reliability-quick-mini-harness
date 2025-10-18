"""
Command-line interface for the evaluation harness.

Provides a simple CLI to run evaluations with different configurations.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from .runner import Runner

# Load environment variables from .env file
load_dotenv()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Prompt Reliability & Invariance Testing Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run invariance evaluation with default settings
  python -m evalharness.cli --config configs/openai.yaml --evaluator invariance
  
  # Run with custom test data
  python -m evalharness.cli --config configs/openai.yaml --data my_tests.json
  
  # Output as CSV
  python -m evalharness.cli --config configs/openai.yaml --format csv
  
  # Specify output directory
  python -m evalharness.cli --config configs/openai.yaml --output my_results/
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/openai.yaml)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="data/test_prompts.json",
        help="Path to test data JSON file (default: data/test_prompts.json)"
    )
    
    parser.add_argument(
        "--evaluator",
        "-e",
        type=str,
        choices=["invariance", "perturbation"],
        default="invariance",
        help="Type of evaluator to use (default: invariance)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )
    
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Test data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Initialize runner
        if args.verbose:
            print(f"Configuration: {args.config}")
            print(f"Test data: {args.data}")
            print(f"Evaluator: {args.evaluator}")
            print(f"Output: {args.output}")
            print()
        
        runner = Runner(
            config_path=str(config_path),
            data_path=str(data_path),
            output_dir=args.output,
            evaluator_type=args.evaluator
        )
        
        # Run evaluation and save results
        output_path = runner.run_and_save(output_format=args.format)
        
        print(f"\n✓ Evaluation completed successfully!")
        print(f"Results saved to: {output_path}")
        
        sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.", file=sys.stderr)
        sys.exit(130)
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
