#!/usr/bin/env python3
"""
Test runner for special character model attack testing.
This script provides a simple interface to run various test configurations.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import AttackTestFramework
from test_suite import TestSuiteGenerator, get_test_prompts
from evaluation.results_analyzer import ResultsAnalyzer


def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_run.log'),
            logging.StreamHandler()
        ]
    )


def run_quick_test(models: list = None, attack_types: list = None):
    """Run a quick test with a small sample of prompts"""
    print("Running quick test suite...")
    
    framework = AttackTestFramework()
    
    # Get sample prompts
    test_suite = TestSuiteGenerator()
    sample_prompts = test_suite.get_sample_test_suite(samples_per_category=3, include_high_risk=False)
    
    print(f"Testing {len(sample_prompts)} sample prompts")
    
    # Run baseline test
    print("Running baseline tests...")
    baseline_results = framework.test_baseline_behavior(models=models)
    
    # Run limited attack tests
    print("Running attack tests...")
    attack_results = framework.test_attack_variants(
        attack_type="unicode" if not attack_types else attack_types[0],
        models=models,
        max_variants_per_attack=2
    )
    
    print("Quick test completed!")
    return baseline_results, attack_results


def run_comprehensive_test(models: list = None):
    """Run comprehensive test suite"""
    print("Running comprehensive test suite...")
    
    framework = AttackTestFramework()
    
    # Run all baseline tests
    print("Running baseline behavior tests...")
    baseline_results = framework.test_baseline_behavior(models=models)
    
    # Run all attack types
    attack_types = ["unicode", "homoglyph", "structural", "encoding"]
    
    for attack_type in attack_types:
        print(f"Running {attack_type} attack tests...")
        attack_results = framework.test_attack_variants(
            attack_type=attack_type,
            models=models,
            max_variants_per_attack=10
        )
    
    print("Comprehensive test completed!")


def run_specific_category_test(category: str, models: list = None):
    """Run tests for a specific category"""
    print(f"Running {category} category tests...")
    
    # Get prompts for specific category
    prompts = get_test_prompts(category=category)
    print(f"Found {len(prompts)} prompts in {category} category")
    
    if not prompts:
        print(f"No prompts found for category: {category}")
        return
    
    framework = AttackTestFramework()
    
    # Convert test prompts to simple strings for testing
    prompt_texts = [p.prompt for p in prompts[:20]]  # Limit to first 20
    
    # Test prompts against models
    results = framework.model_manager.test_prompts_batch(
        prompts=prompt_texts,
        models=models,
        temperature=0.1
    )
    
    print(f"Tested {len(prompt_texts)} prompts from {category} category")
    return results


def validate_test_suite():
    """Validate the test suite for quality and completeness"""
    print("Validating test suite...")
    
    test_suite = TestSuiteGenerator()
    validation_report = test_suite.validate_test_suite()
    
    print(f"Test Suite Validation Report:")
    print(f"- Total prompts: {validation_report['total_prompts']}")
    print(f"- Categories: {list(validation_report['category_distribution'].keys())}")
    print(f"- Risk levels: {list(validation_report['risk_level_distribution'].keys())}")
    print(f"- Balance score: {validation_report['balance_score']:.3f}")
    print(f"- Completeness score: {validation_report['completeness_score']:.3f}")
    print(f"- Overall health: {validation_report['overall_health']}")
    
    if validation_report['quality_issues']:
        print("Quality Issues:")
        for issue in validation_report['quality_issues']:
            print(f"  - {issue}")
    else:
        print("No quality issues found.")


def generate_analysis_report():
    """Generate analysis report from existing results"""
    print("Generating analysis report from existing results...")
    
    analyzer = ResultsAnalyzer()
    
    if not analyzer.results_data:
        print("No results data found. Run some tests first.")
        return
    
    print(f"Found {len(analyzer.results_data)} result files")
    
    # Generate visualizations and report
    report_path = analyzer.export_analysis_report()
    print(f"Analysis report generated: {report_path}")
    
    # Generate statistical summary
    summary = analyzer.generate_statistical_summary()
    print("\nStatistical Summary:")
    print(f"- Models tested: {summary['data_overview']['models_tested']}")
    print(f"- Attack types: {summary['data_overview']['attack_types_tested']}")
    
    if 'model_statistics' in summary and summary['model_statistics']:
        print(f"- Most robust model: {summary['model_statistics']['most_robust_model']}")
        print(f"- Least robust model: {summary['model_statistics']['least_robust_model']}")
    
    if 'attack_statistics' in summary and summary['attack_statistics']:
        print(f"- Most effective attack: {summary['attack_statistics']['most_effective_attack']}")


def export_test_suite(output_path: str, format: str = "json"):
    """Export test suite to file"""
    print(f"Exporting test suite to {output_path} in {format} format...")
    
    test_suite = TestSuiteGenerator()
    test_suite.export_test_suite(output_path, format=format)
    
    print(f"Test suite exported successfully")


def main():
    parser = argparse.ArgumentParser(description="Special Character Attack Testing Framework")
    
    # Test execution options
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with small sample")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test suite (takes longer)")
    parser.add_argument("--category", type=str,
                       help="Test specific category (baseline, safety, jailbreak, context, edge_case)")
    parser.add_argument("--attack-type", choices=["unicode", "homoglyph", "structural", "encoding", "all"],
                       default="all", help="Type of attacks to test")
    
    # Model selection
    parser.add_argument("--models", nargs="+",
                       help="Specific models to test (if not specified, uses all available)")
    parser.add_argument("--pull-models", action="store_true",
                       help="Pull missing models before testing")
    
    # Analysis and reporting
    parser.add_argument("--analyze", action="store_true",
                       help="Generate analysis report from existing results")
    parser.add_argument("--validate-suite", action="store_true",
                       help="Validate test suite quality and completeness")
    
    # Export options
    parser.add_argument("--export-suite", type=str,
                       help="Export test suite to file (specify path)")
    parser.add_argument("--export-format", choices=["json", "yaml", "csv"], default="json",
                       help="Export format for test suite")
    
    # General options
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Handle export request
    if args.export_suite:
        export_test_suite(args.export_suite, args.export_format)
        return
    
    # Handle validation request
    if args.validate_suite:
        validate_test_suite()
        return
    
    # Handle analysis request
    if args.analyze:
        generate_analysis_report()
        return
    
    # Check if Ollama is accessible (basic check)
    try:
        from models.ollama_client import OllamaClient
        client = OllamaClient()
        available_models = client.list_models()
        print(f"Found {len(available_models)} models available in Ollama")
        
        if args.pull_models:
            print("Note: Use --pull-models with the main framework to pull models")
    
    except Exception as e:
        print(f"Warning: Could not connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        if not (args.quick or args.comprehensive or args.category):
            return
    
    # Execute tests based on arguments
    if args.quick:
        attack_types = [args.attack_type] if args.attack_type != "all" else None
        run_quick_test(models=args.models, attack_types=attack_types)
    
    elif args.comprehensive:
        run_comprehensive_test(models=args.models)
    
    elif args.category:
        run_specific_category_test(args.category, models=args.models)
    
    else:
        # Default: run the main framework
        print("Running main framework...")
        framework = AttackTestFramework(config_path=args.config, output_dir=args.output_dir)
        
        if args.pull_models:
            print("Pulling missing models...")
            pull_results = framework.model_manager.pull_missing_models()
            print(f"Pull results: {pull_results}")
        
        # Run baseline and attack tests
        print("Running baseline tests...")
        framework.test_baseline_behavior(models=args.models)
        
        print(f"Running {args.attack_type} attack tests...")
        framework.test_attack_variants(
            attack_type=args.attack_type,
            models=args.models
        )
        
        print("Generating final report...")
        framework.generate_report()
        print("Testing completed!")


if __name__ == "__main__":
    main()