#!/usr/bin/env python3
"""
Comprehensive research test runner for special character attacks.
Tests ALL attack types against ALL research models specified.
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import AttackTestFramework
from src.evaluation.evaluator import ModelEvaluator


def setup_logging():
    """Set up comprehensive logging"""
    log_filename = f"comprehensive_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE RESEARCH TEST SUITE")
    logger.info("Testing ALL attack types against ALL research models")
    logger.info("=" * 80)
    
    # Research models as specified
    research_models = [
        "phi3:3.8b",
        "mistral:7b", 
        "deepseek-r1:7b",
        "deepseek-r1:8b",
        "deepseek-r1:14b", 
        "deepseek-r1:32b",
        "gpt-oss:20b"
    ]
    
    # All attack types
    attack_types = ["unicode", "homoglyph", "structural", "encoding"]
    
    logger.info(f"Research Models: {research_models}")
    logger.info(f"Attack Types: {attack_types}")
    
    # Initialize framework with specific configuration
    try:
        framework = AttackTestFramework(output_dir="./results")
        logger.info("Framework initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize framework: {e}")
        return
    
    # Check model availability first
    logger.info("Checking model availability...")
    try:
        available_models = framework.model_manager.get_available_models()
        available_names = [m.name for m in available_models]
        logger.info(f"Available models: {available_names}")
        
        # Check which research models are missing
        missing_models = [m for m in research_models if m not in available_names]
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.info("Consider pulling missing models with: ollama pull <model_name>")
            
        # Filter to only available research models
        available_research_models = [m for m in research_models if m in available_names]
        if not available_research_models:
            logger.error("No research models available! Please pull the required models.")
            return
            
        logger.info(f"Will test with available models: {available_research_models}")
        
    except Exception as e:
        logger.error(f"Failed to check model availability: {e}")
        return
    
    # Step 1: Run baseline tests
    logger.info("-" * 60)
    logger.info("PHASE 1: BASELINE BEHAVIOR TESTING")
    logger.info("-" * 60)
    
    try:
        baseline_results = framework.test_baseline_behavior(models=available_research_models)
        logger.info("✓ Baseline testing completed")
    except Exception as e:
        logger.error(f"Baseline testing failed: {e}")
        return
    
    # Step 2: Run all attack types systematically
    logger.info("-" * 60)
    logger.info("PHASE 2: ATTACK TESTING")
    logger.info("-" * 60)
    
    all_attack_results = {}
    
    for i, attack_type in enumerate(attack_types, 1):
        logger.info(f"[{i}/{len(attack_types)}] Testing {attack_type.upper()} attacks...")
        
        try:
            attack_results = framework.test_attack_variants(
                attack_type=attack_type,
                models=available_research_models,
                max_variants_per_attack=5  # Reasonable number for comprehensive testing
            )
            
            all_attack_results[attack_type] = attack_results
            logger.info(f"✓ {attack_type.capitalize()} attack testing completed")
            
            # Small delay between attack types
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"✗ {attack_type.capitalize()} attack testing failed: {e}")
            continue
    
    # Step 3: Comprehensive analysis
    logger.info("-" * 60)
    logger.info("PHASE 3: COMPREHENSIVE ANALYSIS")
    logger.info("-" * 60)
    
    try:
        # Initialize evaluator for advanced analysis
        evaluator = ModelEvaluator(output_dir="./results")
        logger.info("Advanced analysis will be available in the results directory")
        
        # Generate final report
        logger.info("Generating comprehensive report...")
        report = framework.generate_report()
        logger.info("✓ Report generation completed")
        
    except Exception as e:
        logger.error(f"Analysis phase failed: {e}")
    
    # Summary
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TEST SUITE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Models tested: {len(available_research_models)}")
    logger.info(f"Attack types tested: {len(all_attack_results)}")
    logger.info(f"Results available in: ./results/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()