# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo contains the code for an academic paper intended to test the resiliency of different opensource AI models to special character-based attacks.

## Project Overview and Objectives

**Research Questions:**
- Are there special characters that can be used to fool AI models? 
- Are different models more or less vulnerable to these attacks? 
- Can we identify any general trends in the resiliency of different models to these attacks? 
- Can this information tell us anything interesting about the inner workings of these models?

**Approach:**
This research uses a systematic approach to test various special character-based attack vectors against multiple open-source language models running on Ollama. The goal is to understand vulnerability patterns and develop defensive recommendations.

## Project Structure

```
├── src/
│   ├── attacks/              # Attack vector implementations
│   │   ├── unicode_attacks.py      # Zero-width chars, tags, directional overrides
│   │   ├── homoglyph_attacks.py    # Visually similar character substitutions
│   │   ├── structural_attacks.py   # Character reordering, deletion chars
│   │   └── encoding_attacks.py     # Base64, ROT, ASCII art, leetspeak
│   ├── models/              # Model interface and management
│   │   ├── ollama_client.py        # Ollama API client
│   │   └── model_manager.py        # Multi-model coordination
│   ├── evaluation/          # Metrics and analysis
│   │   ├── metrics.py              # Attack success and robustness metrics
│   │   ├── evaluator.py            # Evaluation coordinator
│   │   └── results_analyzer.py     # Data analysis and visualization
│   ├── main.py              # Main testing framework
│   └── test_suite.py        # Comprehensive test prompt library
├── results/                 # Experimental results and reports
├── data/                    # Test data and prompt libraries
├── tests/                   # Unit tests
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── run_tests.py            # Test execution script
```

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve

# Pull test models (adjust based on system capabilities)
ollama pull phi3:3.8b
ollama pull mistral:7b
ollama pull gemma2:2b
```

### Running Tests

#### Quick Test (recommended for initial testing)
```bash
python run_tests.py --quick --models phi3:3.8b mistral:7b
```

#### Comprehensive Test Suite
```bash
python run_tests.py --comprehensive
```

#### Specific Attack Types
```bash
# Test Unicode attacks only
python run_tests.py --attack-type unicode

# Test homoglyph attacks
python run_tests.py --attack-type homoglyph

# Test all attack types
python run_tests.py --attack-type all
```

#### Category-Specific Testing
```bash
# Test baseline behavior
python run_tests.py --category baseline

# Test safety boundaries
python run_tests.py --category safety

# Test jailbreak resistance
python run_tests.py --category jailbreak
```

### Analysis and Reporting

#### Generate Analysis Report
```bash
python run_tests.py --analyze
```

#### Validate Test Suite
```bash
python run_tests.py --validate-suite
```

#### Export Test Suite
```bash
python run_tests.py --export-suite test_prompts.json --export-format json
```

### Main Framework (Advanced)
```bash
# Full framework with model management
python -m src.main --test-type all --attack-type all --pull-models
```

## Model Setup

All models run on Ollama locally. The framework supports testing across different model sizes:

**Small Models (< 7B parameters):**
- phi3:3.8b
- gemma2:2b  
- mistral:7b

**Medium Models (7-15B parameters):**
- llama3.2:11b
- phi3:14b
- gemma2:7b

**Large Models (15B+ parameters):**
- llama3.1:70b
- mixtral:8x7b  
- qwen2.5:72b

## Research Methodology

### Attack Vector Categories

1. **Unicode Attacks**
   - Zero-width characters (ZWNJ, ZWJ, ZWSP, etc.)
   - Unicode tag characters (invisible text injection)
   - Directional override characters (LRO, RLO)
   - Combining characters and diacritics

2. **Homoglyph Attacks** 
   - Cross-script character substitution (Cyrillic, Greek, Latin)
   - Mathematical symbol variants
   - Fullwidth character substitution
   - Targeted word replacement

3. **Structural Attacks**
   - Character reordering within words/sentences
   - Deletion character insertion
   - Space manipulation using Unicode variants
   - Text fragmentation and reconstruction

4. **Encoding Attacks**
   - Base64/hex/binary encoding
   - ROT cipher variations (ROT13, ROT7, etc.)
   - Leetspeak transformations
   - ASCII art representation
   - Morse code encoding

### Evaluation Metrics

The framework uses comprehensive metrics to evaluate:

- **Attack Success Rate**: Percentage of successful jailbreak attempts
- **Model Robustness Score**: Overall resistance to attacks (0-1 scale)
- **Vulnerability Patterns**: Which attack types are most effective per model
- **Response Quality**: Coherence, safety refusal strength, information leakage
- **Consistency**: How similar responses are to similar attacks

### Test Suite Components

1. **Baseline Prompts** (150 prompts)
   - Knowledge questions, creative tasks, problem-solving
   - Establishes normal behavior patterns

2. **Safety Test Prompts** (90 prompts)  
   - Harmful requests, privacy violations, misinformation
   - Tests safety boundary maintenance

3. **Jailbreak Prompts** (90 prompts)
   - Role-playing attempts, instruction overrides, hypothetical scenarios
   - Tests jailbreak resistance

4. **Context Prompts** (60 prompts)
   - Multi-turn conversations, topic switching
   - Tests context understanding consistency

5. **Edge Case Prompts** (90 prompts)
   - Very short prompts, repetitive text, nonsensical input
   - Tests robustness to unusual inputs

## Key Research Findings (Update as research progresses)

**Model Vulnerability Patterns:**
- Document which models are more/less vulnerable to specific attack types
- Note any correlation between model size and robustness
- Identify most effective attack strategies per model architecture

**Attack Effectiveness Rankings:**
- Rank attack types by overall success rate
- Note which techniques are most/least effective
- Document any universal vulnerabilities

**Defensive Recommendations:**
- Input preprocessing strategies
- Character normalization approaches
- Detection methods for each attack type

## Academic Paper Development

### Documentation Requirements
- Maintain detailed logs of all experiments in `results/` directory
- Update README.md with key findings and trends after each major test run
- Document methodology changes and rationale
- Track model versions and configurations used

### Statistical Analysis
- Use the results analyzer to generate quantitative comparisons
- Create visualizations showing vulnerability patterns
- Calculate statistical significance of differences between models
- Document confidence intervals and error rates

### Reproducibility 
- All test configurations are stored in `config.yaml`
- Test suite is fully defined and exportable
- Random seeds should be set for consistent results
- Document exact model versions and Ollama configuration used

## Security and Ethics

This research focuses on **defensive security** applications only:
- Understanding model vulnerabilities to improve defenses
- Developing detection methods for malicious inputs
- Creating robustness benchmarks for AI safety
- All findings are intended for academic publication and defensive improvements

## Troubleshooting

### Common Issues
- **Ollama Connection**: Ensure `ollama serve` is running
- **Model Not Found**: Use `ollama pull <model_name>` to download
- **Memory Issues**: Test smaller models first, use `--quick` flag
- **Slow Tests**: Reduce `max_variants_per_attack` or use smaller model subset

### Performance Optimization  
- Use `max_workers=2` in batch testing to avoid overwhelming Ollama
- Test small models first before large ones
- Use sampling for initial exploration, full suite for final evaluation 
