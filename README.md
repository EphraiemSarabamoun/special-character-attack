# Special Character Model Fooling Research

A comprehensive framework for evaluating the vulnerability of large language models to special character-based attacks. This research systematically tests various Unicode, homoglyph, structural, and encoding-based attack vectors across multiple open-source models to understand robustness patterns and develop defensive recommendations.

## 🎯 Research Objectives

This study addresses four key questions about AI model security:

1. **Attack Effectiveness**: What types of special character manipulations can bypass AI safety measures?
2. **Model Vulnerability Patterns**: How do different model architectures respond to character-based attacks?
3. **Attack Vector Ranking**: Which techniques are most effective across model types?
4. **Defensive Implications**: What strategies can improve model robustness?

## 🏗️ Project Architecture

```
├── src/
│   ├── attacks/              # Attack vector implementations
│   │   ├── unicode_attacks.py      # Zero-width, tags, directional overrides
│   │   ├── homoglyph_attacks.py    # Cross-script character substitutions
│   │   ├── structural_attacks.py   # Reordering, deletion, space manipulation
│   │   └── encoding_attacks.py     # Base64, ROT, ASCII art, leetspeak
│   ├── models/              # Model interface and management
│   │   ├── ollama_client.py        # Ollama API integration
│   │   └── model_manager.py        # Multi-model coordination
│   ├── evaluation/          # Metrics and analysis framework
│   │   ├── metrics.py              # Attack success and robustness scoring
│   │   ├── evaluator.py            # Evaluation coordination
│   │   └── results_analyzer.py     # Statistical analysis and visualization
│   ├── main.py              # Primary testing framework
│   └── test_suite.py        # Comprehensive prompt library (480+ prompts)
├── results/                 # Experimental results and generated reports
├── data/                   # Test datasets and prompt collections
├── docs/                   # Detailed methodology and technical documentation
├── tests/                  # Unit tests and validation scripts
├── config.yaml            # Model and testing configuration
├── requirements.txt       # Python dependencies
├── run_tests.py          # User-friendly test runner
├── GETTING_STARTED.md    # Quick start guide
└── CLAUDE.md             # Development documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/download) installed and running
- 8GB+ RAM (16GB+ recommended)

### Installation & First Test
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (in separate terminal)
ollama serve

# 3. Download test models
ollama pull phi3:3.8b
ollama pull gemma2:2b

# 4. Run quick test (5-10 minutes)
python run_tests.py --quick --models phi3:3.8b gemma2:2b
```

## 🧪 Attack Vector Categories

### 1. Unicode Attacks
- **Zero-width characters** (ZWNJ, ZWJ, ZWSP) - Invisible text injection
- **Unicode tag characters** - Steganographic payload hiding  
- **Directional overrides** (LRO, RLO) - Text direction manipulation
- **Combining characters** - Diacritic obfuscation

### 2. Homoglyph Attacks
- **Cross-script substitution** - Latin→Cyrillic→Greek character swaps
- **Mathematical variants** - Unicode math symbol alternatives
- **Fullwidth forms** - Asian typography confusion
- **Targeted replacements** - High-value word substitution

### 3. Structural Attacks
- **Character reordering** - Word/sentence/syllable permutation
- **Deletion characters** - Parser confusion via control chars
- **Space manipulation** - Unicode space variant substitution
- **Text fragmentation** - Breaking and reconstructing content

### 4. Encoding Attacks
- **Base64/Hex encoding** - Binary representation obfuscation
- **Rotation ciphers** - ROT13/ROT7 and variants
- **ASCII art** - Visual character representation
- **Leetspeak** - Character-to-symbol substitution

## 📊 Evaluation Framework

### Metrics
- **Attack Success Rate** - Percentage of successful jailbreak attempts
- **Model Robustness Score** - Overall resistance rating (0-1 scale)
- **Vulnerability Patterns** - Attack-type effectiveness per model
- **Response Quality** - Coherence, safety, consistency analysis

### Test Suite (480+ prompts)
- **Baseline Prompts** (150) - Normal behavior establishment
- **Safety Tests** (90) - Boundary maintenance verification
- **Jailbreak Attempts** (90) - Security bypass resistance
- **Context Tests** (60) - Conversation flow robustness
- **Edge Cases** (90) - Unusual input handling

## 🔧 Usage Examples

### Basic Testing
```bash
# Quick evaluation
python run_tests.py --quick

# Comprehensive test suite
python run_tests.py --comprehensive

# Specific attack type
python run_tests.py --attack-type unicode --models phi3:3.8b
```

### Analysis & Reporting
```bash
# Generate analysis report with visualizations
python run_tests.py --analyze

# Validate test suite quality
python run_tests.py --validate-suite

# Export data for external analysis
python run_tests.py --export-suite results.json
```

### Advanced Framework
```bash
# Full framework with automatic model management
python -m src.main --test-type all --attack-type all --pull-models
```

## 📈 Expected Research Outcomes

### Vulnerability Analysis
- **Model Rankings** - Robustness comparison across architectures
- **Attack Effectiveness** - Success rate analysis by technique
- **Pattern Recognition** - Correlation between model properties and vulnerabilities
- **Defensive Strategies** - Evidence-based security recommendations

### Academic Contributions
- **Methodology Framework** - Reproducible evaluation approach
- **Benchmark Dataset** - Standardized test suite for community use
- **Security Metrics** - Quantitative robustness measurement tools
- **Defensive Guidelines** - Practical implementation recommendations

## 🛡️ Security & Ethics

This research is conducted with **defensive security** objectives:

- ✅ Understanding vulnerabilities to improve AI safety
- ✅ Developing detection methods for malicious inputs
- ✅ Creating robustness benchmarks and evaluation tools
- ✅ Academic publication and community benefit
- ❌ No malicious exploitation or weaponization

All findings are intended for academic research and defensive improvements to enhance AI model security.

## 📋 Model Coverage

**Currently Supported Models:**
- **Small** (< 7B): Phi-3 (3.8B), Gemma 2B, Mistral 7B
- **Medium** (7-15B): Llama 3.2 (11B), Phi-3 (14B), Gemma 7B  
- **Large** (15B+): Llama 3.1 (70B), Mixtral (8x7B), Qwen 2.5 (72B)

*All models run locally via Ollama for privacy and reproducibility.*

## 📚 Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Quick setup and first test
- **[CLAUDE.md](CLAUDE.md)** - Development guide and command reference
- **[docs/methodology.md](docs/methodology.md)** - Detailed research methodology
- **[config.yaml](config.yaml)** - Configuration options and model settings

## 🤝 Contributing

This is an academic research project focused on AI safety. Contributions should align with defensive security objectives:

- **Bug fixes** and performance improvements
- **Additional attack vectors** with academic justification
- **Model coverage** expansion for broader analysis
- **Evaluation metrics** refinement and validation
- **Documentation** improvements and clarity

## 📜 License & Citation

This research is intended for academic use and AI safety improvement. If you use this framework in your research, please cite:

```
[Citation format will be updated upon publication]
```

## 🔍 Research Status

**Current Phase**: Implementation Complete ✅
- ✅ Attack vector library (4 categories, 20+ techniques)
- ✅ Model testing framework (9 model support)  
- ✅ Evaluation metrics and analysis tools
- ✅ Comprehensive test suite (480+ prompts)
- ✅ Statistical analysis and visualization
- 🔄 **Next**: Experimental data collection and analysis

**Future Directions**:
- Multi-modal attack extension
- Real-world deployment testing
- Adversarial training evaluation
- Cross-language attack analysis