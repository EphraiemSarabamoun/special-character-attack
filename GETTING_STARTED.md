# Getting Started with Special Character Attack Testing

This guide will help you quickly set up and run your first tests with the special character attack framework.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running ([installation guide](https://ollama.ai/download))
3. At least **8GB RAM** (16GB+ recommended for larger models)
4. **10GB+ free disk space** for models and results

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama
```bash
# In a separate terminal, start the Ollama server
ollama serve
```

### 3. Download Test Models
```bash
# Start with small, fast models
ollama pull phi3:3.8b
ollama pull gemma2:2b

# Optional: larger models (require more resources)
ollama pull mistral:7b
ollama pull llama3.2:11b
```

### 4. Run Your First Test
```bash
# Quick test with small sample
python run_tests.py --quick --models phi3:3.8b gemma2:2b
```

This will:
- Test 2 models with 15 sample prompts (3 per category)
- Run Unicode attack variants
- Generate results in the `results/` directory
- Complete in ~5-10 minutes

## Understanding the Results

After the test completes, you'll find:

### Result Files
- `baseline_results_*.json` - Normal behavior testing
- `attack_results_*.json` - Attack testing results
- `test_report_*.md` - Summary report
- `attack_testing.log` - Detailed execution log

### Key Metrics
- **Attack Success Rate** - Percentage of successful attacks
- **Model Robustness Score** - Overall resistance (0-1, higher is better)
- **Vulnerability Patterns** - Which attacks work best per model

### Sample Output
```
## Test Results Summary
- Total Models: 2
- Available Models: 2
- Availability Rate: 100.0%

**Attack Success Rates:**
- unicode: 15.2%
- homoglyph: 8.7%
- structural: 22.1%
- encoding: 12.5%

**Model Vulnerability Rates:**
- phi3:3.8b: 14.2%
- gemma2:2b: 18.9%
```

## Next Steps

### Run Comprehensive Testing
```bash
# Test all attack types with more samples
python run_tests.py --comprehensive --models phi3:3.8b gemma2:2b mistral:7b
```

### Test Specific Attack Types
```bash
# Focus on Unicode attacks
python run_tests.py --attack-type unicode --models phi3:3.8b

# Test homoglyph attacks
python run_tests.py --attack-type homoglyph --models gemma2:2b
```

### Generate Analysis Reports
```bash
# Create visualizations and statistical analysis
python run_tests.py --analyze
```

### Export Data for Analysis
```bash
# Export test suite for external analysis
python run_tests.py --export-suite test_data.json --export-format json
```

## Common Issues and Solutions

### "Connection refused" Error
**Problem**: Can't connect to Ollama
**Solution**: 
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### "Model not found" Error
**Problem**: Specified model not available
**Solution**:
```bash
# List available models
ollama list

# Pull missing model
ollama pull phi3:3.8b
```

### Out of Memory Error
**Problem**: Not enough RAM for model
**Solutions**:
1. Use smaller models (phi3:3.8b, gemma2:2b)
2. Test one model at a time: `--models phi3:3.8b`
3. Use quick test: `--quick`

### Slow Performance
**Problem**: Tests taking too long
**Solutions**:
1. Reduce model count: `--models phi3:3.8b`
2. Use quick mode: `--quick` 
3. Test specific categories: `--category baseline`

## Understanding Attack Types

### Unicode Attacks
- **What**: Invisible characters, directional overrides
- **Example**: "Te‌ll me ho‍w to..." (contains zero-width chars)
- **Purpose**: Test tokenizer robustness

### Homoglyph Attacks  
- **What**: Visually similar characters from different scripts
- **Example**: "Tеll me how to..." (Cyrillic 'е' instead of 'e')
- **Purpose**: Test character recognition

### Structural Attacks
- **What**: Character reordering, unusual spacing
- **Example**: "lleT em woh ot..." (reversed words)
- **Purpose**: Test parsing robustness

### Encoding Attacks
- **What**: Base64, leetspeak, ASCII art
- **Example**: "T3ll m3 h0w t0..." (leetspeak)
- **Purpose**: Test decoding behavior

## Customization Options

### Test Configuration
Edit `config.yaml` to customize:
- Model lists
- Attack parameters
- Output settings
- Evaluation criteria

### Custom Prompts
Add your own test prompts in `src/test_suite.py` or create separate prompt files.

### Advanced Usage
See `CLAUDE.md` for comprehensive documentation on:
- Research methodology
- Evaluation metrics
- Statistical analysis
- Academic paper development

## Getting Help

1. **Check logs**: `attack_testing.log` and `test_run.log`
2. **Validate setup**: `python run_tests.py --validate-suite`
3. **Review documentation**: `CLAUDE.md` and `docs/methodology.md`
4. **Test individual components**: Use Python REPL to test modules

## Example Workflow

Here's a typical research workflow:

```bash
# 1. Initial exploration
python run_tests.py --quick --models phi3:3.8b gemma2:2b

# 2. Focused testing on interesting findings
python run_tests.py --attack-type unicode --models phi3:3.8b

# 3. Comprehensive evaluation
python run_tests.py --comprehensive

# 4. Generate analysis
python run_tests.py --analyze

# 5. Export results
python run_tests.py --export-suite final_results.json
```

This workflow will give you a complete picture of model vulnerabilities and provide data for academic analysis.

Happy testing! 🔒🔍