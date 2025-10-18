# 🧪 LLM Prompt Reliability & Invariance Testing Harness

A lightweight, production-ready evaluation framework for testing Large Language Model (LLM) consistency and robustness under prompt transformations and perturbations.

## 🎯 Purpose

This harness evaluates whether LLMs produce consistent, reliable answers when subjected to:
- **Invariance Transformations**: Modifications that preserve semantic meaning (e.g., case changes, whitespace, rephrasing)
- **Perturbation Testing**: Challenging scenarios with distractors, misleading context, and format interference

**Key Problem Solved**: Ensures your LLM-powered applications behave reliably when users phrase questions differently or when prompts contain noise.

## ✨ Key Features

- **🔄 10 Invariance Transformations**: Test robustness to case, whitespace, punctuation, politeness, rephrasing, and more
- **⚡ 10 Perturbation Types**: Test handling of distractors, misleading context, position bias, and format interference
- **📊 Advanced Scoring**: Multiple methods (`contains`, `similarity`, `normalized_equal`, `regex`) optimized for verbose LLM responses
- **🔌 Easy Integration**: Simple adapter pattern for any LLM API (OpenAI, Azure, custom endpoints)
- **📈 Detailed Reports**: JSON/CSV output with per-variant analysis and UUID tracking
- **🎯 Production-Ready**: Handles verbose LLM responses correctly with smart scoring algorithms

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd Prompt-Reliability-quick-mini-harness

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in development mode
pip install -e .

# 4. Set up your configuration
cp configs/openai.example.yaml configs/openai.yaml
# Edit configs/openai.yaml with your API key
```

### Environment Variables (Alternative)

```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 🚀 Quick Start

### Run Your First Test

```bash
# Run invariance evaluation on all test prompts
python -m evalharness.cli --config configs/openai.yaml --evaluator invariance

# Run perturbation evaluation
python -m evalharness.cli --config configs/perturbation.yaml --evaluator perturbation

# Use custom test data
python -m evalharness.cli --config configs/openai.yaml --evaluator invariance --data data/test_2_cases.json

# Export results as CSV
python -m evalharness.cli --config configs/openai.yaml --evaluator invariance --format csv
```

### View Results

```bash
# Results are saved to results/ directory with timestamps
cat results/invariance_YYYYMMDD_HHMMSS.json

# Or use jq for pretty viewing (if available)
cat results/invariance_*.json | jq '.summary'
```

## ⚙️ Configuration

### Scoring Methods (IMPORTANT!)

The harness provides multiple scoring methods. **Choose based on your LLM's verbosity**:

#### ✅ **`contains` (Recommended for GPT-4, Claude, etc.)**
```yaml
evaluator:
  scoring_method: contains
```
- **Use when**: LLM provides verbose, explanatory responses
- **How it works**: Checks if expected output appears as substring in actual output
- **Example**: 
  - Expected: `"Hypertext Transfer Protocol"`
  - Actual: `"HTTP stands for Hypertext Transfer Protocol. It is..."`
  - Result: ✅ **Match**

#### 📊 **`similarity` with `overlap` coefficient**
```yaml
evaluator:
  scoring_method: similarity
  similarity_method: overlap  # More lenient than jaccard
  scoring_threshold: 0.7
```
- **Use when**: You need quantitative similarity scores
- **How it works**: Calculates `intersection / min(expected_words, actual_words)`
- **Better than Jaccard**: Doesn't penalize verbose responses

#### 🎯 **`normalized_equal` (Strict matching)**
```yaml
evaluator:
  scoring_method: normalized_equal
```
- **Use when**: Responses should be brief and exact
- **Requires**: Prompts that request concise answers only

### Full Configuration Example

```yaml
# configs/openai.yaml
model:
  type: openai
  name: gpt-4o-mini
  temperature: 0.0
  max_tokens: 100
  timeout: 30

evaluator:
  # Transformations to test
  transformations:
    - case
    - whitespace
    - punctuation
    - politeness
    - rephrasing
    - instruction
    - numerical
    - typos
    - context
    - formality
  
  # Scoring configuration (RECOMMENDED)
  scoring_method: contains
  
  # Alternative: Use similarity with overlap
  # scoring_method: similarity
  # similarity_method: overlap
  # scoring_threshold: 0.7
```

## 🧪 Test Data Format

Create your test cases in JSON format:

```json
[
  {
    "input": "What is the HTTP acronym meaning?",
    "output": "Hypertext Transfer Protocol"
  },
  {
    "input": "What is 2 + 2?",
    "output": "4"
  }
]
```

### For Concise Responses

Add explicit instructions in your prompts:

```json
[
  {
    "input": "What is the HTTP acronym meaning? Answer with the expansion only.",
    "output": "Hypertext Transfer Protocol"
  }
]
```

Example file: `data/test_prompts_concise.json`

## 📊 Understanding Results

### Summary Metrics

```json
{
  "summary": {
    "average_consistency": 0.95,
    "total_variants_tested": 340,
    "total_passed": 323,
    "total_failed": 17
  }
}
```

### Per-Test Results

```json
{
  "test_case_id": "test_0",
  "original_prompt": "What is the capital of Japan?",
  "expected_output": "Tokyo",
  "variants_tested": 34,
  "passed": 34,
  "failed": 0,
  "consistency_score": 1.0,
  "success_rate": 100.0,
  "variant_results": [
    {
      "variant": "what is the capital of japan?",
      "actual_output": "The capital of Japan is Tokyo.",
      "expected_output": "Tokyo",
      "match": true,
      "match_score": 1.0,
      "metadata": {
        "transformation_type": "case"
      }
    }
  ]
}
```

## 🧪 Invariance Transformations

The harness tests 10 types of label-preserving transformations:

| Transformation | Example | Purpose |
|----------------|---------|---------|
| **Case Variation** | `"What is X?"` → `"WHAT IS X?"` | Test case sensitivity |
| **Whitespace** | `"What is X?"` → `"What  is  X?"` | Test whitespace handling |
| **Punctuation** | `"What is X?"` → `"What is X"` | Test punctuation robustness |
| **Politeness** | `"What is X?"` → `"Please, what is X?"` | Test instruction robustness |
| **Rephrasing** | `"What is X?"` → `"Tell me about X"` | Test semantic understanding |
| **Instruction Format** | `"What is X?"` → `"Q: What is X?"` | Test format invariance |
| **Numerical** | `"2 + 2"` → `"two plus two"` | Test number representations |
| **Typos** | `"What is X?"` → `"Waht is X?"` | Test error tolerance |
| **Context** | `"What is X?"` → `"Quick question: What is X?"` | Test context filtering |
| **Formality** | `"What's X?"` → `"What is X?"` | Test formality shifts |

## ⚡ Perturbation Testing

Tests robustness to challenging scenarios:

1. **Order Shuffle**: Shuffle answer options
2. **Distractor Facts**: Add irrelevant true facts
3. **Misleading Prefix**: Add misleading context
4. **Position Bias**: Test answer position effects
5. **Verbose Wrapper**: Wrap in verbose text
6. **Format Interference**: Add unusual formatting
7. **Semantic Distractors**: Add related but irrelevant info
8. **Negation Flip**: Add double negatives
9. **Multi-Part**: Add multiple sub-questions
10. **Length Extreme**: Test extreme length variations

```bash
python -m evalharness.cli --config configs/perturbation.yaml --evaluator perturbation
```

## 📂 Project Structure

```
Prompt-Reliability-quick-mini-harness/
├── README.md                          # This file
├── SCORING_GUIDE.md                   # Comprehensive scoring documentation
├── pyproject.toml                     # Package configuration
├── requirements.txt                   # Dependencies
│
├── configs/                           # Configuration files
│   ├── openai.example.yaml           # Example OpenAI config
│   ├── openai.yaml                   # Your API config (gitignored)
│   ├── perturbation.example.yaml     # Example perturbation config
│   └── perturbation.yaml             # Your config (gitignored)
│
├── data/                              # Test data
│   ├── test_prompts.json             # Main test cases
│   ├── test_prompts_concise.json     # Tests with brevity instructions
│   └── test_2_cases.json             # Small test set for quick runs
│
├── src/evalharness/                   # Main package
│   ├── __init__.py
│   ├── cli.py                        # Command-line interface
│   ├── runner.py                     # Test orchestration
│   ├── scoring.py                    # Scoring algorithms
│   ├── tracking.py                   # UUID tracking
│   │
│   ├── evaluators/                   # Evaluation strategies
│   │   ├── __init__.py
│   │   ├── base.py                   # Base classes
│   │   ├── invariance.py             # Invariance evaluator
│   │   └── perturbation.py           # Perturbation evaluator
│   │
│   └── model_clients/                # LLM adapters
│       ├── __init__.py
│       ├── base.py                   # Base client interface
│       └── openai_client.py          # OpenAI implementation
│
├── results/                           # Test results (gitignored)
│   ├── invariance_*.json
│   ├── perturbation_*.json
│   └── tracking_*.json
│
└── tests/                             # Unit tests
    ├── test_invariance_eval.py
    └── test_scoring.py
```

## 🔧 Advanced Usage

### Custom Test Data

```bash
# Create your own test file
echo '[{"input": "Your prompt", "output": "Expected answer"}]' > my_tests.json

# Run with custom data
python -m evalharness.cli --config configs/openai.yaml --evaluator invariance --data my_tests.json
```

### Adding Custom Transformations

Edit `src/evalharness/evaluators/invariance.py`:

```python
def my_custom_transform(prompt: str) -> List[str]:
    """Your custom invariance transformation."""
    variants = []
    # Add your transformation logic
    variants.append(prompt.replace("foo", "bar"))
    return variants

# Register in TRANSFORMATIONS dict
TRANSFORMATIONS = {
    # ... existing transformations
    "my_custom": TransformationType(
        "my_custom_transform",
        my_custom_transform,
        "Description of what it tests"
    )
}
```

## 📊 Scoring Deep Dive

### Why `contains` Method?

Modern LLMs (GPT-4, Claude, Gemini) are trained to be helpful and provide context. When asked "What is HTTP?", they respond:

> "HTTP stands for **Hypertext Transfer Protocol**. It is an application layer protocol..."

Not just: "Hypertext Transfer Protocol"

**The Problem**: Token overlap scoring (Jaccard) gives this a score of ~0.1 (10%) because:
- Expected: 3 words
- Actual: 30+ words
- Overlap: 3 / 30 = 0.1 ❌

**The Solution**: Use `contains` method which checks if the expected answer appears anywhere in the response:
- Expected substring found: ✅ **Match**

### When to Use Each Method

| Method | Best For | Threshold | Use Case |
|--------|----------|-----------|----------|
| `contains` | Verbose LLMs | N/A | Production systems, GPT-4, Claude |
| `similarity` + `overlap` | Quantitative scoring | 0.7-0.9 | Research, metrics collection |
| `similarity` + `jaccard` | Similar-length responses | 0.3-0.5 | Concise response systems |
| `normalized_equal` | Exact matching | N/A | Strict validation, classification |
| `regex` | Pattern matching | N/A | Flexible format validation |

See **`SCORING_GUIDE.md`** for complete documentation.

## 🐛 Troubleshooting

### Issue: Very Low Scores (~0.1-0.3)

**Cause**: Using Jaccard similarity with verbose LLM responses

**Solution**: 
```yaml
# Change in your config file
scoring_method: contains
```

### Issue: All Tests Failing

**Solutions**:
```bash
# Check API key
echo $OPENAI_API_KEY

# Run with verbose mode
python -m evalharness.cli --config configs/openai.yaml --evaluator invariance --verbose

# Try contains method (edit config)
```

### Issue: "match_score": 0.33 even though answer is correct

**Cause**: Using Jaccard similarity instead of overlap or contains

**Explanation**: 
- Jaccard: 3 matching words / 30 total words = 0.33
- Overlap: 3 matching words / min(3, 30) = 1.0 ✅
- Contains: Substring found = Match ✅

**Solution**: Update your config to use `contains` or `overlap` method

## 📈 Best Practices

### 1. Start with `contains` Method
```yaml
scoring_method: contains
```

### 2. Test on Small Dataset First
```bash
python -m evalharness.cli --config configs/openai.yaml --evaluator invariance --data data/test_2_cases.json
```

### 3. Review Results Regularly
- Check `match_score` in results
- Identify patterns in failures
- Adjust scoring method or prompts accordingly

### 4. For Job/Interview Projects
- Keep this README comprehensive
- Document all configuration options
- Include troubleshooting section
- Provide working examples
- Show before/after comparisons in scoring

## 📝 Documentation Files

- **`README.md`** (this file): Complete project documentation
- **`SCORING_GUIDE.md`**: Detailed scoring methods explanation
- **`PERTURBATION_GUIDE.md`**: Perturbation types reference
- **`HOW_TO_RUN.md`**: Step-by-step execution guide
- **`PROJECT_SUMMARY.md`**: Architecture overview

## 🤝 Contributing

This project welcomes contributions:

- 🐛 Bug reports and fixes
- ✨ New transformation types
- 🔌 Additional model adapters (Claude, Gemini, etc.)
- 📊 Improved scoring algorithms
- 📖 Documentation improvements

## 📄 License

MIT License - Free for research and commercial use.

## 🎯 Use Cases

- **Production LLM Testing**: Ensure reliability before deployment
- **Model Comparison**: Compare robustness across different LLMs
- **Prompt Engineering**: Validate prompt designs
- **Research**: Study LLM behavior under transformations
- **Quality Assurance**: Automated testing in CI/CD pipelines
- **Job Interviews**: Demonstrate LLM evaluation expertise

## 📞 Support

For questions:
1. Check `SCORING_GUIDE.md` for scoring issues
2. Review troubleshooting section above
3. Examine example configurations in `configs/`
4. Check test results in `results/` directory

---

**Built for reliable LLM evaluation in production environments.**

**Ready for job interviews and production deployment.**
