# ECLIPSE Complete Pipeline Runner

## 🚀 Quick Start

### **Option 1: Quick Demo (15-30 minutes)**
```bash
cd /Users/sahilkapadia/development/ECLIPSE
python3 run_complete_pipeline.py
```

This runs the essential components:
- ✅ Baseline evaluation
- ✅ Evolutionary search  
- ✅ Comprehensive report

### **Option 2: Full Pipeline (2-4 hours)**
```bash
python3 run_complete_pipeline.py --full-pipeline
```

This runs everything including:
- ✅ Baseline evaluation
- ✅ Evolutionary search
- ✅ Multi-run convergence analysis
- ✅ Robustness testing
- ✅ Ablation study
- ✅ Comprehensive report

### **Option 3: Custom Configuration**
```bash
python3 run_complete_pipeline.py --config pipeline_config.json
```

### **Option 4: With LLM Mutation**
```bash
# Set your API key
export OPENAI_API_KEY="your-openai-api-key"

# Run with LLM
python3 run_complete_pipeline.py --use-llm
```

## ⚙️ Configuration Options

| Command Line | Config File | Default | Description |
|-------------|-------------|----------|-------------|
| `--results DIR` | `results_dir` | `results` | Output directory |
| `--seed N` | `base_seed` | `0` | Random seed |
| `--generations N` | `evolution_generations` | `20` | Evolution generations |
| `--population N` | `population_size` | `50` | Population size |
| `--episodes N` | `evolution_episodes` | `50` | Episodes per evaluation |
| `--full-pipeline` | - | `false` | Enable all optional steps |
| `--use-llm` | `use_llm` | `false` | Use LLM mutation |

## 📊 Pipeline Steps

1. **🏁 Baseline Evaluation** - Evaluate 6 baseline mechanisms
2. **🧬 Evolutionary Search** - Find optimal mechanisms via evolution
3. **📊 Convergence Analysis** - Multi-run convergence study (optional)
4. **🛡️ Robustness Testing** - Test under distribution shifts (optional)
5. **🔬 Ablation Study** - Component importance analysis (optional)
6. **🎯 Pareto Analysis** - Multi-objective optimization analysis
7. **📋 Report Generation** - Comprehensive results report

## 📁 Output Structure

After completion, you'll find:

```
results/
├── comprehensive_report_combined.md      # Main results summary
├── comprehensive_report_combined.json    # Detailed results data
├── baselines_results.csv              # Baseline performance
├── evolution_result.json              # Best evolved mechanism
├── best_mechanism_main_evolution.json # Best mechanism JSON
├── convergence_*.json                 # Convergence data
├── convergence_suite_results.json      # Multi-run analysis (if run)
├── robustness_suite_results.json      # Robustness testing (if run)
├── ablation_study_results.json        # Ablation study (if run)
├── pareto_frontier.json              # Pareto analysis
├── checkpoints/                       # Pipeline checkpoints
│   ├── pipeline_checkpoint.json      # Intermediate checkpoints
│   └── pipeline_final.json         # Final summary
└── logs/                            # Detailed execution logs
    └── pipeline_*.log
```

## 🐛 Troubleshooting

### **Common Issues:**

1. **Import Errors:**
   ```bash
   cd /Users/sahilkapadia/development/ECLIPSE
   python3 run_complete_pipeline.py
   ```

2. **Permission Denied:**
   ```bash
   chmod +x run_complete_pipeline.py
   ```

3. **Missing Dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Memory Issues:**
   - Reduce `--population` and `--generations`
   - Reduce `--episodes`

5. **LLM API Issues:**
   ```bash
   export OPENAI_API_KEY="your-key"
   python3 run_complete_pipeline.py --use-llm
   ```

### **Check Progress:**
```bash
# View live logs
tail -f results/logs/pipeline_*.log

# Check latest checkpoint
cat results/checkpoints/pipeline_checkpoint.json
```

## ⏱️ Runtime Estimates

| Pipeline Mode | Runtime | Episodes | Generations |
|---------------|----------|-----------|-------------|
| Quick Demo | 15-30 min | 100 | 15 |
| Standard | 1-2 hours | 200 | 20 |
| Full Pipeline | 2-4 hours | 200 | 20 |
| Full + LLM | 3-6 hours | 200 | 20 |

## 🎯 Success Criteria

The pipeline is successful when:

1. ✅ All baselines evaluated without errors
2. ✅ Evolution converges to feasible solution(s)
3. ✅ Best mechanism outperforms at least 3 baselines
4. ✅ Comprehensive report generated
5. ✅ All artifacts saved to results directory

## 📈 Monitoring Progress

The pipeline provides real-time progress updates:
- 🏁 Step completion indicators
- ✅ Checkpoint saves after each step
- 📊 Performance metrics
- ⚠️ Error handling and recovery
- 📋 Final summary with statistics

## 🔄 Resuming from Checkpoint

If the pipeline is interrupted, you can resume:

```bash
# The pipeline automatically detects and resumes from the last checkpoint
python3 run_complete_pipeline.py --results results
```

The checkpoint system saves progress after each major step, allowing you to resume from any point without re-running completed steps.