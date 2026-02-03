# 🧪 Testing LLM Mutation

## Quick Test

To test if LLM mutation is working with your OpenAI API key:

```bash
# Set your API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Run the test
cd /Users/sahilkapadia/development/ECLIPSE
python3 test_llm_mutation.py
```

## What the Test Does

1. **✅ Checks API Key** - Verifies your OpenAI API key is available
2. **✅ Validates Imports** - Tests if all required modules can be imported
3. **✅ Creates Test Mechanisms** - Builds 2 sample healthcare mechanisms
4. **✅ Calls LLM Mutation** - Sends request to OpenAI with failure analysis
5. **✅ Validates Results** - Checks if generated mechanisms are valid
6. **✅ Shows Output** - Displays the new mechanisms created by LLM

## Expected Output

**Success:**
```
🧪 ECLIPSE LLM Mutation Test
==================================================
✅ API key found
✅ Modules imported successfully
✅ Test mechanism 1 valid
✅ Test mechanism 2 valid

🧪 Testing LLM mutation...
   Input mechanisms: 2
   Model to use: gpt-4o-mini (recommended)

✅ LLM mutation SUCCESS! Generated 2 new mechanisms:

   Mechanism 1: ✅ VALID
   Info policy: coarse_bins
   Service rule: hybrid
   Redirect mode: combined
   ID: a1b2c3d4-...

   Mechanism 2: ✅ VALID
   Info policy: exact
   Service rule: severity_priority
   Redirect mode: congestion_cutoff
   ID: e5f6g7h8-...

==================================================
🎉 LLM mutation test PASSED!
✅ Your LLM integration is working correctly
✅ Ready to use with: python3 run_complete_pipeline.py --use-llm
==================================================
```

## Troubleshooting

**If test fails:**

### ❌ "OPENAI_API_KEY not found"
```bash
export OPENAI_API_KEY="sk-your-actual-api-key"
python3 test_llm_mutation.py
```

### ❌ "Import error"
```bash
pip3 install openai
```

### ❌ "LLM mutation FAILED"
Common causes:
1. **Insufficient credits** - Check your OpenAI dashboard
2. **Network issues** - Check internet connectivity
3. **API restrictions** - Some regions have different endpoints
4. **Rate limits** - Wait a few minutes and try again

Try with a simpler model:
```bash
# Edit test_llm_mutation.py and change model="gpt-4o-mini" to:
model="gpt-3.5-turbo"
```

## Next Steps

If the test passes, you're ready for LLM-enhanced evolution:

```bash
# Full pipeline with LLM
python3 run_complete_pipeline.py --full-pipeline --use-llm

# Or just evolution with LLM
python3 scripts/run_evolution_only.py --use-llm --llm_api_key $OPENAI_API_KEY
```

## Model Options

The test uses `gpt-4o-mini` (fast, cost-effective). Available options:
- `gpt-4o` - Most capable, higher cost
- `gpt-4o-mini` - Fast, good balance (recommended)
- `gpt-3.5-turbo` - Fastest, lowest cost (good for testing)

Run the test now to verify your LLM integration is working! 🚀