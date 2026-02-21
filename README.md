# capstone-BlueAlphaProject2

See documentation in [Documentation of Code](https://docs.google.com/document/d/1glQWezaB3eBH13Mxp2eR0Y7SM1zAaVAaS1qHFy2uu-o/edit?usp=sharing)

## Running tests

From the project root, run all tests with:

```bash
python test.py
```

This runs config/loading tests, spend generation tests, and the placeholder suites for impressions, revenue, and pipeline. To run a single suite:

```bash
python -m tests.test_config
python -m tests.test_spend_generation
python -m tests.test_impressions_simulation
python -m tests.test_revenue_simulation
python -m tests.test_pipeline
```