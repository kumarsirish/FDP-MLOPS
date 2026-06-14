# MLOps Teaching Demo — What MLflow Is Doing Here

This project trains a small CNN on MNIST (PyTorch) and uses **[MLflow](https://mlflow.org)** to turn a throwaway notebook experiment into a tracked, versioned, reproducible workflow. This README explains exactly what MLflow does at each step of the notebook (`MLOps_MNIST_CNN_Teaching_Demo.ipynb`).

## The problem MLflow solves

Without it, a typical training notebook leaves three questions unanswerable:

1. **Which run was best?** Results live in terminal scrollback and memory.
2. **Can you reproduce it?** Hyperparameters were hardcoded and changed between runs.
3. **Which model is "the" model?** Weights files get passed around with no version or provenance.

MLflow fixes this by treating **experiments as data**: every run's inputs, outcomes, and outputs are recorded in a queryable store.

## What MLflow records (Tracking)

The notebook configures a local tracking store and an experiment:

```python
mlflow.set_tracking_uri("file:./mlruns")   # everything is saved to ./mlruns
mlflow.set_experiment("mnist-cnn-mlops")   # a named group of comparable runs
```

Each training run is wrapped in `mlflow.start_run()`, and three kinds of information are logged:

| What | API call | In this project |
|---|---|---|
| **Parameters** — the knobs we chose | `mlflow.log_params(config)` | `lr`, `epochs`, `batch_size`, `optimizer`, `seed` |
| **Metrics** — outcomes over time | `mlflow.log_metric(..., step=epoch)` | `train_loss`, `val_accuracy`, `epoch_seconds` per epoch |
| **Artifacts** — files produced | `mlflow.pytorch.log_model(model)` | the trained CNN plus its environment (`requirements.txt`, `MLmodel` metadata) |

The notebook runs a **learning-rate sweep** (`lr ∈ {0.0005, 0.001, 0.005}`, 2 epochs each), so the experiment ends up containing three fully documented runs.

## Comparing runs

Instead of remembering results, the notebook queries the tracking store like a database:

```python
runs = mlflow.search_runs(experiment_names=["mnist-cnn-mlops"])
```

This returns a pandas DataFrame of all runs, which the notebook sorts by `val_accuracy` to build a leaderboard and pick the winner on evidence. The full web dashboard is available with:

```bash
mlflow ui --backend-store-uri file:./mlruns   # then open http://localhost:5000
```

(In Colab, the optional cell proxies this UI into a browser tab.)

## Versioning the winner (Model Registry)

The best run is promoted into MLflow's **Model Registry** under a stable name:

```python
mlflow.register_model(f"runs:/{best_run_id}/model", name="mnist-cnn")   # -> version 1
```

From then on, any code loads the model **by name and version, not by file path**:

```python
model = mlflow.pytorch.load_model("models:/mnist-cnn/1")
```

Every registered version links back to the exact run — parameters, metrics, code — that produced it (provenance), and a bad deployment can be rolled back by simply pointing to a previous version.

## How the registry model is used downstream

After registration, the notebook never touches raw weight files again:

1. **Validation gates** — the registry model must pass automated checks (accuracy ≥ threshold, latency, sanity predictions, model size) before "deployment."
2. **Serving** — a Gradio app serves the registry model for live digit drawing.
3. **Monitoring** — simulated drift (rotated/noisy inputs) is evaluated and the results are logged back to MLflow as a `production-monitoring` run, closing the lifecycle loop.
4. **Distribution** — the approved weights are published to the Hugging Face Hub with a model card recording the registry version, metrics, and MLflow run id.

## Where everything lives

```
./mlruns/                      # the entire MLflow store (local mode)
├── <experiment_id>/
│   └── <run_id>/
│       ├── params/            # one file per logged parameter
│       ├── metrics/           # time series, one file per metric
│       └── artifacts/model/   # MLmodel, weights, environment files
└── models/                    # registry: names, versions, source runs
```

In this demo all of it is a local folder (zip it and the full history travels with you). In a team setting the same code points at a shared tracking server (`mlflow.set_tracking_uri("http://...")`) backed by a database and S3/GCS artifact storage — **only that one line changes**.

## Quick start

```bash
pip install mlflow torch torchvision
```

Run the notebook top to bottom (≈10 min on a free Colab T4 GPU), or reuse the pattern in any script:

```python
import mlflow

mlflow.set_experiment("my-experiment")
with mlflow.start_run():
    mlflow.log_params(config)
    for epoch in range(epochs):
        ...train...
        mlflow.log_metric("val_acc", acc, step=epoch)
    mlflow.pytorch.log_model(model, name="model")
```

## One-line summary

**MLflow is the project's lab notebook and model ledger:** it records what was tried (params), what happened (metrics), and what was produced (model artifacts); lets us pick winners with queries instead of memory; and gives the chosen model a version, a name, and a traceable history all the way to deployment.
