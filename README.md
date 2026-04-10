# Ariadne

Ariadne is a calculator-agent training package with four main pieces:

- `agents/`: HTTP-serving data collection and rollout workers
- `core/`: tokenization, dataset preparation, and transformer model code
- `trainers/`: supervised, DAgger, and RL training entry points
- `web_ui/`: a small FastAPI app for interactive checkpoint testing

The main orchestration entry point is [orchestrate.py](/home/aryan/projects/auxila/ariadne/orchestrate.py). It reads [experiment.yaml](/home/aryan/projects/auxila/ariadne/configs/experiment.yaml), generates data when enabled, runs pretraining, optionally runs DAgger, and then optionally runs RL fine-tuning.

The public repo should keep source, configs, and lightweight UI assets here. Generated datasets, checkpoints, logs, and Python cache files should stay out of version control.

Typical commands:

```bash
python -m ariadne.orchestrate --config ariadne/configs/experiment.yaml
uvicorn ariadne.web_ui.app:app --reload --port 7000
```
