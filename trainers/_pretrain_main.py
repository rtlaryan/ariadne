"""
trainers/_pretrain_main.py — Entry point for supervised pre-training.

Called by orchestrate.py as a subprocess:
  python -m ariadne.trainers._pretrain_main --config cfg.json --run-dir runs/exp/pre_train
"""
import argparse, json
from ariadne.trainers.pretrain import Trainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    Trainer(cfg, run_dir=args.run_dir, use_dagger=False).train()

if __name__ == "__main__":
    main()
