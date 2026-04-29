"""
trainers/_dagger_main.py — Entry point for DAgger fine-tuning.
"""
import argparse, json
from ariadne.trainers.pretrain import Trainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--run-dir",     required=True)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--tb-log-dir",  default=None,
                   help="Shared TensorBoard log dir for all DAgger iterations. "
                        "Defaults to <run-dir>/logs if not specified.")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    if args.resume_from:
        cfg.setdefault("dagger", {}).setdefault("training", {})["resume_from"] = args.resume_from
    Trainer(
        cfg,
        run_dir=args.run_dir,
        use_dagger=True,
        tb_log_dir=args.tb_log_dir,
    ).train()

if __name__ == "__main__":
    main()
