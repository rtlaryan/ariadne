"""
trainers/_rl_main.py — Entry point for RL training.
"""
import argparse, json
from ariadne.trainers.rl import RLTrainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          required=True)
    p.add_argument("--run-dir",         required=True)
    p.add_argument("--rollout-dir",     required=True)
    p.add_argument("--resume-from",     required=True)
    p.add_argument("--reference-from",  required=True)
    p.add_argument("--tokenizer-path",  required=True)
    p.add_argument("--iteration-index", type=int, default=0)
    p.add_argument("--decay-step",      type=int, default=0)
    p.add_argument("--run-name",        default="rl")
    p.add_argument("--tb-log-dir",      default=None,
                   help="Shared TensorBoard log dir for all RL iterations. "
                        "Defaults to <run-dir>/logs if not specified.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Discover supervised data for anchor
    import glob, os
    sup_files = None
    pre_dir   = cfg.get("pretrain", {}).get("data_dir")
    if pre_dir:
        # pre_dir may be a str or a list of dirs (as set by orchestrate.py)
        if isinstance(pre_dir, str):
            pre_dir = [pre_dir]
        files = []
        for d in pre_dir:
            files.extend(glob.glob(os.path.join(d, "**/*.jsonl"), recursive=True))
        if files:
            sup_files = files

    trainer = RLTrainer(
        cfg             = cfg,
        run_dir         = args.run_dir,
        rollout_dir     = args.rollout_dir,
        resume_from     = args.resume_from,
        reference_from  = args.reference_from,
        tokenizer_path  = args.tokenizer_path,
        iteration_index = args.iteration_index,
        decay_step      = args.decay_step,
        run_name        = args.run_name,
        tb_log_dir      = args.tb_log_dir,
    )
    trainer.train(sup_files=sup_files)

if __name__ == "__main__":
    main()
