# Configs

[experiment.yaml](/home/aryan/projects/auxila/ariadne/configs/experiment.yaml) is the reference experiment config for the Ariadne pipeline.

Copy it, adjust the run name, data paths, and enabled phases for your environment, then pass the resulting file to:

```bash
python -m ariadne.orchestrate --config path/to/experiment.yaml
```

[tokenizer.json](/home/aryan/projects/auxila/ariadne/configs/tokenizer.json) is the base tokenizer seed. Training runs may write expanded tokenizers into their own run directories.
