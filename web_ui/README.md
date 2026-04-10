# Web UI

The web UI is a small FastAPI app for inspecting Ariadne checkpoints against a live calculator bridge.

Run it from the repository root with:

```bash
uvicorn ariadne.web_ui.app:app --reload --port 7000
```

Static assets live in [static/index.html](/home/aryan/projects/auxila/ariadne/web_ui/static/index.html), [static/script.js](/home/aryan/projects/auxila/ariadne/web_ui/static/script.js), and [static/style.css](/home/aryan/projects/auxila/ariadne/web_ui/static/style.css).
