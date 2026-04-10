"""
web_ui/app.py — Ariadne web UI backend.

The UI mirrors the rollout inference path:
  - Full action history by default
  - Left truncation to reserve space for the predicted action
  - Explicit episode resets through /set_goal and /reset
  - The same valid-action masking used during training and rollouts
"""

import glob
import json
import os
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.abspath(os.path.join(current_dir, ".."))

from ariadne.core.dataset import StateSerializer
from ariadne.core.model import build_model, load_checkpoint
from ariadne.core.tokenizer import TokenMap

# ---------------------------------------------------------------------------
# Key normalization shared with rollout and training code
# ---------------------------------------------------------------------------
_NORMALIZE_KEYS: dict[str, str] = {
    "÷": "/", "×": "*", "⌫": "Backspace", "AC": "Escape", "=": "Enter"
}

app = FastAPI()


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class GlobalState:
    def __init__(self) -> None:
        # Model components (set by load_model)
        self.tokenizer:  Optional[TokenMap]        = None
        self.serializer: Optional[StateSerializer] = None
        self.model                                 = None
        self.max_len:    int                       = 256
        self.device:     str                       = "cpu"

        # Session state
        self.goal:            Optional[str]  = None
        self.action_history:  List[str]      = []
        self.frame_count:     int            = 0
        self.latest_frame_b64: Optional[str] = None
        self.failover_active: bool           = False
        self.pending_actions: List[str]      = []

        # WebSocket connections
        self.active_websockets: List[WebSocket] = []

        # Inference settings
        self.history_window:        int   = -1     # -1 = full history (matches training)
        self.confidence_threshold:  float = 0.85
        self.top_k:                 int   = 3
        self.inference_mode:        str   = "greedy"

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self.action_history = []
        self.frame_count    = 0
        self.failover_active = False
        self.pending_actions = []


STATE = GlobalState()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_input(
    goal:           str,
    state:          dict,
    action_history: List[str],
    history_window: int,
    serializer:     StateSerializer,
    tokenizer:      TokenMap,
    max_len:        int,
    device:         str,
) -> torch.Tensor:
    """Build the inference input tensor."""
    state_copy = state.copy()
    hist = action_history if history_window < 0 else action_history[-history_window:]
    state_copy["action_history"] = hist

    goal_toks  = serializer.tokenize_expr(goal)
    state_toks = serializer.serialize(state_copy)
    full       = ["[GOAL]"] + goal_toks + ["[STATE]"] + state_toks + ["[ACTION]"]

    ids = tokenizer.encode(full)
    # Truncate from the left and reserve one slot for the predicted action.
    if len(ids) > max_len - 1:
        ids = ids[-(max_len - 1):]

    return torch.tensor([ids], dtype=torch.long).to(device)


def _apply_action_mask(
    logits:    torch.Tensor,
    state:     dict,
    tokenizer: TokenMap,
    device:    str,
) -> torch.Tensor:
    """Apply the additive valid-action mask used during rollout inference."""
    avail = state.get("availableInteractions", [])
    if not avail:
        return logits
    norm_avail = [_NORMALIZE_KEYS.get(k, k) for k in avail]
    ids = [tokenizer.token_to_id[k] for k in norm_avail if k in tokenizer.token_to_id]
    if not ids:
        return logits
    mask = torch.full((len(tokenizer),), float("-inf"), device=device)
    mask[ids] = 0.0
    return logits + mask


def _infer_greedy(goal: str, state: dict) -> str:
    """Greedy inference."""
    inp = _build_input(
        goal, state,
        STATE.action_history, STATE.history_window,
        STATE.serializer, STATE.tokenizer, STATE.max_len, STATE.device,
    )
    with torch.no_grad():
        logits = STATE.model(inp)
    last   = logits[0, -1, :].clone()
    masked = _apply_action_mask(last, state, STATE.tokenizer, STATE.device)
    action_id = int(masked.argmax())
    return STATE.tokenizer.decode([action_id])[0]


def _infer_beam(goal: str, state: dict) -> tuple[str, float]:
    """Top-k fallback inference with confidence gating."""
    inp = _build_input(
        goal, state,
        STATE.action_history, STATE.history_window,
        STATE.serializer, STATE.tokenizer, STATE.max_len, STATE.device,
    )
    with torch.no_grad():
        logits = STATE.model(inp)
    last   = logits[0, -1, :].clone()
    masked = _apply_action_mask(last, state, STATE.tokenizer, STATE.device)

    probs             = F.softmax(masked, dim=-1)
    top_probs, top_ids = probs.topk(STATE.top_k)

    greedy_prob  = top_probs[0].item()
    greedy_token = STATE.tokenizer.decode([int(top_ids[0])])[0]

    if greedy_prob >= STATE.confidence_threshold:
        return greedy_token, greedy_prob

    # The UI does not maintain a full expression simulator, so the fallback is
    # the masked greedy choice.
    return greedy_token, greedy_prob


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_tokenizer(start_dir: str) -> Optional[str]:
    d = start_dir
    for _ in range(6):
        for name in ("tokenizer.json",):
            t = os.path.join(d, name)
            if os.path.isfile(t):
                return t
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    fallback = os.path.join(package_root, "configs", "tokenizer.json")
    return fallback if os.path.isfile(fallback) else None


def load_model(model_path: str) -> None:
    """Load a tokenizer and checkpoint from a run directory."""
    run_dir = os.path.dirname(model_path)
    if os.path.basename(run_dir) == "checkpoints":
        run_dir = os.path.dirname(run_dir)

    # Tokenizer
    tok_path = _find_tokenizer(run_dir)
    if not tok_path:
        raise FileNotFoundError(f"tokenizer.json not found near {model_path}")

    # Config
    defaults = {"embed_dim": 256, "num_layers": 6, "num_heads": 8, "max_len": 256}
    cfg = dict(defaults)
    d = run_dir
    for _ in range(6):
        c = os.path.join(d, "config.json")
        if os.path.isfile(c):
            with open(c) as f:
                cfg = {**defaults, **json.load(f)}
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer          = TokenMap.load(tok_path)
    cfg["vocab_size"]  = len(tokenizer)

    model = build_model(cfg).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    STATE.tokenizer  = tokenizer
    STATE.serializer = StateSerializer(tokenizer)
    STATE.model      = model
    STATE.max_len    = cfg.get("max_len", 256)
    STATE.device     = device
    STATE.reset()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[web_ui] Loaded {model_path}\n"
        f"         vocab={len(tokenizer)}  max_len={STATE.max_len}  "
        f"device={device}  params={n_params:,}"
    )


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def list_available_models() -> list:
    runs_dir    = os.path.join(package_root, "runs")
    experiments = {}

    if os.path.exists(runs_dir):
        for exp_name in os.listdir(runs_dir):
            exp_path = os.path.join(runs_dir, exp_name)
            if not os.path.isdir(exp_path):
                continue
            experiments[exp_name] = {}
            for phase_name in os.listdir(exp_path):
                phase_path = os.path.join(exp_path, phase_name)
                if not os.path.isdir(phase_path):
                    continue
                ckpt_dir = os.path.join(phase_path, "checkpoints")
                if os.path.isdir(ckpt_dir):
                    checkpoints = [
                        {
                            "name":     os.path.basename(p),
                            "path":     p,
                            "modified": os.path.getmtime(p),
                        }
                        for p in glob.glob(os.path.join(ckpt_dir, "*.pt"))
                    ]
                    if checkpoints:
                        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
                        experiments[exp_name][phase_name] = checkpoints
            if not experiments[exp_name]:
                del experiments[exp_name]

    result = []
    for exp_name, phases in experiments.items():
        exp_obj = {
            "name":   exp_name,
            "phases": [
                {"name": pn, "checkpoints": ck}
                for pn, ck in sorted(phases.items())
            ],
        }
        result.append(exp_obj)
    return sorted(result, key=lambda x: x["name"])


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class GoalRequest(BaseModel):
    goal: str

class LoadModelRequest(BaseModel):
    model_path: str

class SettingsRequest(BaseModel):
    inference_mode:       Optional[str]   = None
    confidence_threshold: Optional[float] = None
    top_k:                Optional[int]   = None
    history_window:       Optional[int]   = None


# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------

async def broadcast(message: Dict) -> None:
    for ws in list(STATE.active_websockets):
        try:
            await ws.send_json(message)
        except Exception:
            STATE.active_websockets.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    STATE.active_websockets.append(websocket)
    try:
        await websocket.send_json({
            "type":           "init",
            "goal":           STATE.goal,
            "action_history": STATE.action_history,
            "latest_frame":   STATE.latest_frame_b64,
            "settings": {
                "inference_mode":       STATE.inference_mode,
                "confidence_threshold": STATE.confidence_threshold,
                "top_k":                STATE.top_k,
                "history_window":       STATE.history_window,
            },
        })
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            if data_json.get("type") == "set_goal":
                STATE.goal = data_json["goal"]
                STATE.reset()
                await broadcast({"type": "status", "msg": f"Goal set to: {STATE.goal}"})
    except WebSocketDisconnect:
        STATE.active_websockets.remove(websocket)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
async def get_models():
    return {"models": list_available_models()}


@app.post("/load_model")
async def switch_model(req: LoadModelRequest):
    try:
        load_model(req.model_path)
        STATE.reset()
        await broadcast({"type": "status", "msg": f"Model loaded: {os.path.basename(req.model_path)}"})
        await broadcast({"type": "reset"})
        return {"status": "ok", "model": req.model_path}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/set_goal")
async def set_goal(req: GoalRequest):
    STATE.goal = req.goal
    STATE.reset()
    STATE.pending_actions.append("Escape")
    await broadcast({"type": "status", "msg": f"Goal set to: {STATE.goal}"})
    await broadcast({"type": "reset"})
    return {"status": "ok", "goal": STATE.goal}


@app.post("/reset")
async def reset_agent():
    STATE.goal = None
    STATE.reset()
    STATE.pending_actions.extend(["Escape", "Escape", "Escape"])
    await broadcast({"type": "reset"})
    await broadcast({"type": "status", "msg": "Agent reset. Calculator cleared."})
    return {"status": "ok"}


@app.get("/settings")
async def get_settings():
    return {
        "inference_mode":       STATE.inference_mode,
        "confidence_threshold": STATE.confidence_threshold,
        "top_k":                STATE.top_k,
        "history_window":       STATE.history_window,
    }


@app.post("/update_settings")
async def update_settings(req: SettingsRequest):
    if req.inference_mode in ("greedy", "beam_search"):
        STATE.inference_mode = req.inference_mode
    if req.confidence_threshold is not None:
        STATE.confidence_threshold = max(0.0, min(1.0, req.confidence_threshold))
    if req.top_k is not None:
        STATE.top_k = max(1, min(10, req.top_k))
    if req.history_window is not None:
        STATE.history_window = req.history_window  # -1 = full

    await broadcast({
        "type": "status",
        "msg": (
            f"Settings — mode={STATE.inference_mode}  "
            f"thr={STATE.confidence_threshold:.2f}  "
            f"top_k={STATE.top_k}  "
            f"hist_win={STATE.history_window}"
        ),
    })
    return {
        "status":               "ok",
        "inference_mode":       STATE.inference_mode,
        "confidence_threshold": STATE.confidence_threshold,
        "top_k":                STATE.top_k,
        "history_window":       STATE.history_window,
    }


@app.post("/step")
async def step(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"type": "wait"}

    # Handle screenshot separately (keep latest frame for UI, strip before inference)
    if "screenshot" in data:
        STATE.latest_frame_b64 = data.pop("screenshot")
        await broadcast({
            "type":     "frame",
            "frame":    STATE.latest_frame_b64,
            "frame_id": STATE.frame_count,
        })
        STATE.frame_count += 1
        STATE.failover_active = False
    else:
        if STATE.latest_frame_b64 is None and not STATE.failover_active:
            STATE.failover_active = True
            await broadcast({"type": "status", "msg": "[FAILOVER] No video signal."})
            await broadcast({"type": "blind_mode"})

    if not STATE.model_loaded:
        return {"type": "wait"}

    if STATE.pending_actions:
        key = STATE.pending_actions.pop(0)
        await broadcast({"type": "status", "msg": f"Priority action: {key}"})
        return {"type": "keypress", "key": key}

    if not STATE.goal:
        return {"type": "wait"}

    try:
        confidence = None
        if STATE.inference_mode == "greedy":
            action_token = _infer_greedy(STATE.goal, data)
        else:
            action_token, confidence = _infer_beam(STATE.goal, data)

        STATE.action_history.append(action_token)

        await broadcast({
            "type":       "action",
            "action":     action_token,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "history":    STATE.action_history[-STATE.history_window:]
                          if STATE.history_window > 0
                          else STATE.action_history,
        })

        response_action = {"type": "keypress", "key": action_token}

        if action_token in ("Enter", "=", "<TERMINATE>"):
            if action_token == "<TERMINATE>":
                response_action = {"type": "terminate"}
            await broadcast({"type": "status", "msg": f"Episode complete: {action_token}"})
            STATE.goal = None
            await broadcast({"type": "status", "msg": "Waiting for new objective."})

        return response_action

    except Exception as e:
        import traceback; traceback.print_exc()
        await broadcast({"type": "status", "msg": f"Inference error: {e}"})
        return {"type": "wait"}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
app.mount(
    "/",
    StaticFiles(directory=os.path.join(current_dir, "static"), html=True),
    name="static",
)

if __name__ == "__main__":
    print("Use: uvicorn ariadne.web_ui.app:app --reload --port 7000")
