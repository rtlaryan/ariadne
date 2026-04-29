"""
core/tokenizer.py — Simple reversible vocabulary map.

Reserved special tokens are pre-seeded; any new tokens found in data
can be added via add_token() or build_from_data().
"""

import json

from ariadne.core.dataset import _KEY_NORMALIZE

_SPECIAL_TOKENS = [
    "[PAD]", "[EOS]", "[GOAL]", "[STATE]", "[ACTION]", "[UNK]"
]


class TokenMap:
    """Bidirectional token ↔ id mapping.

    The first six IDs (0-5) are always reserved for special tokens so that
    the padding index (0) is stable across saves/loads.
    """

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        for tok in _SPECIAL_TOKENS:
            self._add(tok)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, token: str) -> None:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_token(self, token: str) -> None:
        """Add a single token if not already present."""
        self._add(str(token))

    def encode(self, tokens: list[str]) -> list[int]:
        """Map a list of token strings to IDs (unknown → [UNK])."""
        unk_id = self.token_to_id["[UNK]"]
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Map a list of IDs back to token strings."""
        return [self.id_to_token.get(i, "[UNK]") for i in ids]

    def build_from_data(self, episode_iterator) -> None:
        """Scan episodes and add any new action/interaction tokens.

        episode_iterator: yields lists of step-dicts (one list per episode).
        Each step-dict must have a 'state' key with 'availableInteractions'
        and optionally an 'action' key with a 'key' field.
        """
        for episode in episode_iterator:
            for step in episode:
                state = step.get("state", {})
                if "angleMode" in state:
                    self._add("angleMode:")
                    self._add(str(state.get("angleMode", "deg")))
                for tok in state.get("availableInteractions", []):
                    k_norm = _KEY_NORMALIZE.get(tok, tok)
                    self._add(str(k_norm))
                action = step.get("action", {})
                key = action.get("key") if isinstance(action, dict) else None
                if key:
                    k_norm = _KEY_NORMALIZE.get(key, key)
                    self._add(str(k_norm))

    def __len__(self) -> int:
        return len(self.token_to_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TokenMap":
        with open(path, "r") as f:
            token_to_id: dict[str, int] = json.load(f)
        obj = cls.__new__(cls)
        obj.token_to_id = token_to_id
        obj.id_to_token = {v: k for k, v in token_to_id.items()}
        return obj
