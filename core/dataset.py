"""
core/dataset.py — Dataset and state-serialization utilities.

Public API
----------
StateSerializer
    Converts icalc state dicts + goal expressions into token lists.
    Shared by the dataset, inference agent, and data-gen agents.

SoftwareTrajectoryDataset (IterableDataset)
    Streams episodes from JSONL files.  Supports three file categories:
      supervised, crawler, dagger
    and two training modes:
      standard (supervised + crawler only)
      dagger    (mixes dagger data with expert data at a configurable ratio)
"""

import glob
import json
import os
import random
from typing import Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# Key normalization — display symbols → canonical action tokens
# ---------------------------------------------------------------------------

_KEY_NORMALIZE: dict[str, str] = {
    "÷": "/",
    "×": "*",
    "⌫": "Backspace",
    "AC": "Escape",
    "=": "Enter",
}


# ---------------------------------------------------------------------------
# StateSerializer
# ---------------------------------------------------------------------------

class StateSerializer:
    """Converts icalc state dicts and math expressions into token sequences.

    This is the single source of truth for serialization — used by the
    dataset, the inference agent, and the data-gen agents.  Previously this
    logic was duplicated across dataset.py and inference.py (with subtle
    differences); here it lives in one place.
    """

    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer

    # ------------------------------------------------------------------
    # Expression tokenization
    # ------------------------------------------------------------------

    def tokenize_expr(self, expr) -> list[str]:
        """Tokenize a math expression string into token pieces.

        If *expr* is already a list (hindsight goals) it is returned as-is
        (stringified).  Otherwise greedy longest-match tokenization is used
        so that multi-char tokens like 'sin', 'sqrt', '10' match correctly.
        """
        if isinstance(expr, list):
            return [str(x) for x in expr if str(x).strip()]
        return self._greedy(str(expr).replace(" ", ""))

    def _greedy(self, text: str) -> list[str]:
        """Greedy longest-match tokenizer against the vocabulary."""
        if not text:
            return []
        if text in self.tok.token_to_id:
            return [text]
        tokens: list[str] = []
        i = 0
        while i < len(text):
            best = None
            for end in range(min(i + 12, len(text)), i, -1):
                if text[i:end] in self.tok.token_to_id:
                    best = text[i:end]
                    break
            if best:
                tokens.append(best)
                i += len(best)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def _tokenize_val(self, val) -> list[str]:
        return self._greedy(str(val))

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def serialize(self, state: dict) -> list[str]:
        """Serialize an icalc state dict to a list of tokens.

        Format:
          mode: <m>
          [readout: <r>]
          [history: <h1> <h2> ...]
          [keys: <k1> <k2> ...]
          [past: <a1> <a2> ...]
        """
        tokens: list[str] = []

        # mode
        tokens.append("mode:")
        tokens.append(str(state.get("mode", "unknown")))

        # readout
        if "readout" in state:
            val = state["readout"]
            if val and val != "0":
                tokens.append("readout:")
                tokens.extend(self._tokenize_val(val))

        # history (expression history — list of strings)
        hist = state.get("history", [])
        if hist:
            tokens.append("history:")
            for h in hist:
                tokens.extend(self._tokenize_val(h))

        # available keys
        avail = state.get("availableInteractions", [])
        if avail:
            tokens.append("keys:")
            for k in avail:
                k_norm = _KEY_NORMALIZE.get(k, k)
                tokens.extend(self._tokenize_val(k_norm))

        # action history
        past = state.get("action_history", [])
        if past:
            tokens.append("past:")
            for a in past:
                tokens.extend(self._tokenize_val(a))

        return tokens


# ---------------------------------------------------------------------------
# JSONL file helpers
# ---------------------------------------------------------------------------

def _collect_jsonl(directories: list[str]) -> list[str]:
    files = []
    for d in directories:
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, "**/*.jsonl"), recursive=True)))
        elif d.endswith(".jsonl") and os.path.isfile(d):
            files.append(d)
    return files


def _iter_episodes(files: list[str]) -> Iterator[list[dict]]:
    """Yield episodes (lists of step dicts) from a list of JSONL files.

    Each JSONL file has one step per line.  Steps with the same episode_id
    are grouped into an episode, in file order.
    """
    for path in files:
        try:
            with open(path, "r") as f:
                episode: list[dict] = []
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        step = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    eid = step.get("episode_id")
                    if episode and eid != episode[0].get("episode_id"):
                        yield episode
                        episode = []
                    episode.append(step)
                if episode:
                    yield episode
        except OSError:
            continue


def _count_episodes(files: list[str]) -> tuple[int, int]:
    """Return (episode_count, step_count) for a list of files."""
    ep_count = step_count = 0
    for path in files:
        try:
            with open(path, "r") as f:
                cur_eid = None
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        step = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    eid = step.get("episode_id")
                    if eid != cur_eid:
                        ep_count += 1
                        cur_eid = eid
                    step_count += 1
        except OSError:
            continue
    return ep_count, step_count


def _episode_success(episode: list[dict]) -> bool:
    """Infer whether an episode completed successfully."""
    if not episode:
        return False

    final_step = episode[-1]
    if "success" in final_step:
        return bool(final_step.get("success", False))

    action = final_step.get("action", {})
    if isinstance(action, dict):
        key = action.get("key")
    else:
        key = action

    if key is None:
        return False

    return _KEY_NORMALIZE.get(str(key), str(key)) == "Enter"


# ---------------------------------------------------------------------------
# File categorization
# ---------------------------------------------------------------------------

def _categorize_files(
    files: list[str],
    current_iteration: int = 1,
    decay_factor: float = 1.0,
) -> tuple[list[str], list[str], list[str]]:
    """Partition *files* into (supervised, crawler, dagger) lists.

    Classification is by filename (basename), not full path, to avoid
    misclassifying nested supervised files inside dagger_iter_X folders.

    Dagger files from older iterations may be stochastically dropped
    according to the exponential decay schedule:
      P(keep) = decay_factor ^ age,   age = current_iteration - iter_num
    """
    supervised: list[str] = []
    crawler:    list[str] = []
    dagger:     list[str] = []

    for path in files:
        fname = os.path.basename(path)
        if "dagger" in fname:
            # Determine iteration age from the path
            iter_num = -1
            for part in path.split(os.sep):
                if part.startswith("dagger_iter_"):
                    try:
                        iter_num = int(part.split("_")[-1])
                    except ValueError:
                        pass
            if iter_num != -1:
                age  = max(0, current_iteration - iter_num)
                prob = decay_factor ** age
                if random.random() > prob:
                    continue  # dropped by decay
            dagger.append(path)
        elif "supervised" in fname:
            supervised.append(path)
        else:
            crawler.append(path)

    return supervised, crawler, dagger


# ---------------------------------------------------------------------------
# Shuffled infinite / finite file generators
# ---------------------------------------------------------------------------

def _file_generator(
    files: list[str],
    limit: Optional[int] = None,
    repeat: bool = False,
) -> Iterator[list[dict]]:
    """Yield episodes from *files*, optionally repeating forever."""
    if not files:
        return
    count = 0
    while True:
        random.shuffle(files)
        gens = [_iter_episodes([f]) for f in files]
        while gens:
            idx = random.randrange(len(gens))
            try:
                ep = next(gens[idx])
                yield ep
                count += 1
                if limit and count >= limit:
                    return
            except StopIteration:
                gens.pop(idx)
        if not repeat:
            return
        if limit and count >= limit:
            return


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SoftwareTrajectoryDataset(IterableDataset):
    """Streams processed training samples from JSONL episode files.

    Each yielded sample is a dict:
      input_ids : LongTensor [L]   (or [max_len] when use_packing=False)
      labels    : LongTensor [L]   (-100 for non-action positions)
      src       : LongTensor []    (0=supervised, 1=crawler, 2=dagger)
      episode_success : FloatTensor [] (1.0 if the episode finished successfully)

    Parameters
    ----------
    data_files      : list of .jsonl paths
    tokenizer       : TokenMap instance
    serializer      : StateSerializer (created automatically if None)
    max_len         : max token length per sample
    supervised_ratio: probability of sampling supervised vs crawler data
    use_dagger      : whether to include dagger files
    expert_multiplier: ratio of expert episodes per dagger episode (>=0)
    decay_factor    : per-iteration exponential decay for old dagger data
    current_iteration: current DAgger iteration (for decay)
    max_episodes    : approximate total episode limit across all workers
    use_packing     : if True, don't pad (collate with PackedCollator)
    """

    # Source ID constants
    SRC_SUPERVISED = 0
    SRC_CRAWLER    = 1
    SRC_DAGGER     = 2

    def __init__(
        self,
        data_files:        list[str],
        tokenizer,
        serializer:        Optional[StateSerializer] = None,
        max_len:           int   = 256,
        supervised_ratio:  float = 0.5,
        use_dagger:        bool  = False,
        expert_multiplier: float = 1.0,
        decay_factor:      float = 1.0,
        current_iteration: int   = 1,
        max_episodes:      Optional[int] = None,
        use_packing:       bool  = False,
    ) -> None:
        self.tokenizer         = tokenizer
        self.serializer        = serializer or StateSerializer(tokenizer)
        self.max_len           = max_len
        self.supervised_ratio  = supervised_ratio
        self.use_dagger        = use_dagger
        self.expert_multiplier = expert_multiplier
        self.max_episodes      = max_episodes
        self.use_packing       = use_packing

        self.supervised_files, self.crawler_files, self.dagger_files = _categorize_files(
            data_files, current_iteration, decay_factor
        )

        # Enforce ratio extremes
        if supervised_ratio >= 1.0:
            self.crawler_files = []
        elif supervised_ratio <= 0.0:
            self.supervised_files = []

        # Stats
        self.sup_ep,  sup_st   = _count_episodes(self.supervised_files)
        self.crl_ep,  crl_st   = _count_episodes(self.crawler_files)
        self.dag_ep,  dag_st   = _count_episodes(self.dagger_files)
        print(
            f"[Dataset] Files — Sup: {len(self.supervised_files)}, "
            f"Crawl: {len(self.crawler_files)}, DAgger: {len(self.dagger_files)}"
        )
        print(
            f"[Dataset] Episodes — Sup: {self.sup_ep}, Crawl: {self.crl_ep}, DAgger: {self.dag_ep} "
            f"(steps: {sup_st+crl_st+dag_st})"
        )

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _process(self, episode: list[dict], src_id: int) -> dict:
        """Turn a raw episode into a training sample.

        For supervised/dagger episodes we use the oracle task as the goal.
        For crawler episodes we use hindsight relabelling (future state).
        """
        T  = len(episode)
        t  = random.randint(0, T - 1)
        st = episode[t]
        episode_success = float(_episode_success(episode))

        mode = st.get("mode", "unknown")

        # Goal tokens
        goal_tokens: list[str]
        if mode in ("supervised", "dagger") and st.get("task"):
            goal_tokens = self.serializer.tokenize_expr(st["task"])
        else:
            # Hindsight: use a future state's history as the goal
            goal_tokens = ["None"]
            if t < T - 1:
                k = random.randint(t + 1, T - 1)
                future_hist = episode[k].get("state", {}).get("history", [])
                if future_hist:
                    goal_tokens = [str(x) for x in future_hist]

        state_tokens = self.serializer.serialize(st.get("state", {}))

        action_obj = st.get("action", {})
        action_key = (
            action_obj.get("key", "[PAD]")
            if isinstance(action_obj, dict)
            else str(action_obj)
        )

        full_tokens = ["[GOAL]"] + goal_tokens + ["[STATE]"] + state_tokens + ["[ACTION]"]
        full_tokens.append(action_key)
        full_tokens.append("[EOS]")

        input_ids = self.tokenizer.encode(full_tokens)

        # Labels: -100 everywhere except the action token(s)
        try:
            action_sep = full_tokens.index("[ACTION]")
        except ValueError:
            action_sep = 0

        labels = [-100] * len(input_ids)
        for i in range(action_sep + 1, len(input_ids)):
            labels[i] = input_ids[i]

        # Truncate or pad
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels    = labels[:self.max_len]
        elif not self.use_packing:
            pad_id  = self.tokenizer.token_to_id.get("[PAD]", 0)
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            labels    = labels    + [-100]   * pad_len

        return {
            "input_ids":        torch.tensor(input_ids,       dtype=torch.long),
            "labels":           torch.tensor(labels,          dtype=torch.long),
            "src":              torch.tensor(src_id,          dtype=torch.long),
            "episode_success":  torch.tensor(episode_success, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        def shard(lst: list, n: int, i: int) -> list:
            return lst[i::n]

        if worker_info is None:
            sup_files  = self.supervised_files
            crl_files  = self.crawler_files
            dag_files  = self.dagger_files
            limit      = self.max_episodes
        else:
            n, wid     = worker_info.num_workers, worker_info.id
            sup_files  = shard(self.supervised_files, n, wid)
            crl_files  = shard(self.crawler_files,    n, wid)
            dag_files  = shard(self.dagger_files,     n, wid)
            limit      = (
                max(1, self.max_episodes // n)
                if self.max_episodes else None
            )

        if not self.use_dagger:
            # ---- Standard mode: single pass through supervised + crawler ----
            sup_gen = _file_generator(sup_files,  limit=limit, repeat=False)
            crl_gen = _file_generator(crl_files,  limit=limit, repeat=False)
            sup_done = crl_done = False
            while not (sup_done and crl_done):
                use_sup = False
                if not sup_done and not crl_done:
                    use_sup = random.random() < self.supervised_ratio
                elif not sup_done:
                    use_sup = True
                elif not crl_done:
                    use_sup = False
                else:
                    break
                try:
                    if use_sup:
                        yield self._process(next(sup_gen), self.SRC_SUPERVISED)
                    else:
                        yield self._process(next(crl_gen), self.SRC_CRAWLER)
                except StopIteration:
                    if use_sup:
                        sup_done = True
                    else:
                        crl_done = True
            return

        # ---- DAgger mode: interleave dagger (single pass) + expert (repeat) ----
        # P(expert) = M/(M+1) where M = expert_multiplier
        p_expert = (
            self.expert_multiplier / (1.0 + self.expert_multiplier)
            if self.expert_multiplier >= 0
            else 0.0
        )

        dag_gen  = _file_generator(dag_files,  limit=limit, repeat=False)
        sup_gen  = _file_generator(sup_files,  limit=limit, repeat=True)
        crl_gen  = _file_generator(crl_files,  limit=limit, repeat=True)

        while True:
            use_expert = random.random() < p_expert
            if use_expert:
                use_sup = random.random() < self.supervised_ratio
                gen     = sup_gen if (use_sup and sup_files) or not crl_files else crl_gen
                src     = self.SRC_SUPERVISED if gen is sup_gen else self.SRC_CRAWLER
                try:
                    yield self._process(next(gen), src)
                except StopIteration:
                    pass   # infinite generators rarely exhaust, but be safe
            else:
                try:
                    yield self._process(next(dag_gen), self.SRC_DAGGER)
                except StopIteration:
                    break  # DAgger data exhausted → end of epoch
