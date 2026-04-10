const modelSelect = document.getElementById('model-select');
const loadModelBtn = document.getElementById('load-model-btn');
const inferenceMode = document.getElementById('inference-mode');
const confThreshold = document.getElementById('conf-threshold');
const topK = document.getElementById('top-k');
const resetBtn = document.getElementById('reset-btn');

const goalInput = document.getElementById('goal-input');
const setGoalBtn = document.getElementById('set-goal-btn');

const screenView = document.getElementById('screen-view');
const connectionStatus = document.getElementById('connection-status');
const blindModeOverlay = document.getElementById('blind-mode-overlay');

const actionHistory = document.getElementById('action-history');
const currentConfidence = document.getElementById('current-confidence');
const sysLogs = document.getElementById('sys-logs');

let ws;
let reconnectInterval = 2000;

async function init() {
    log('System initializing...', 'info');
    await fetchModels();
    connectWebSocket();
    setupEventListeners();
}

async function fetchModels() {
    try {
        const res = await fetch('/models');
        const data = await res.json();
        populateModelDropdown(data.models);
    } catch (e) {
        log(`Failed to fetch models: ${e}`, 'error');
    }
}

function populateModelDropdown(models) {
    modelSelect.innerHTML = '';

    if (models.length === 0) {
        const opt = document.createElement('option');
        opt.value = "";
        opt.textContent = "No models found in ariadne/runs/";
        modelSelect.appendChild(opt);
        return;
    }

    models.forEach(exp => {
        const group = document.createElement('optgroup');
        group.label = exp.name;

        exp.phases.forEach(phase => {
            phase.checkpoints.forEach(ckpt => {
                const opt = document.createElement('option');
                opt.value = ckpt.path;
                opt.textContent = `[${phase.name}] ${ckpt.name}`;
                group.appendChild(opt);
            });
        });

        if (group.children.length > 0) {
            modelSelect.appendChild(group);
        }
    });
}

async function loadSelectedModel() {
    const path = modelSelect.value;
    if (!path) return;

    loadModelBtn.disabled = true;
    loadModelBtn.textContent = 'Loading...';
    try {
        const res = await fetch('/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: path })
        });
        const data = await res.json();
        if (data.status === 'ok') {
            log(`Model loaded: ${path.split('/').pop()}`, 'success');
        } else {
            log(`Failed to load model: ${data.error}`, 'error');
        }
    } catch (e) {
        log(`API Error: ${e}`, 'error');
    } finally {
        loadModelBtn.disabled = false;
        loadModelBtn.textContent = 'Load';
    }
}

async function updateSettings() {
    const settings = {
        inference_mode: inferenceMode.value,
        confidence_threshold: parseFloat(confThreshold.value),
        top_k: parseInt(topK.value, 10)
    };

    try {
        await fetch('/update_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
    } catch (e) {
        log(`Failed to update settings: ${e}`, 'error');
    }
}

async function setGoal() {
    const text = goalInput.value.trim();
    if (!text) return;

    setGoalBtn.disabled = true;
    try {
        await fetch('/set_goal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ goal: text })
        });
        log(`Objective set: ${text}`, 'info');
    } catch (e) {
        log(`Failed to set goal: ${e}`, 'error');
    } finally {
        setGoalBtn.disabled = false;
    }
}

async function forceReset() {
    try {
        await fetch('/reset', { method: 'POST' });
        goalInput.value = '';
    } catch (e) {
        log(`Failed to reset: ${e}`, 'error');
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        setConnStatus(true);
        log('WebSocket connected', 'success');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWsMessage(data);
    };

    ws.onclose = () => {
        setConnStatus(false);
        setTimeout(connectWebSocket, reconnectInterval);
    };

    ws.onerror = (err) => {
        console.error("WS Error", err);
        ws.close();
    };
}

function handleWsMessage(data) {
    if (data.type === 'init') {
        if (data.goal) goalInput.value = data.goal;
        if (data.latest_frame) updateFrame(data.latest_frame);
        if (data.action_history) renderHistory(data.action_history);

        if (data.settings) {
            inferenceMode.value = data.settings.inference_mode;
            confThreshold.value = data.settings.confidence_threshold;
            topK.value = data.settings.top_k;
        }
    }
    else if (data.type === 'frame') {
        updateFrame(data.frame);
        blindModeOverlay.classList.add('hidden');
    }
    else if (data.type === 'blind_mode') {
        blindModeOverlay.classList.remove('hidden');
    }
    else if (data.type === 'action') {
        renderHistory(data.history);
        if (data.confidence !== null) {
            currentConfidence.textContent = `Conf: ${(data.confidence * 100).toFixed(1)}%`;
            if (data.confidence < parseFloat(confThreshold.value)) {
                currentConfidence.style.color = 'var(--danger)';
            } else {
                currentConfidence.style.color = 'var(--success)';
            }
        } else {
            currentConfidence.textContent = '';
        }
    }
    else if (data.type === 'status') {
        log(data.msg, 'info');
    }
    else if (data.type === 'reset') {
        actionHistory.innerHTML = '';
        currentConfidence.textContent = '';
    }
}

function updateFrame(b64) {
    screenView.src = `data:image/jpeg;base64,${b64}`;
}

function renderHistory(tokens) {
    actionHistory.innerHTML = '';
    tokens.forEach(tok => {
        const el = document.createElement('div');
        el.className = 'token';
        el.textContent = tok;
        actionHistory.appendChild(el);
    });
}

function log(msg, type = 'info') {
    const el = document.createElement('p');
    el.className = type;

    const now = new Date();
    const time = now.toTimeString().split(' ')[0];

    const ts = document.createElement('span');
    ts.className = 'timestamp';
    ts.textContent = `[${time}]`;

    const txt = document.createTextNode(` ${msg}`);

    el.appendChild(ts);
    el.appendChild(txt);

    sysLogs.appendChild(el);
    sysLogs.scrollTop = sysLogs.scrollHeight;
}

function setConnStatus(isConnected) {
    if (isConnected) {
        connectionStatus.textContent = 'Connected';
        connectionStatus.className = 'status-badge connected';
    } else {
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.className = 'status-badge disconnected';
    }
}

// --- Event Listeners ---

function setupEventListeners() {
    loadModelBtn.addEventListener('click', loadSelectedModel);

    inferenceMode.addEventListener('change', updateSettings);
    confThreshold.addEventListener('change', updateSettings);
    topK.addEventListener('change', updateSettings);

    setGoalBtn.addEventListener('click', setGoal);
    goalInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') setGoal();
    });

    resetBtn.addEventListener('click', forceReset);
}

// Boot
init();
