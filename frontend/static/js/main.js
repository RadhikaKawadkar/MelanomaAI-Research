// ── Tab routing ──────────────────────────────────────────
const tabMeta = {
  diagnosis:  { crumb: 'Diagnosis',         title: 'Diagnosis — MelanomaAI' },
  evaluation: { crumb: 'Model Evaluation',  title: 'Evaluation — MelanomaAI' },
  history:    { crumb: 'Session History',   title: 'History — MelanomaAI' },
};

function switchTab(id, el) {
  document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  if (el) el.classList.add('active');
  const meta = tabMeta[id];
  document.getElementById('crumb-page').textContent = meta.crumb;
  document.title = meta.title;
  if (id === 'evaluation') { loadMetrics(); }
  if (id === 'history')    { loadHistory(); }
}

// ── Status pill ──────────────────────────────────────────
function setStatus(state, text) {
  const pill = document.getElementById('status-pill');
  const txt  = document.getElementById('status-text');
  pill.className = 'status-pill ' + (state || '');
  txt.textContent = text || 'Ready';
}

// ── Toast ────────────────────────────────────────────────
function showToast(msg, icon = 'ℹ️', dur = 3000) {
  const wrap = document.getElementById('toast-wrap');
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = `<span class="toast-icon">${icon}</span><span>${msg}</span>`;
  wrap.appendChild(t);
  setTimeout(() => {
    t.classList.add('fadeout');
    setTimeout(() => t.remove(), 280);
  }, dur);
}

// ── Sidebar metrics ──────────────────────────────────────
async function loadSidebarMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const d   = await res.json();
    document.getElementById('sm-acc').textContent   = d.accuracy.toFixed(2) + '%';
    document.getElementById('sm-auc').textContent   = d.auc.toFixed(4);
    document.getElementById('sm-thresh').textContent = d.threshold.toFixed(2);
    const mode = document.getElementById('sm-mode');
    // Detect demo mode from server header or just show Live
    mode.textContent = 'DEMO';
    mode.classList.add('mode-demo');
  } catch (e) {
    console.warn('Could not load sidebar metrics', e);
  }
}
