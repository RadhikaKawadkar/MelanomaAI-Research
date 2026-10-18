// ── History state ─────────────────────────────────────────
let historyData = [];
let sortState   = { key: 'time', dir: -1 };

// ── Load from server session ──────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    historyData = await res.json();
  } catch(e) {
    historyData = [];
  }
  renderHistory();
  renderHistoryChart();
  updateHistoryKPIs();

  // Badge on nav
  const badge = document.getElementById('badge-history');
  if (historyData.length) {
    badge.textContent = historyData.length;
    badge.style.display = 'inline-block';
  }
}

// ── KPI update ────────────────────────────────────────────
function updateHistoryKPIs() {
  const total = historyData.length;
  const mal   = historyData.filter(s => s.verdict === 'MALIGNANT').length;
  const ben   = total - mal;
  const avgP  = total
    ? (historyData.reduce((a,s) => a + s.prob_raw, 0) / total * 100).toFixed(1) + '%'
    : '—';
  const high  = historyData.filter(s => s.risk === 'HIGH' || s.risk === 'CRITICAL').length;

  document.getElementById('sk-total').textContent = total;
  document.getElementById('sk-mal').textContent   = mal;
  document.getElementById('sk-ben').textContent   = ben;
  document.getElementById('sk-avg').textContent   = avgP;
  document.getElementById('sk-high').textContent  = high;
}

// ── Table render ──────────────────────────────────────────
function renderHistory() {
  const tbody = document.getElementById('history-tbody');
  if (!historyData.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="table-empty">No scans yet — run a diagnosis to see history here</td></tr>`;
    return;
  }

  tbody.innerHTML = historyData.map((s, i) => {
    const rClass  = s.risk.toLowerCase();
    const vClass  = s.verdict.toLowerCase();
    const rColors = { low:'#16a34a', moderate:'#d97706', high:'#ea580c', critical:'#dc2626' };
    const rColor  = rColors[rClass] || '#9ca3af';

    return `<tr class="row-${vClass}">
      <td>
        <span style="color:var(--text4);margin-right:6px">#${historyData.length - i}</span>
        <span style="color:var(--navy);font-weight:500">${escHtml(s.name)}</span>
      </td>
      <td>
        <span class="verdict-pill ${vClass}">
          ${vClass === 'malignant' ? '⚠' : '✓'} ${s.verdict}
        </span>
      </td>
      <td style="color:var(--navy);font-weight:600">${s.prob}</td>
      <td>
        <span class="risk-tag r-${rClass}" style="font-size:11px;padding:3px 9px">${s.risk}</span>
      </td>
      <td>
        <div class="mini-bar-wrap">
          <div class="mini-bar">
            <div class="mini-bar-fill" style="width:${s.score}%;background:${rColor};"></div>
          </div>
          <span style="min-width:28px;text-align:right">${s.score}</span>
        </div>
      </td>
      <td>${escHtml(s.tone)}</td>
      <td style="color:var(--text4)">${escHtml(s.time)}</td>
    </tr>`;
  }).join('');
}

// ── Sort ──────────────────────────────────────────────────
function sortTable(key) {
  if (sortState.key === key) sortState.dir *= -1;
  else sortState = { key, dir: 1 };

  historyData.sort((a, b) => {
    let av = a[key], bv = b[key];
    if (key === 'prob_raw' || key === 'score') { av = +av; bv = +bv; }
    return av < bv ? -sortState.dir : av > bv ? sortState.dir : 0;
  });

  renderHistory();
}

// ── Clear ─────────────────────────────────────────────────
async function clearHistory() {
  if (!historyData.length) { showToast('History is already empty', 'ℹ️'); return; }
  await fetch('/api/history/clear', { method: 'POST' });
  historyData = [];
  renderHistory();
  updateHistoryKPIs();
  renderHistoryChart();
  document.getElementById('badge-history').style.display = 'none';
  showToast('History cleared', '🗑');
}

// ── Export CSV ────────────────────────────────────────────
function exportCSV() {
  if (!historyData.length) { showToast('No history to export', '❌'); return; }
  const header = ['Image Name','Verdict','Probability','Risk Level','Risk Score','Skin Tone','Time'];
  const rows   = historyData.map(s => [s.name, s.verdict, s.prob, s.risk, s.score, s.tone, s.time]);
  const csv    = [header, ...rows].map(r => r.map(v => `"${v}"`).join(',')).join('\n');
  const blob   = new Blob([csv], { type: 'text/csv' });
  const a      = document.createElement('a');
  a.href       = URL.createObjectURL(blob);
  a.download   = `melanoma_history_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  showToast('CSV exported ✓', '📄');
}

// ── History bar chart ─────────────────────────────────────
function renderHistoryChart() {
  const canvas = document.getElementById('c-history');
  if (!canvas) return;
  const W = canvas.parentElement.offsetWidth || 600;
  canvas.width = W; canvas.height = 160;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, 160);

  if (!historyData.length) {
    ctx.fillStyle = '#9ca3af'; ctx.font = '13px DM Sans'; ctx.textAlign = 'center';
    ctx.fillText('No scan history yet', W/2, 85);
    return;
  }

  const data  = [...historyData].reverse();
  const p     = { t:16, b:28, l:30, r:12 };
  const bw    = Math.min(56, (W-p.l-p.r)/data.length - 5);
  const rColors = { low:'#16a34a', moderate:'#d97706', high:'#ea580c', critical:'#dc2626' };

  // Grid lines
  [25,50,75,100].forEach(v => {
    const y = p.t + (160-p.t-p.b)*(1-v/100);
    ctx.strokeStyle='rgba(0,0,0,0.05)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(p.l,y); ctx.lineTo(W-p.r,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#9ca3af'; ctx.font='9px JetBrains Mono'; ctx.textAlign='right';
    ctx.fillText(v, p.l-4, y+3);
  });

  data.forEach((s, i) => {
    const x   = p.l + (W-p.l-p.r)/data.length*(i+0.5) - bw/2;
    const bh  = (160-p.t-p.b)*(s.score/100);
    const y   = 160-p.b-bh;
    const col = rColors[s.risk.toLowerCase()] || '#9ca3af';

    // Bar with rounded top
    ctx.fillStyle   = col + '30';
    ctx.strokeStyle = col;
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    const r2 = Math.min(4, bw/4);
    ctx.moveTo(x+r2, y); ctx.lineTo(x+bw-r2, y);
    ctx.quadraticCurveTo(x+bw, y, x+bw, y+r2);
    ctx.lineTo(x+bw, 160-p.b); ctx.lineTo(x, 160-p.b); ctx.lineTo(x, y+r2);
    ctx.quadraticCurveTo(x, y, x+r2, y);
    ctx.closePath(); ctx.fill(); ctx.stroke();

    // Score label on taller bars
    if (bh > 22) {
      ctx.fillStyle = col; ctx.font = 'bold 9px JetBrains Mono'; ctx.textAlign = 'center';
      ctx.fillText(s.score, x+bw/2, y+12);
    }

    // X-axis label
    ctx.fillStyle='#9ca3af'; ctx.font='9px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText('#'+(i+1), x+bw/2, 160-p.b+12);
  });
}

function escHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
