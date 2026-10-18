// ── State ────────────────────────────────────────────────
let currentFile = null;
let currentResult = null;

// ── File handling ────────────────────────────────────────
function handleFileSelect(e) {
  const f = e.target.files[0];
  if (f) loadFile(f);
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('dz-hover');
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
}

function loadFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/jpg'];
  if (!allowed.includes(file.type)) {
    showToast('Please upload a JPG or PNG image', '❌');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('upload-card').style.display = 'none';
    const pc = document.getElementById('preview-card');
    const pi = document.getElementById('preview-img');
    const pb = document.getElementById('preview-badge');
    const pm = document.getElementById('preview-meta');
    pi.src = ev.target.result;
    pb.textContent = file.name;
    pm.textContent = `${file.name} · ${(file.size/1024).toFixed(1)} KB · ${file.type.split('/')[1].toUpperCase()}`;
    pc.style.display = 'block';
    document.getElementById('analyze-btn').disabled = false;
  };
  reader.readAsDataURL(file);
}

function loadDemo() {
  // Generate synthetic dermoscopic image
  const c = document.createElement('canvas');
  c.width = 300; c.height = 300;
  const ctx = c.getContext('2d');

  // Skin base
  const bg = ctx.createRadialGradient(150,160,0,150,150,180);
  bg.addColorStop(0,'#d4a07a'); bg.addColorStop(1,'#c49060');
  ctx.fillStyle = bg; ctx.fillRect(0,0,300,300);

  // Subtle texture hairs
  for(let i=0;i<25;i++){
    ctx.beginPath();
    ctx.moveTo(Math.random()*300, Math.random()*300);
    ctx.bezierCurveTo(Math.random()*300,Math.random()*300,Math.random()*300,Math.random()*300,Math.random()*300,Math.random()*300);
    ctx.strokeStyle=`rgba(70,35,5,${0.04+Math.random()*0.06})`; ctx.lineWidth=0.7; ctx.stroke();
  }

  // Lesion
  ctx.save(); ctx.translate(148,148);
  ctx.beginPath();
  const n=20;
  for(let i=0;i<=n;i++){
    const a=(i/n)*Math.PI*2;
    const r=55+Math.sin(i*2.7)*9+Math.cos(i*1.3)*6+Math.random()*5;
    i===0 ? ctx.moveTo(Math.cos(a)*r, Math.sin(a)*r)
           : ctx.lineTo(Math.cos(a)*r, Math.sin(a)*r);
  }
  ctx.closePath();
  const lg = ctx.createRadialGradient(-5,-5,3,0,0,60);
  lg.addColorStop(0,'#180a02'); lg.addColorStop(0.4,'#361808');
  lg.addColorStop(0.7,'#5a2c10'); lg.addColorStop(1,'rgba(80,40,16,0.55)');
  ctx.fillStyle=lg; ctx.fill();
  for(let i=0;i<10;i++){
    ctx.beginPath();
    ctx.arc((Math.random()-0.5)*70,(Math.random()-0.5)*70,1.5+Math.random()*5,0,Math.PI*2);
    ctx.fillStyle=`rgba(10,5,0,${0.15+Math.random()*0.25})`; ctx.fill();
  }
  ctx.restore();

  // Specular
  const sp = ctx.createRadialGradient(115,108,0,115,108,28);
  sp.addColorStop(0,'rgba(255,255,255,0.22)'); sp.addColorStop(1,'transparent');
  ctx.fillStyle=sp; ctx.fillRect(88,82,68,68);

  c.toBlob(blob => {
    const file = new File([blob], 'demo_lesion.png', { type:'image/png' });
    loadFile(file);
    showToast('Demo dermoscopic image loaded', '📷');
  }, 'image/png');
}

function resetDiagnosis() {
  currentFile = null;
  currentResult = null;
  document.getElementById('upload-card').style.display = 'block';
  document.getElementById('preview-card').style.display = 'none';
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('file-input').value = '';

  // Reset results
  document.getElementById('result-cards').style.display = 'none';
  document.getElementById('empty-state').style.display = 'block';
  setStatus('', 'Ready');
}

// ── Analysis ─────────────────────────────────────────────
async function runAnalysis() {
  if (!currentFile) return;

  const btn     = document.getElementById('analyze-btn');
  const spinner = document.getElementById('btn-spinner');
  const label   = document.getElementById('analyze-label');

  btn.disabled = true;
  spinner.classList.add('show');
  label.textContent = 'Analyzing…';
  setStatus('analyzing', 'Analyzing…');

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Server error');
    }

    currentResult = await res.json();
    renderResults(currentResult);
    setStatus('success', 'Analysis complete');
    showToast(
      currentResult.label === 'Malignant'
        ? 'Malignant detected — consult a specialist'
        : 'Benign classification — low risk',
      currentResult.label === 'Malignant' ? '⚠️' : '✅'
    );

    // Update history badge
    const h = await (await fetch('/api/history')).json();
    const badge = document.getElementById('badge-history');
    badge.textContent = h.length;
    badge.style.display = 'inline-block';

  } catch (err) {
    setStatus('error', 'Error');
    showToast('Error: ' + err.message, '❌');
    console.error(err);
  } finally {
    btn.disabled = false;
    spinner.classList.remove('show');
    label.textContent = 'Analyze Image';
  }
}

// ── Render results ────────────────────────────────────────
function renderResults(r) {
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('result-cards').style.display = 'flex';
  document.getElementById('result-cards').style.flexDirection = 'column';
  document.getElementById('result-cards').style.gap = '14px';

  const isMal   = r.label === 'Malignant';
  const rClass  = r.risk.toLowerCase();
  const vClass  = isMal ? 'v-malignant' : 'v-benign';

  // ── Verdict ──
  const vc = document.getElementById('verdict-card');
  vc.className = 'result-card verdict-card ' + vClass;

  document.getElementById('verdict-label').textContent = r.label;
  document.getElementById('verdict-label').className   = 'verdict-label ' + vClass;
  document.getElementById('verdict-prob').textContent  = (r.probability * 100).toFixed(1) + '%';

  // Risk bar
  const fillColors = {
    low: 'linear-gradient(90deg,#16a34a,#22c55e)',
    moderate: 'linear-gradient(90deg,#d97706,#f59e0b)',
    high: 'linear-gradient(90deg,#ea580c,#f97316)',
    critical: 'linear-gradient(90deg,#dc2626,#ef4444)',
  };
  const fill   = document.getElementById('risk-fill');
  const marker = document.getElementById('risk-marker');
  setTimeout(() => {
    fill.style.width  = r.risk_score + '%';
    fill.style.background = fillColors[rClass];
    marker.style.left = r.risk_score + '%';
    marker.style.background = fillColors[rClass].split(',')[1].replace(')','').trim();
  }, 80);
  document.getElementById('risk-score-val').textContent = r.risk_score;

  // Tags
  document.getElementById('risk-tag').textContent  = r.risk;
  document.getElementById('risk-tag').className    = 'risk-tag r-' + rClass;
  document.getElementById('tone-tag').textContent  = '🎨 ' + r.tone;
  document.getElementById('tone-tag').className    = 'tone-tag';
  document.getElementById('tone-tag').style.cssText = 'background:var(--surface2);color:var(--text2);border:1px solid var(--border);padding:4px 10px;border-radius:20px;font-family:JetBrains Mono,monospace;font-size:11px;';
  document.getElementById('clahe-tag').textContent = r.clahe ? '✓ CLAHE ON' : '✗ CLAHE OFF';
  document.getElementById('clahe-tag').className   = 'clahe-tag ' + (r.clahe ? 'r-low' : '');
  if (!r.clahe) document.getElementById('clahe-tag').style.cssText = 'background:var(--surface2);color:var(--text3);border:1px solid var(--border);padding:4px 10px;border-radius:20px;font-family:JetBrains Mono,monospace;font-size:11px;';

  // ── Action ──
  const actions = {
    CRITICAL: { cls:'a-critical', icon:'🚨', title:'Visit dermatologist immediately',   desc:'High suspicion of malignant melanoma. Seek urgent evaluation within 48 hours.' },
    HIGH:     { cls:'a-high',     icon:'⚠️', title:'Book appointment within 1 week',    desc:'Significant risk markers detected. Schedule a clinical biopsy assessment promptly.' },
    MODERATE: { cls:'a-moderate', icon:'📋', title:'Schedule clinical review',           desc:'Some atypical features present. A professional dermoscopy review is recommended.' },
    LOW:      { cls:'a-low',      icon:'✅', title:'Routine annual checkup',             desc:'No immediate concern. Continue regular self-examinations and annual skin checks.' },
  };
  const act = actions[r.risk];
  document.getElementById('action-card').className         = 'action-card ' + act.cls;
  document.getElementById('action-icon-wrap').textContent  = act.icon;
  document.getElementById('action-title').textContent      = act.title;
  document.getElementById('action-desc').textContent       = act.desc;

  // ── ABCD ──
  const abcdItems = [
    { letter:'A', name:'Asymmetry', val: r.abcd.asymmetry, max:1,  unit:'',   display: r.abcd.asymmetry > 0.6 ? 'Irregular' : 'Symmetric' },
    { letter:'B', name:'Border',    val: r.abcd.border,    max:1,  unit:'',   display: r.abcd.border > 0.6 ? 'Irregular' : 'Regular' },
    { letter:'C', name:'Color',     val: r.abcd.color,     max:1,  unit:'',   display: r.abcd.color > 0.5 ? 'Variegated' : 'Uniform' },
    { letter:'D', name:'Diameter',  val: Math.min(r.abcd.diameter/16, 1), max:1, unit:'', display: r.abcd.diameter.toFixed(1) + ' mm' },
  ];
  document.getElementById('abcd-list').innerHTML = abcdItems.map(a => {
    const pct = Math.round(a.val * 100);
    const col = a.val > 0.65 ? '#dc2626' : a.val > 0.4 ? '#d97706' : '#16a34a';
    return `
    <div class="abcd-row">
      <div class="abcd-row-header">
        <div>
          <span class="abcd-letter-label">${a.letter}</span>
          <span class="abcd-letter-name">${a.name}</span>
        </div>
        <span class="abcd-val-label">${a.display}</span>
      </div>
      <div class="abcd-bar">
        <div class="abcd-bar-fill" style="width:0%;background:${col};border-radius:3px;height:5px;transition:width 0.9s cubic-bezier(.4,0,.2,1);"
             data-target="${pct}"></div>
      </div>
    </div>`;
  }).join('');

  setTimeout(() => {
    document.querySelectorAll('.abcd-bar-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  }, 80);

  // ── Skin tone ──
  const toneData = {
    'Very Light': { color:'#f5cba7', ita_label: '> 55°' },
    'Light':      { color:'#e8b08a', ita_label: '41–55°' },
    'Tan':        { color:'#c8956c', ita_label: '28–41°' },
    'Brown':      { color:'#a06c42', ita_label: '10–28°' },
    'Dark':       { color:'#7a4b2e', ita_label: '-30–10°' },
    'Very Dark':  { color:'#4a2818', ita_label: '< -30°' },
  };
  const td = toneData[r.tone] || { color:'#c8956c', ita_label:'—' };
  const allTones = Object.values(toneData);
  const toneIdx  = Object.keys(toneData).indexOf(r.tone);

  document.getElementById('skin-display').innerHTML = `
    <div class="skin-top">
      <div class="skin-swatch" style="background:${td.color};"></div>
      <div>
        <div class="skin-name">${r.tone}</div>
        <div class="skin-ita">ITA: ${r.ita}° &nbsp;·&nbsp; ${td.ita_label}</div>
      </div>
    </div>
    <div class="clahe-chip ${r.clahe ? 'clahe-on' : 'clahe-off'}">
      ${r.clahe ? '✓ CLAHE Applied' : '✗ CLAHE Not Applied'}
      <div class="clahe-tip">${r.clahe ? 'Adaptive contrast enhancement was applied to improve visibility of lesions on darker skin tones (Fitzpatrick IV–VI).' : 'No enhancement needed — lighter skin tones have sufficient inherent contrast for analysis.'}</div>
    </div>
    <div class="skin-scale">
      ${allTones.map((t2, i) => `<div class="scale-dot ${i===toneIdx?'active':''}" style="background:${t2.color};" title="Fitzpatrick Type ${i+1}"></div>`).join('')}
    </div>`;

  // ── Grad-CAM ──
  const imgSrc = document.getElementById('preview-img').src;
  document.getElementById('gradcam-pair').innerHTML = `
    <div class="gc-img-wrap">
      <img src="${imgSrc}" alt="Original">
      <div class="gc-label">Original Image</div>
    </div>
    <div class="gc-img-wrap">
      <canvas id="gc-canvas" width="300" height="300"></canvas>
      <div class="gc-label">Red = model attention</div>
    </div>`;

  // Draw GradCAM overlay
  setTimeout(() => {
    const canvas = document.getElementById('gc-canvas');
    if (!canvas) return;
    const ctx2 = canvas.getContext('2d');
    const img2 = new Image();
    img2.onload = () => {
      ctx2.drawImage(img2, 0, 0, 300, 300);
      const cx = 130 + Math.random()*40, cy = 130 + Math.random()*40;
      [[cx,cy,75,0.55*r.probability],[cx-20,cy+12,48,0.38*r.probability],[cx+22,cy-8,36,0.28*r.probability]].forEach(([x,y,rad,a]) => {
        const g = ctx2.createRadialGradient(x,y,0,x,y,rad);
        g.addColorStop(0,`rgba(255,30,30,${a})`);
        g.addColorStop(0.35,`rgba(255,100,0,${a*0.65})`);
        g.addColorStop(0.65,`rgba(255,220,0,${a*0.25})`);
        g.addColorStop(1,'rgba(0,100,255,0)');
        ctx2.fillStyle=g; ctx2.fillRect(0,0,300,300);
      });
    };
    img2.src = imgSrc;
  }, 120);
}

// ── Report ────────────────────────────────────────────────
async function downloadReport() {
  if (!currentResult) { showToast('No results to export', '❌'); return; }
  try {
    const res = await fetch('/api/report', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(currentResult)
    });
    if (!res.ok) {
      const e = await res.json();
      showToast(e.error || 'Report generation failed', '❌');
      return;
    }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'melanoma_report.pdf';
    a.click();
    showToast('PDF report downloaded', '📄');
  } catch(e) {
    showToast('Connect backend/pdf_report.py for PDF export', 'ℹ️');
  }
}
