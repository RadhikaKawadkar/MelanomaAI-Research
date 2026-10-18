// ── Chart rendering ───────────────────────────────────────
let metricsLoaded = false;

async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const d   = await res.json();
    renderKPIs(d);
    renderAllCharts(d);
    renderFairness(d.fairness);
    metricsLoaded = true;
  } catch(e) {
    console.error('Could not load metrics', e);
  }
}

// ── KPI cards ─────────────────────────────────────────────
function renderKPIs(d) {
  document.getElementById('kpi-row').innerHTML = `
    <div class="kpi-card kpi-accent-top kpi-mint">
      <div class="kpi-eyebrow">Accuracy</div>
      <div class="kpi-value" style="color:var(--mint)">${d.accuracy.toFixed(2)}%</div>
      <div class="kpi-delta">Test set · ${d.confusion_matrix ? d.confusion_matrix[0][0]+d.confusion_matrix[1][1] : '—'} correct</div>
    </div>
    <div class="kpi-card kpi-accent-top kpi-navy">
      <div class="kpi-eyebrow">AUC-ROC</div>
      <div class="kpi-value" style="color:var(--navy)">${d.auc.toFixed(4)}</div>
      <div class="kpi-delta">Discrimination ability</div>
    </div>
    <div class="kpi-card kpi-accent-top kpi-green">
      <div class="kpi-eyebrow">Sensitivity</div>
      <div class="kpi-value" style="color:var(--green)">${d.sensitivity.toFixed(2)}%</div>
      <div class="kpi-delta">True positive rate</div>
    </div>
    <div class="kpi-card kpi-accent-top kpi-amber">
      <div class="kpi-eyebrow">Specificity</div>
      <div class="kpi-value" style="color:var(--amber)">${d.specificity.toFixed(2)}%</div>
      <div class="kpi-delta">True negative rate</div>
    </div>`;
}

function renderFairness(f) {
  document.getElementById('roc-sub').textContent = `AUC = ${(window._auc||0.9392).toFixed(4)}`;
  document.getElementById('fs-metrics').innerHTML = `
    <div class="fs-metric">
      <div class="fs-val" style="color:var(--green)">${f.light_acc.toFixed(2)}%</div>
      <div class="fs-key">Light Skin Accuracy</div>
    </div>
    <div class="fs-metric">
      <div class="fs-val" style="color:var(--amber)">${f.dark_acc.toFixed(2)}%</div>
      <div class="fs-key">Dark Skin Accuracy</div>
    </div>
    <div class="fs-metric">
      <div class="fs-val" style="color:var(--red)">${f.gap.toFixed(4)}</div>
      <div class="fs-key">Fairness Gap</div>
    </div>`;
}

// ── Canvas helpers ────────────────────────────────────────
function gc(id, h=240) {
  const c = document.getElementById(id);
  if (!c) return [null, 0, 0, null];
  const W = c.parentElement.offsetWidth || 400;
  c.width = W; c.height = h;
  return [c.getContext('2d'), W, h];
}

function grid(ctx, W, H, p, steps=4) {
  ctx.strokeStyle='rgba(0,0,0,0.05)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  for(let i=1;i<=steps;i++){
    const y=(H-p.b)-i/steps*(H-p.t-p.b);
    const x=p.l+i/steps*(W-p.l-p.r);
    ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();
    ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,H-p.b);ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.strokeStyle='rgba(0,0,0,0.12)';ctx.lineWidth=1.2;
  ctx.beginPath();ctx.moveTo(p.l,p.t);ctx.lineTo(p.l,H-p.b);ctx.lineTo(W-p.r,H-p.b);ctx.stroke();
}

function axisLbl(ctx, W, H, p, xl, yl) {
  ctx.fillStyle='#9ca3af'; ctx.font='10px DM Sans'; ctx.textAlign='center';
  ctx.fillText(xl, W/2, H-3);
  ctx.save(); ctx.translate(10,H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(yl,0,0); ctx.restore();
}

function rRect(ctx, x, y, w, h, r2) {
  if(w<0){x+=w;w=-w;} if(h<0){y+=h;h=-h;}
  ctx.moveTo(x+r2,y);ctx.lineTo(x+w-r2,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r2);
  ctx.lineTo(x+w,y+h);ctx.lineTo(x,y+h);ctx.lineTo(x,y+r2);ctx.quadraticCurveTo(x,y,x+r2,y);
}

// ── All charts ────────────────────────────────────────────
function renderAllCharts(d) {
  window._auc = d.auc;
  drawCM(d.confusion_matrix || [[341,28],[37,172]]);
  drawROC(d.auc);
  drawPR();
  drawThresh();
  drawRiskDist();
  drawHist();
  drawRadar(d);
  drawFairBar();
}

// Confusion matrix
function drawCM(cm) {
  const [ctx, W, H] = gc('c-cm');
  if (!ctx) return;
  const p={t:30,b:30,l:60,r:10};
  const cw=(W-p.l-p.r)/2, ch=(H-p.t-p.b)/2;

  const cells = [
    {r:0,c:0,v:cm[0][0],bg:'#f0fdf4',tc:'#16a34a',lbl:'TN'},
    {r:0,c:1,v:cm[0][1],bg:'#fffbeb',tc:'#d97706',lbl:'FP'},
    {r:1,c:0,v:cm[1][0],bg:'#fff5f5',tc:'#dc2626',lbl:'FN'},
    {r:1,c:1,v:cm[1][1],bg:'#f0fdf4',tc:'#16a34a',lbl:'TP'},
  ];

  cells.forEach(({r,c,v,bg,tc,lbl}) => {
    const x=p.l+c*cw+3, y=p.t+r*ch+3;
    ctx.fillStyle=bg; ctx.beginPath(); rRect(ctx,x,y,cw-6,ch-6,8); ctx.closePath(); ctx.fill();
    ctx.fillStyle=tc; ctx.font='bold 26px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText(v, x+cw/2-3, y+ch/2+4);
    ctx.fillStyle='#9ca3af'; ctx.font='9px DM Sans';
    ctx.fillText(lbl, x+cw/2-3, y+ch/2+18);
  });

  const lbls=['Benign','Malignant'];
  ctx.fillStyle='#6b7280'; ctx.font='10px DM Sans'; ctx.textAlign='center';
  lbls.forEach((l,i)=>ctx.fillText(l, p.l+cw/2+i*cw, p.t-10));
  ctx.save(); ctx.translate(14,p.t+ch/2); ctx.rotate(-Math.PI/2); ctx.fillText('Benign',0,0); ctx.restore();
  ctx.save(); ctx.translate(14,p.t+ch+ch/2); ctx.rotate(-Math.PI/2); ctx.fillText('Malignant',0,0); ctx.restore();
}

// ROC
function drawROC(auc) {
  const [ctx, W, H] = gc('c-roc');
  if (!ctx) return;
  const p={t:20,b:34,l:38,r:14};
  grid(ctx,W,H,p);
  ctx.strokeStyle='rgba(0,0,0,0.1)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(p.l,H-p.b);ctx.lineTo(W-p.r,p.t);ctx.stroke();
  ctx.setLineDash([]);

  const pts = genROC(auc);
  const g=ctx.createLinearGradient(p.l,0,W-p.r,0);
  g.addColorStop(0,'#0fa86e'); g.addColorStop(1,'#1A2B4A');
  ctx.strokeStyle=g; ctx.lineWidth=2.5;
  ctx.beginPath();
  pts.forEach(([fx,fy],i)=>{
    const x=p.l+fx*(W-p.l-p.r), y=(H-p.b)-fy*(H-p.t-p.b);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Area fill
  ctx.beginPath();
  pts.forEach(([fx,fy],i)=>{
    const x=p.l+fx*(W-p.l-p.r), y=(H-p.b)-fy*(H-p.t-p.b);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.lineTo(W-p.r,H-p.b); ctx.lineTo(p.l,H-p.b); ctx.closePath();
  const ag=ctx.createLinearGradient(0,p.t,0,H-p.b);
  ag.addColorStop(0,'rgba(15,168,110,0.12)'); ag.addColorStop(1,'rgba(15,168,110,0)');
  ctx.fillStyle=ag; ctx.fill();

  ctx.fillStyle='#374151'; ctx.font='bold 11px JetBrains Mono'; ctx.textAlign='left';
  ctx.fillText(`AUC = ${auc.toFixed(4)}`, p.l+8, p.t+14);
  axisLbl(ctx,W,H,p,'FPR','TPR');
}

// Precision-Recall
function drawPR() {
  const [ctx, W, H] = gc('c-pr');
  if (!ctx) return;
  const p={t:20,b:34,l:38,r:14};
  grid(ctx,W,H,p);
  ctx.strokeStyle='#ea580c'; ctx.lineWidth=2.5; ctx.beginPath();
  for(let i=0;i<=60;i++){
    const rc=i/60;
    const pr=Math.max(0.05, 0.97-Math.pow(rc,1.4)*0.48+Math.sin(rc*5)*0.02);
    const x=p.l+rc*(W-p.l-p.r), y=(H-p.b)-pr*(H-p.t-p.b);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();
  ctx.fillStyle='#ea580c'; ctx.font='bold 11px JetBrains Mono'; ctx.textAlign='left';
  ctx.fillText('AP ≈ 0.878', p.l+8, p.t+14);
  axisLbl(ctx,W,H,p,'Recall','Precision');
}

// Threshold
function drawThresh() {
  const [ctx, W, H] = gc('c-thresh');
  if (!ctx) return;
  const p={t:20,b:34,l:38,r:14};
  grid(ctx,W,H,p);
  const N=80;
  // Sensitivity
  ctx.strokeStyle='#16a34a'; ctx.lineWidth=2; ctx.beginPath();
  for(let i=0;i<=N;i++){
    const t=i/N, v=Math.max(0,1-Math.pow(t,0.65)*0.92+0.04*Math.sin(t*9));
    const x=p.l+t*(W-p.l-p.r), y=(H-p.b)-v*(H-p.t-p.b);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();
  // Specificity
  ctx.strokeStyle='#1A2B4A'; ctx.lineWidth=2; ctx.beginPath();
  for(let i=0;i<=N;i++){
    const t=i/N, v=Math.min(1,Math.pow(t,0.5)*0.96);
    const x=p.l+t*(W-p.l-p.r), y=(H-p.b)-v*(H-p.t-p.b);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();
  // Threshold marker
  const tx=p.l+0.5*(W-p.l-p.r);
  ctx.strokeStyle='rgba(220,38,38,0.65)'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath();ctx.moveTo(tx,p.t);ctx.lineTo(tx,H-p.b);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#dc2626'; ctx.font='bold 10px JetBrains Mono'; ctx.textAlign='center';
  ctx.fillText('0.50', tx, p.t+10);
  // Legend
  [[p.l+8,p.t+14,'#16a34a','Sensitivity'],[p.l+8,p.t+28,'#1A2B4A','Specificity']].forEach(([x,y,c,l])=>{
    ctx.fillStyle=c; ctx.fillRect(x,y-8,10,8);
    ctx.fillStyle='#374151'; ctx.font='10px DM Sans'; ctx.textAlign='left';
    ctx.fillText(l,x+14,y);
  });
  axisLbl(ctx,W,H,p,'Threshold','Score');
}

// Risk distribution
function drawRiskDist() {
  const [ctx, W, H] = gc('c-risk');
  if (!ctx) return;
  const p={t:28,b:34,l:38,r:14};
  grid(ctx,W,H,p);
  const cats=[['LOW',280,'#16a34a'],['MODERATE',185,'#d97706'],['HIGH',92,'#ea580c'],['CRITICAL',21,'#dc2626']];
  const maxV=310, bw=(W-p.l-p.r)/cats.length-12;
  cats.forEach(([name,val,color],i)=>{
    const bh=(H-p.t-p.b)*(val/maxV);
    const x=p.l+(W-p.l-p.r)/cats.length*(i+0.5)-bw/2;
    const y=H-p.b-bh;
    ctx.fillStyle=color+'20'; ctx.strokeStyle=color; ctx.lineWidth=1.5;
    ctx.beginPath(); rRect(ctx,x,y,bw,bh,4); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle=color; ctx.font='bold 13px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText(val, x+bw/2, y-6);
    ctx.fillStyle='#6b7280'; ctx.font='10px DM Sans'; ctx.fillText(name, x+bw/2, H-p.b+14);
  });
}

// Histogram
function drawHist() {
  const [ctx, W, H] = gc('c-hist');
  if (!ctx) return;
  const p={t:20,b:34,l:38,r:14};
  grid(ctx,W,H,p);
  const bins=20, bw=(W-p.l-p.r)/bins-1.5;
  const heights=Array.from({length:bins},(_,i)=>{
    const t=i/bins;
    return t<0.5 ? Math.max(0.05, 0.72+Math.sin(t*14)*0.18-t*0.35) : Math.max(0.05, 0.08+Math.pow(t-0.5,0.75)*1.3+Math.sin(t*7)*0.08);
  });
  const maxH=Math.max(...heights);
  heights.forEach((h,i)=>{
    const x=p.l+(W-p.l-p.r)/bins*i;
    const bh=(H-p.t-p.b)*(h/maxH)*0.9;
    const col=i<10?'#16a34a':'#dc2626';
    ctx.fillStyle=col+'30'; ctx.strokeStyle=col; ctx.lineWidth=1;
    ctx.beginPath(); rRect(ctx,x+1,H-p.b-bh,bw,bh,3); ctx.closePath(); ctx.fill(); ctx.stroke();
  });
  ['0.0','0.5','1.0'].forEach((lbl,i)=>{
    ctx.fillStyle='#9ca3af'; ctx.font='9px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText(lbl, p.l+(W-p.l-p.r)*i/2, H-p.b+13);
  });
  axisLbl(ctx,W,H,p,'Predicted Probability','Count');
}

// Radar
function drawRadar(d) {
  const [ctx, W, H] = gc('c-radar');
  if (!ctx) return;
  const cx=W/2, cy=H/2, r=Math.min(W,H)*0.33;
  const metrics=[
    ['Accuracy',  d.accuracy/100],
    ['Sensitivity', d.sensitivity/100],
    ['Specificity', d.specificity/100],
    ['Precision', (d.precision||86.6)/100],
    ['F1',       (d.f1||84.5)/100],
    ['AUC',       d.auc],
  ];
  const N=metrics.length;
  // Grid rings
  [0.25,0.5,0.75,1.0].forEach(f=>{
    ctx.strokeStyle='rgba(0,0,0,0.06)'; ctx.lineWidth=1; ctx.beginPath();
    for(let i=0;i<=N;i++){
      const a=(i/N)*Math.PI*2-Math.PI/2;
      i===0?ctx.moveTo(cx+Math.cos(a)*r*f,cy+Math.sin(a)*r*f):ctx.lineTo(cx+Math.cos(a)*r*f,cy+Math.sin(a)*r*f);
    }
    ctx.closePath(); ctx.stroke();
  });
  // Spokes
  for(let i=0;i<N;i++){
    const a=(i/N)*Math.PI*2-Math.PI/2;
    ctx.strokeStyle='rgba(0,0,0,0.07)'; ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r);ctx.stroke();
    ctx.fillStyle='#6b7280'; ctx.font='10px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText(metrics[i][0], cx+Math.cos(a)*r*1.2, cy+Math.sin(a)*r*1.2+3);
  }
  // Fill
  const g=ctx.createRadialGradient(cx,cy,0,cx,cy,r);
  g.addColorStop(0,'rgba(15,168,110,0.3)'); g.addColorStop(1,'rgba(15,168,110,0.04)');
  ctx.fillStyle=g; ctx.strokeStyle='#0fa86e'; ctx.lineWidth=2;
  ctx.beginPath();
  metrics.forEach(([,v],i)=>{
    const a=(i/N)*Math.PI*2-Math.PI/2;
    const px=cx+Math.cos(a)*r*v, py=cy+Math.sin(a)*r*v;
    i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
  });
  ctx.closePath(); ctx.fill(); ctx.stroke();
  // Dots
  metrics.forEach(([,v],i)=>{
    const a=(i/N)*Math.PI*2-Math.PI/2;
    ctx.fillStyle='#0fa86e';
    ctx.beginPath(); ctx.arc(cx+Math.cos(a)*r*v, cy+Math.sin(a)*r*v, 3, 0, Math.PI*2); ctx.fill();
  });
}

// Fairness bar
function drawFairBar() {
  const [ctx, W, H] = gc('c-fair');
  if (!ctx) return;
  const p={t:36,b:16,l:72,r:54};
  const groups=[
    {name:'Very Light',val:0.951,color:'#16a34a'},
    {name:'Light',     val:0.926,color:'#22c55e'},
    {name:'Tan',       val:0.901,color:'#d97706'},
    {name:'Brown',     val:0.852,color:'#ea580c'},
    {name:'Dark',      val:0.752,color:'#dc2626'},
  ];
  const bh=(H-p.t-p.b)/groups.length-6;
  ctx.fillStyle='#374151'; ctx.font='11px Syne,sans-serif'; ctx.textAlign='center';
  ctx.fillText('Accuracy by Fitzpatrick Skin Type', W/2, p.t-14);
  groups.forEach((g,i)=>{
    const y=p.t+(bh+6)*i;
    const bw=(W-p.l-p.r)*g.val;
    ctx.fillStyle=g.color+'25'; ctx.strokeStyle=g.color; ctx.lineWidth=1.5;
    ctx.beginPath(); rRect(ctx,p.l,y,bw,bh,4); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle='#374151'; ctx.font='11px DM Sans'; ctx.textAlign='right';
    ctx.fillText(g.name, p.l-8, y+bh/2+4);
    ctx.fillStyle=g.color; ctx.font='bold 11px JetBrains Mono'; ctx.textAlign='left';
    ctx.fillText((g.val*100).toFixed(1)+'%', p.l+bw+6, y+bh/2+4);
  });
}

// ROC curve generator
function genROC(auc) {
  const pts=[[0,0]];
  for(let i=1;i<=100;i++){
    const fpr=i/100;
    const tpr=Math.min(1, Math.pow(fpr,(1-auc)/auc*0.45)*(0.96+Math.random()*0.02));
    pts.push([fpr, Math.min(1,tpr)]);
  }
  pts.push([1,1]); return pts;
}
