import json
DATA = open("attr_analysis/mixer_data.json").read()
HTML = r"""<title>Persona mixer</title>
<style>
:root{--surface:#fbfaf7;--raised:#f4f2ec;--ink:#14120e;--muted:#6d6a61;--line:#e2dfd5;
 --blue:#2a6fc9;--hot:#a3231a;--ok:#1f7a4d;--mono:ui-monospace,"SF Mono",Menlo,monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;--line:#2d2b26;
 --blue:#63a2ee;--hot:#f0736a;--ok:#4ec98a;}}
:root[data-theme="dark"]{--surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;
 --line:#2d2b26;--blue:#63a2ee;--hot:#f0736a;--ok:#4ec98a;}
*{box-sizing:border-box}
body{background:var(--surface);color:var(--ink);margin:0;font:15px/1.55 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:30px 20px 60px}
h1{font-size:22px;margin:0 0 5px;font-weight:650;letter-spacing:-.015em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 18px;max-width:70ch}
.cols{display:grid;grid-template-columns:1fr 300px;gap:20px}
@media(max-width:840px){.cols{grid-template-columns:1fr}}
.stage{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;touch-action:none}
canvas{display:block;width:100%;cursor:grab}canvas.drag{cursor:grabbing}
.panel{border:1px solid var(--line);border-radius:12px;padding:13px;background:var(--raised);
 max-height:520px;overflow-y:auto}
.lab{color:var(--muted);font-family:var(--mono);font-size:10px;letter-spacing:.07em;
 text-transform:uppercase;margin-bottom:7px}
.row{display:flex;align-items:center;gap:7px;margin-bottom:3px;font-size:11.5px;font-family:var(--mono)}
.row span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row input[type=range]{width:78px}
#filter{width:100%;margin-bottom:9px;padding:5px 8px;font-family:var(--mono);font-size:11.5px;
 border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)}
#active{border-bottom:1px solid var(--line);margin-bottom:7px;padding-bottom:5px}
#active:empty{display:none}
.row.hide{display:none}
.row b{width:38px;text-align:right;font-variant-numeric:tabular-nums;font-weight:500}
.res{margin-top:14px;border:1px solid var(--line);border-radius:12px;padding:14px}
table{border-collapse:collapse;font-family:var(--mono);font-size:11.5px;width:100%;
 font-variant-numeric:tabular-nums}
th,td{padding:3px 7px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:400;font-size:10px}
td:first-child,th:first-child{text-align:left}
.bar{height:7px;border-radius:4px;background:var(--blue);opacity:.75}
button{font-family:var(--mono);font-size:11px;padding:4px 9px;border:1px solid var(--line);
 background:var(--surface);color:var(--muted);border-radius:6px;cursor:pointer}
button:hover{color:var(--ink)}
button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.hd{display:flex;gap:6px;align-items:center;margin-bottom:9px;flex-wrap:wrap}
.big{font-size:20px;font-weight:650}
.note{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;
 margin-top:24px;max-width:76ch}
.note b{color:var(--ink)}
</style>
<div class="wrap">
<h1>Persona mixer</h1>
<p class="sub">Build a direction out of other personas and ask the model's own geometry who you have made,
without generating a word. All 300 personas from the paper's role set, at three layers. The
oracle names a held-out response correctly <b>23%</b> of the time out of 301 candidates
&mdash; 70&times; chance, but a hint rather than a verdict. Watch the Mahalanobis column: a
nonsense mixture still gets a confident-looking percentage while sitting far outside every
cloud.</p>

<div class="hd">
  <span class="lab" style="margin:0 8px 0 0">layer</span><span id="layers"></span>
  <button id="clear">clear</button>
  <button data-preset="old_musician">retiree+musician</button>
  <button data-preset="young_musician">musician+teenager</button>
  <button data-preset="medieval_spy">ancient+spy+pirate</button>
  <button data-preset="old_soldier">soldier+grandparent</button>
  <button data-preset="analogy">young_musician+(old−young)</button>
  <button data-preset="vampire_ish">ghost+pirate+ancient</button>
</div>

<div class="cols">
  <div>
    <div class="stage"><canvas id="c"></canvas></div>
    <div class="res">
      <div class="lab">what did you build?</div>
      <div id="verdict" class="big">nothing yet</div>
      <table id="post"><thead><tr><th>persona</th><th>posterior</th><th>mahal</th><th></th></tr></thead>
      <tbody></tbody></table>
    </div>
  </div>
  <div class="panel">
    <div class="lab">mix (drag sliders)</div>
    <input id="filter" placeholder="filter 300 personas..." />
    <div id="active"></div>
    <div id="sliders"></div>
  </div>
</div>

<p class="note"><b>Why this is not circular.</b> Steering straight at a persona's own mean
trivially scores 1.00 and teaches nothing. The question worth asking is whether a persona can
be reached out of <em>other</em> personas &mdash; so the sliders are the ingredients and the
oracle is the judge. A 96% reconstruction from the other 40 personas says most of them can.</p>
</div>
<script>
const D = __DATA__;
const K=D.k, NAMES=D.names, cv=document.getElementById('c'), ctx=cv.getContext('2d');
let LAY=D.layers[1];
const LD=()=>D.L[LAY];
let w={}, yaw=-0.9, pitch=0.32, zoom=1, W=0,H=0,scale=1, drag=null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function mixVec(){const v=new Array(K).fill(0);
  for(const n in w){if(!w[n])continue;const d=LD().dirs[n];for(let i=0;i<K;i++)v[i]+=w[n]*d[i];}
  return v;}
function score(v){
  const out=[];
  for(const p of D.personas){const g=LD().gauss[p];let q=0;
    const d=v.map((x,i)=>x-g.mu[i]);
    for(let i=0;i<K;i++){let s=0;const Si=g.Si[i];for(let j=0;j<K;j++)s+=Si[j]*d[j];q+=d[i]*s;}
    out.push({p,ll:-0.5*(q+g.logdet),m:Math.sqrt(Math.max(q,0))});}
  const mx=Math.max(...out.map(o=>o.ll)); let z=0;
  out.forEach(o=>{o.e=Math.exp(o.ll-mx);z+=o.e;});
  out.forEach(o=>o.post=o.e/z);
  return out.sort((a,b)=>b.post-a.post);}
function to3(v){const E=LD().E3;const o=[0,0,0];
  for(let i=0;i<K;i++)for(let j=0;j<3;j++)o[j]+=v[i]*E[i][j];return o;}
function proj(p){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const x=p[0]*cy-p[2]*sy,z=p[0]*sy+p[2]*cy,y=p[1]*cp-z*sp,d=p[1]*sp+z*cp;
  return [W/2+x*scale*zoom,H/2-y*scale*zoom,d];}
function fit(){W=cv.width=cv.clientWidth*devicePixelRatio;
  H=cv.height=Math.round(cv.clientWidth*0.62)*devicePixelRatio;
  cv.style.height=Math.round(cv.clientWidth*0.62)+'px';
  let m=1;for(const p in LD().view)m=Math.max(m,Math.hypot(...LD().view[p]));
  scale=Math.min(W,H)/(m*2.6);}
function draw(){
  const ink=css('--ink'),mut=css('--muted'),sur=css('--surface');
  ctx.clearRect(0,0,W,H);
  const items=[];
  for(const p of D.personas){if(p==='default')continue;
    const P=proj(LD().view[p]);items.push({z:P[2],f:()=>{
      ctx.beginPath();ctx.arc(P[0],P[1],3.2*devicePixelRatio,0,6.2832);
      ctx.fillStyle=mut;ctx.globalAlpha=.5;ctx.fill();ctx.globalAlpha=1;}});}
  const O=proj([0,0,0]);
  items.push({z:O[2],f:()=>{ctx.beginPath();ctx.arc(O[0],O[1],5*devicePixelRatio,0,6.2832);
    ctx.fillStyle=css('--blue');ctx.fill();ctx.strokeStyle=sur;ctx.lineWidth=1.4*devicePixelRatio;ctx.stroke();}});
  const v=mixVec(), any=Object.values(w).some(x=>x);
  if(any){const M=proj(to3(v));
    items.push({z:1e9,f:()=>{
      ctx.beginPath();ctx.moveTo(O[0],O[1]);ctx.lineTo(M[0],M[1]);
      ctx.strokeStyle=css('--hot');ctx.lineWidth=2.4*devicePixelRatio;ctx.stroke();
      ctx.beginPath();ctx.arc(M[0],M[1],7*devicePixelRatio,0,6.2832);
      ctx.fillStyle=css('--hot');ctx.fill();ctx.strokeStyle=sur;
      ctx.lineWidth=1.6*devicePixelRatio;ctx.stroke();}});
    const s=score(v), top=s[0];
    document.getElementById('verdict').textContent =
      `${top.p}  ${(top.post*100).toFixed(0)}%`;
    document.getElementById('verdict').style.color =
      top.post>0.9?css('--ok'):(top.post>0.5?ink:mut);
    const tb=document.querySelector('#post tbody');tb.innerHTML='';
    s.slice(0,6).forEach(o=>{const tr=document.createElement('tr');
      tr.innerHTML=`<td>${o.p}</td><td>${(o.post*100).toFixed(1)}%</td>`+
        `<td>${o.m.toFixed(1)}</td><td style="width:70px">`+
        `<div class="bar" style="width:${Math.max(2,o.post*66)}px"></div></td>`;
      tb.appendChild(tr);});
    // label the nearest few personas in the view
    s.slice(0,3).forEach(o=>{if(o.p==='default')return;const P=proj(LD().view[o.p]);
      items.push({z:P[2]+1e5,f:()=>{ctx.fillStyle=ink;ctx.textAlign='center';ctx.lineJoin='round';
        ctx.font=`600 ${10.5*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
        ctx.strokeStyle=sur;ctx.lineWidth=3*devicePixelRatio;
        ctx.strokeText(o.p,P[0],P[1]-9*devicePixelRatio);
        ctx.fillText(o.p,P[0],P[1]-9*devicePixelRatio);}});});
  } else {
    document.getElementById('verdict').textContent='nothing yet';
    document.getElementById('verdict').style.color=mut;
    document.querySelector('#post tbody').innerHTML='';}
  items.sort((a,b)=>a.z-b.z).forEach(i=>i.f());}
const box=document.getElementById('sliders');
NAMES.forEach(n=>{w[n]=0;
  const r=document.createElement('div');r.className='row';
  r.innerHTML=`<span title="${n}">${n}</span>`;
  const i=document.createElement('input');
  i.type='range';i.min=-100;i.max=100;i.value=0;i.step=5;
  const b=document.createElement('b');b.textContent='0';
  i.oninput=()=>{w[n]=+i.value/100;b.textContent=(+i.value/100).toFixed(2);draw();};
  i.dataset.n=n;r.dataset.name=n;r.appendChild(i);r.appendChild(b);box.appendChild(r);});
const filt=document.getElementById('filter');
filt.oninput=()=>{const q=filt.value.toLowerCase();
  box.querySelectorAll('.row').forEach(r=>
    r.classList.toggle('hide', q && !r.dataset.name.includes(q)));};
function setW(obj){for(const n in w)w[n]=0;Object.assign(w,obj);
  box.querySelectorAll('input').forEach(i=>{i.value=Math.round((w[i.dataset.n]||0)*100);
    i.nextSibling.textContent=(w[i.dataset.n]||0).toFixed(2);});draw();}
document.getElementById('clear').onclick=()=>setW({});
const PRESETS={
  old_musician:{retiree:0.54,musician:0.43,consultant:-0.20},
  young_musician:{musician:0.48,teenager:0.45,philosopher:-0.13},
  medieval_spy:{ancient:0.55,spy:0.47,pirate:0.39,therapist:-0.37},
  old_soldier:{soldier:0.45,grandparent:0.33,therapist:-0.14},
  analogy:{young_musician:1.0,old_spy:1.0,young_spy:-1.0},
  vampire_ish:{ghost:0.6,pirate:0.3,ancient:0.3}};
document.querySelectorAll('[data-preset]').forEach(b=>
  b.onclick=()=>setW(PRESETS[b.dataset.preset]));
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.classList.add('drag');
  cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('pointermove',e=>{if(!drag)return;
  yaw+=(e.clientX-drag[0])*0.0095;
  pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*0.0095));
  drag=[e.clientX,e.clientY];draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();
  zoom=Math.max(.4,Math.min(4,zoom*Math.exp(-e.deltaY*0.0013)));draw();},{passive:false});
addEventListener('resize',()=>{fit();draw();});
const lb=document.getElementById('layers');
D.layers.forEach(v=>{const b=document.createElement('button');b.textContent=v;b.dataset.l=v;
  b.onclick=()=>{LAY=v;markL();fit();draw();};lb.appendChild(b);});
function markL(){lb.querySelectorAll('button').forEach(b=>
  b.setAttribute('aria-pressed',b.dataset.l===LAY));}
markL();fit();draw();
</script>
"""
open("attr_analysis/mixer.html","w").write(HTML.replace("__DATA__", DATA))
print(f"mixer.html {len(HTML)+len(DATA)} bytes")
