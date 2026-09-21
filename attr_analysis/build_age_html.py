import json
VIZ = open("attr_analysis/age_viz.json").read()
A1  = open("attr_analysis/age_results_a1.json").read()
A2  = open("attr_analysis/age_results.json").read()

HTML = r"""<title>Steering along the age ladder</title>
<style>
:root{--surface:#fbfaf7;--raised:#f4f2ec;--ink:#14120e;--muted:#6d6a61;--line:#e2dfd5;
 --blue:#2a6fc9;--young:#e8b road;--y0:#f0c9a8;--y1:#8c3b12;--chord:#2a7a5c;--bad:#a3231a;
 --mono:ui-monospace,"SF Mono",Menlo,monospace;}
:root{--y0:#f0c9a8;--y1:#8c3b12;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;--line:#2d2b26;
 --blue:#63a2ee;--y0:#8a5a34;--y1:#f0a468;--chord:#4ec98a;--bad:#f0736a;}}
:root[data-theme="dark"]{--surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;
 --line:#2d2b26;--blue:#63a2ee;--y0:#8a5a34;--y1:#f0a468;--chord:#4ec98a;--bad:#f0736a;}
*{box-sizing:border-box}
body{background:var(--surface);color:var(--ink);margin:0;font:15px/1.6 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:23px;margin:0 0 6px;font-weight:650;letter-spacing:-.015em}
h2{font-size:15px;margin:30px 0 6px;font-weight:650}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 18px;max-width:70ch}
.bar{display:flex;gap:7px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.lab{color:var(--muted);font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase}
button{font-family:var(--mono);font-size:11.5px;padding:5px 10px;border:1px solid var(--line);
 background:var(--raised);color:var(--muted);border-radius:6px;cursor:pointer}
button:hover{color:var(--ink)}
button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.stage{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;touch-action:none}
canvas{display:block;width:100%;cursor:grab}canvas.drag{cursor:grabbing}
.card{border:1px solid var(--line);border-radius:12px;padding:15px;margin-top:12px}
table{border-collapse:collapse;font-family:var(--mono);font-size:11.5px;width:100%;
 font-variant-numeric:tabular-nums}
th,td{padding:4px 7px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:400;font-size:10px}
td:first-child,th:first-child{text-align:left}
td.good{color:var(--chord);font-weight:650}
td.bad{color:var(--bad)}
.key{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px;color:var(--muted);font-size:12.5px}
.k{display:flex;align-items:center;gap:7px}
.sw{width:22px;height:3px;border-radius:2px}
.note{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:15px;
 margin-top:26px;max-width:76ch}
.note b{color:var(--ink)}
.big{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:8px}
.big b{color:var(--ink);font-size:15px}
</style>
<div class="wrap">
<h1>Steering along the age ladder</h1>
<p class="sub">Nine life-stage personas, from infant to elder. The question: if you want the
model to sound 40, is it better to follow the ladder through its rungs, or to walk the
straight line from youngest to oldest? Drag to rotate.</p>

<div class="bar">
  <span class="lab">layer</span><span id="layers"></span>
  <span class="lab" style="margin-left:12px">show</span><span id="shows"></span>
  <span style="flex:1"></span><button id="reset">reset</button>
</div>
<div class="stage"><canvas id="c"></canvas></div>
<div class="key">
  <div class="k"><span class="sw" style="background:linear-gradient(90deg,var(--y0),var(--y1))"></span>the ladder, infant → elder</div>
  <div class="k"><span class="sw" style="background:var(--chord)"></span>straight chord</div>
  <div class="k"><span class="sw" style="background:var(--blue);height:9px;width:9px;border-radius:50%"></span>assistant</div>
</div>
<div class="big" id="geom"></div>

<h2>What the prompts can do, and what steering can do</h2>
<div class="card">
  <table id="ceil"><thead><tr><th>rung</th><th>nominal</th><th>prompt says</th><th>err</th>
  <th>steering says</th><th>err</th><th>nonsense</th></tr></thead><tbody></tbody></table>
  <div class="big" id="ceilsum"></div>
</div>

<h2>Three ways to reach a target age</h2>
<div class="card">
  <table id="paths"><thead><tr><th>β</th><th>target</th>
  <th>piecewise</th><th>spline</th><th>linear</th></tr></thead><tbody></tbody></table>
  <div class="big" id="pathsum"></div>
</div>

<p class="note"><b>Read this as a negative result.</b> The prompts encode age almost perfectly
(Pearson +0.996, one year of median error), so the ladder is real and the judge can read it.
Steering along it does not reproduce that. At the displacement where personas appear at all,
half the answers are degenerate; at lower displacement nothing happens. Neither the curved
path nor the straight one recovers the intended age, and the ladder's shape explains why: it
turns by more than 115° at three of its rungs and travels 2.9× the straight-line distance, so
there is very little consistent "age direction" to follow.</p>
</div>
<script>
const VIZ=__VIZ__, A1=__A1__, A2=__A2__;
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
let L=VIZ.layers[1], show=new Set(['ladder','chord','rungs']),
    yaw=-0.9,pitch=0.3,zoom=1,W=0,H=0,sc=1,ctr=[0,0,0],drag=null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
const mix=(a,b,t)=>{const A=hex(a),B=hex(b);return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(',')})`;};
const LD=()=>VIZ.L[L];
function proj(q){const p=[q[0]-ctr[0],q[1]-ctr[1],q[2]-ctr[2]];
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const x=p[0]*cy-p[2]*sy,z=p[0]*sy+p[2]*cy,y=p[1]*cp-z*sp,d=p[1]*sp+z*cp;
 return [W/2+x*sc*zoom,H/2-y*sc*zoom,d];}
function fit(){const d=LD();const pts=d.rungs.concat([[0,0,0]]);
 ctr=[0,1,2].map(i=>pts.reduce((s,p)=>s+p[i],0)/pts.length);
 W=cv.width=cv.clientWidth*devicePixelRatio;
 H=cv.height=Math.round(cv.clientWidth*0.6)*devicePixelRatio;
 cv.style.height=Math.round(cv.clientWidth*0.6)+'px';
 let m=1;for(const p of pts)m=Math.max(m,Math.hypot(p[0]-ctr[0],p[1]-ctr[1],p[2]-ctr[2]));
 sc=Math.min(W,H)/(m*2.6);}
function draw(){const d=LD(),ink=css('--ink'),sur=css('--surface'),mut=css('--muted');
 ctx.clearRect(0,0,W,H);const it=[];
 if(show.has('ladder')){const S=d.spline_dense;
   for(let i=0;i<S.length-1;i++){const A=proj(S[i]),B=proj(S[i+1]),t=i/(S.length-1);
     it.push({z:(A[2]+B[2])/2,f:()=>{ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);
       ctx.strokeStyle=mix(css('--y0'),css('--y1'),t);ctx.lineWidth=3.4*devicePixelRatio;
       ctx.lineCap='round';ctx.stroke();}});}}
 if(show.has('chord')){const A=proj(d.rungs[0]),B=proj(d.rungs[d.rungs.length-1]);
   it.push({z:(A[2]+B[2])/2,f:()=>{ctx.beginPath();ctx.setLineDash([7*devicePixelRatio,5*devicePixelRatio]);
     ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.strokeStyle=css('--chord');
     ctx.lineWidth=2.6*devicePixelRatio;ctx.stroke();ctx.setLineDash([]);}});}
 if(show.has('rungs'))d.rungs.forEach((r,i)=>{const P=proj(r),t=i/(d.rungs.length-1);
   it.push({z:P[2]+1e5,f:()=>{ctx.beginPath();ctx.arc(P[0],P[1],5.4*devicePixelRatio,0,6.2832);
     ctx.fillStyle=mix(css('--y0'),css('--y1'),t);ctx.fill();ctx.strokeStyle=sur;
     ctx.lineWidth=1.5*devicePixelRatio;ctx.stroke();
     ctx.fillStyle=ink;ctx.textAlign='center';ctx.lineJoin='round';
     ctx.font=`600 ${10.5*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
     ctx.strokeStyle=sur;ctx.lineWidth=3*devicePixelRatio;
     const dy=(i%2?15:-11)*devicePixelRatio;
     ctx.strokeText(VIZ.ladder[i],P[0],P[1]+dy);ctx.fillText(VIZ.ladder[i],P[0],P[1]+dy);}});});
 const O=proj([0,0,0]);
 it.push({z:O[2]+2e5,f:()=>{ctx.beginPath();ctx.arc(O[0],O[1],5*devicePixelRatio,0,6.2832);
   ctx.fillStyle=css('--blue');ctx.fill();ctx.strokeStyle=sur;ctx.lineWidth=1.5*devicePixelRatio;ctx.stroke();
   ctx.fillStyle=mut;ctx.textAlign='center';
   ctx.font=`${10*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
   ctx.fillText('assistant',O[0],O[1]+18*devicePixelRatio);}});
 it.sort((a,b)=>a.z-b.z).forEach(x=>x.f());
 document.getElementById('geom').innerHTML =
  `ladder length <b>${d.path_len}</b> to cover a straight-line distance of <b>${d.chord}</b> `+
  `&mdash; <b>${(d.path_len/d.chord).toFixed(1)}&times;</b> longer. turns: `+
  d.turns.map(t=>`${t}&deg;`).join(' · ');}
function tables(){
 const a1=A1.by_alpha["1.0"], tb=document.querySelector('#ceil tbody');tb.innerHTML='';
 a1.rungs.forEach(r=>{
   const p=r.prompt,c=r.control;
   const pe=p&&p.med!=null?Math.abs(p.med-r.nominal):null;
   const ce=c&&c.med!=null?Math.abs(c.med-r.nominal):null;
   tb.innerHTML+=`<tr><td>${r.rung}</td><td>${r.nominal}</td>`+
     `<td>${p&&p.med!=null?p.med.toFixed(0):'&mdash;'}</td>`+
     `<td class="${pe!=null&&pe<=3?'good':''}">${pe!=null?pe.toFixed(0):'&mdash;'}</td>`+
     `<td>${c&&c.med!=null?c.med.toFixed(0):'&mdash;'}</td>`+
     `<td class="${ce!=null&&ce>10?'bad':(ce!=null&&ce<=5?'good':'')}">${ce!=null?ce.toFixed(0):'&mdash;'}</td>`+
     `<td>${c?c.ns.toFixed(2):'&mdash;'}</td></tr>`;});
 document.getElementById('ceilsum').innerHTML =
  `prompt: median error <b>${a1.median_err_prompt.toFixed(1)}y</b>, pearson `+
  `<b>${a1.pearson_prompt.toFixed(3)}</b> &nbsp;·&nbsp; steering: median error `+
  `<b>${a1.median_err_control.toFixed(1)}y</b>, pearson <b>${a1.pearson_control.toFixed(3)}</b>`;
 const pb=document.querySelector('#paths tbody');pb.innerHTML='';
 a1.betas.forEach(r=>{
   let row=`<tr><td>${r.beta}</td><td>${r.target.toFixed(0)}</td>`;
   for(const arm of ["piecewise","spline","linear"]){
     const s=r[arm];
     row += s&&s.med!=null
       ? `<td>${s.med.toFixed(0)} <span style="color:var(--bad)">e${Math.abs(s.med-r.target).toFixed(0)}</span> <span style="color:var(--muted)">ns${s.ns.toFixed(2)}</span></td>`
       : `<td style="color:var(--muted)">no age${s?` ns${s.ns.toFixed(2)}`:''}</td>`;}
   pb.innerHTML+=row+'</tr>';});
 const S=a1.summary;
 document.getElementById('pathsum').innerHTML =
  ["piecewise","spline","linear"].map(a=>{
    const s=S[a];
    return `${a}: median err <b>${s.median_err!=null?s.median_err.toFixed(0)+'y':'n/a'}</b>, `+
           `nonsense <b>${s.nonsense.toFixed(2)}</b>`;}).join(' &nbsp;·&nbsp; ')
  + `<br>at lower displacement (&alpha; 0.45 and 0.65) every arm produced <b>no persona at all</b>.`;}
function mk(box,items,get,set){const el=document.getElementById(box);
 items.forEach(v=>{const b=document.createElement('button');b.textContent=v;b.dataset.v=v;
  b.onclick=()=>{set(v);mark();fit();draw();};el.appendChild(b);});
 function mark(){el.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',get(b.dataset.v)));}
 return mark;}
const mL=mk('layers',VIZ.layers,v=>v===L,v=>L=v);
const mS=mk('shows',['ladder','chord','rungs'],v=>show.has(v),
            v=>{show.has(v)?show.delete(v):show.add(v);if(!show.size)show.add(v);});
document.getElementById('reset').onclick=()=>{yaw=-0.9;pitch=0.3;zoom=1;draw();};
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.classList.add('drag');cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.0095;
 pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*0.0095));drag=[e.clientX,e.clientY];draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*Math.exp(-e.deltaY*0.0013)));draw();},{passive:false});
addEventListener('resize',()=>{fit();draw();});
mL();mS();fit();draw();tables();
</script>
"""
open("attr_analysis/age.html","w").write(
    HTML.replace("__VIZ__",VIZ).replace("__A1__",A1).replace("__A2__",A2))
print(f"age.html {len(HTML)+len(VIZ)+len(A1)+len(A2)} bytes")
