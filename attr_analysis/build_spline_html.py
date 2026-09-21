"""Standalone viewer for the attribute ladders."""
import json
DATA = open("attr_analysis/spline_data.json").read()

HTML = r"""<title>Persona attribute ladders</title>
<style>
:root{--surface:#fbfaf7;--raised:#f4f2ec;--ink:#14120e;--muted:#6d6a61;--line:#e2dfd5;
 --blue:#2a6fc9;--age0:#f0c9a8;--age1:#8c3b12;--era0:#a9c9e8;--era1:#173f6b;--occ:#7a7770;
 --mono:ui-monospace,"SF Mono",Menlo,"DejaVu Sans Mono",monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;--line:#2d2b26;
 --blue:#63a2ee;--age0:#8a5a34;--age1:#f0a468;--era0:#2f5578;--era1:#8fc2f0;--occ:#8b8880;}}
:root[data-theme="dark"]{--surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;
 --line:#2d2b26;--blue:#63a2ee;--age0:#8a5a34;--age1:#f0a468;--era0:#2f5578;--era1:#8fc2f0;--occ:#8b8880;}
*{box-sizing:border-box}
body{background:var(--surface);color:var(--ink);margin:0;font:15px/1.6 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:34px 20px 60px}
h1{font-size:23px;margin:0 0 6px;letter-spacing:-.015em;font-weight:650}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 20px;max-width:66ch}
.bar{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:9px}
button{font-family:var(--mono);font-size:11.5px;padding:5px 11px;border:1px solid var(--line);
 background:var(--raised);color:var(--muted);border-radius:6px;cursor:pointer;
 font-variant-numeric:tabular-nums}
button:hover{color:var(--ink)}
button:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.lab{color:var(--muted);font-family:var(--mono);font-size:10px;letter-spacing:.07em;
 text-transform:uppercase}
.spacer{flex:1}
.stage{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;
 background:var(--surface);touch-action:none}
canvas{display:block;width:100%;cursor:grab}canvas.drag{cursor:grabbing}
#tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--raised);
 border:1px solid var(--line);border-radius:7px;padding:7px 10px;font-family:var(--mono);
 font-size:11px;white-space:nowrap;box-shadow:0 6px 18px rgba(0,0,0,.16)}
.key{display:flex;gap:26px;flex-wrap:wrap;margin-top:14px;align-items:flex-end}
.ramp{width:170px}.ramp .g{height:9px;border-radius:5px}
.ramp .t{display:flex;justify-content:space-between;color:var(--muted);font-family:var(--mono);
 font-size:10px;margin-top:4px}
.mk{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12.5px}
.dot{width:10px;height:10px;border-radius:50%;flex:none}
.note{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:15px;
 margin-top:26px;max-width:74ch}
.note b{color:var(--ink);font-weight:650}
#keep{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:8px}
</style>
<div class="wrap">
<h1>Persona attribute ladders</h1>
<p class="sub">Ten life stages and five historical eras, each a persona the model was prompted
into. Drag to rotate, scroll to zoom, hover a rung for its name. The curve is drawn through
the rungs in their true order &mdash; if the ordering were arbitrary it would double back on
itself.</p>

<div class="bar">
  <span class="lab">axes</span><span id="views"></span>
  <span class="lab" style="margin-left:10px">layer</span><span id="layers"></span>
  <span class="spacer"></span>
  <button id="spin">spin</button><button id="reset">reset</button>
</div>
<div class="bar">
  <span class="lab">show</span><span id="groups"></span>
</div>

<div class="stage"><canvas id="c"></canvas><div id="tip"></div></div>
<div id="keep"></div>

<div class="key">
  <div class="ramp"><div class="lab">age ladder</div>
    <div class="g" id="rampAge"></div>
    <div class="t"><span>infant</span><span>ancient</span></div></div>
  <div class="ramp"><div class="lab">era ladder</div>
    <div class="g" id="rampEra"></div>
    <div class="t"><span>medieval</span><span>future</span></div></div>
  <div><div class="lab">marks</div>
    <div class="mk"><span class="dot" style="background:var(--blue)"></span>assistant</div>
    <div class="mk"><span class="dot" style="background:var(--occ)"></span>occupation personas</div></div>
</div>

<p class="note"><b>Only three directions fit on a screen.</b> Each choice of axes is the top
three directions of one group, so that group is shown faithfully and the others are squashed.
The line under the plot reports how much of each ladder the current axes actually keep &mdash;
switch axes before judging whether a ladder looks straight.</p>
</div>
<script>
const DATA = __DATA__;
const cv=document.getElementById('c'), ctx=cv.getContext('2d'), tip=document.getElementById('tip');
const VIEWS=Object.keys(DATA.views);
let V=VIEWS[0], L=DATA.layers[1], on=new Set(['age','assistant','spread']),
    yaw=-0.9, pitch=0.3, zoom=1, spinning=false, ctr=[0,0,0], W=0,H=0,scale=1, nodes=[];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
const mix=(a,b,t)=>{const A=hex(a),B=hex(b);
  return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(',')})`;};
const lay=()=>DATA.views[V][L];
function proj(q){
  const p=[q[0]-ctr[0],q[1]-ctr[1],q[2]-ctr[2]];
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const x=p[0]*cy-p[2]*sy, z=p[0]*sy+p[2]*cy;
  const y=p[1]*cp-z*sp, d=p[1]*sp+z*cp;
  return [W/2+x*scale*zoom, H/2-y*scale*zoom, d];
}
function eig3(S){let a=S.map(r=>r.slice()),v=[[1,0,0],[0,1,0],[0,0,1]];
 for(let s=0;s<24;s++){let p=0,q=1,m=Math.abs(a[0][1]);
  for(const[i,j]of[[0,2],[1,2]])if(Math.abs(a[i][j])>m){m=Math.abs(a[i][j]);p=i;q=j;}
  if(m<1e-12)break;const th=0.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(th),sn=Math.sin(th);
  for(let k=0;k<3;k++){const x=a[k][p],y=a[k][q];a[k][p]=c*x-sn*y;a[k][q]=sn*x+c*y;}
  for(let k=0;k<3;k++){const x=a[p][k],y=a[q][k];a[p][k]=c*x-sn*y;a[q][k]=sn*x+c*y;
   const u=v[k][p],w=v[k][q];v[k][p]=c*u-sn*w;v[k][q]=sn*u+c*w;}}
 return{vec:v,val:[a[0][0],a[1][1],a[2][2]]};}
function rings(mu,S,k){const{vec,val}=eig3(S),out=[];
 for(const[i,j]of[[0,1],[0,2],[1,2]]){const r=[];
  for(let t=0;t<=40;t++){const u=t/40*2*Math.PI,
   ci=k*Math.sqrt(Math.max(val[i],1e-9))*Math.cos(u),
   cj=k*Math.sqrt(Math.max(val[j],1e-9))*Math.sin(u);
   r.push([0,1,2].map(d=>mu[d]+ci*vec[d][i]+cj*vec[d][j]));}
  out.push(r);}return out;}
function fit(){
  const d=lay(); const pts=[[0,0,0]];
  for(const g of ['age','era','occupations']) if(on.has(g))
    d.groups[g].nodes.forEach(n=>pts.push(n.mu));
  ctr=[0,1,2].map(i=>pts.reduce((s,p)=>s+p[i],0)/pts.length);
  W=cv.width=cv.clientWidth*devicePixelRatio;
  H=cv.height=Math.round(cv.clientWidth*0.66)*devicePixelRatio;
  cv.style.height=Math.round(cv.clientWidth*0.66)+'px';
  let m=1; for(const p of pts) m=Math.max(m,Math.hypot(p[0]-ctr[0],p[1]-ctr[1],p[2]-ctr[2]));
  scale=Math.min(W,H)/(m*2.5);
}
function draw(){
  const d=lay(), ink=css('--ink'), sur=css('--surface'), muted=css('--muted');
  ctx.clearRect(0,0,W,H); const prim=[]; nodes=[];
  const push=(z,f)=>prim.push({z,f});
  const ring=(rs,col,al)=>{for(const r of rs){const P=r.map(proj);
    push(P.reduce((s,p)=>s+p[2],0)/P.length,()=>{ctx.beginPath();
      P.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
      ctx.strokeStyle=col;ctx.globalAlpha=al;ctx.lineWidth=1*devicePixelRatio;
      ctx.stroke();ctx.globalAlpha=1;});}};
  if(on.has('assistant')&&on.has('spread')) ring(rings([0,0,0],d.assistant.S,1.5),css('--blue'),0.4);
  const RAMP={age:[css('--age0'),css('--age1')],era:[css('--era0'),css('--era1')]};
  for(const g of ['age','era']){
    if(!on.has(g))continue; const G=d.groups[g], N=G.nodes.length;
    const S=G.spline, seg=(S.length-1);
    for(let i=0;i<seg;i++){const A=proj(S[i]),B=proj(S[i+1]);
      const t=i/seg, col=mix(RAMP[g][0],RAMP[g][1],t);
      push((A[2]+B[2])/2,()=>{ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);
        ctx.strokeStyle=col;ctx.lineWidth=3.2*devicePixelRatio;ctx.lineCap='round';ctx.stroke();});}
    G.nodes.forEach((n,i)=>{const t=i/(N-1), col=mix(RAMP[g][0],RAMP[g][1],t), P=proj(n.mu);
      if(on.has('spread')) ring(rings(n.mu,n.S,1.5),col,0.16);
      nodes.push({x:P[0],y:P[1],name:n.name,group:g});
      push(P[2]+1e5,()=>{ctx.beginPath();ctx.arc(P[0],P[1],5.6*devicePixelRatio,0,6.2832);
        ctx.fillStyle=col;ctx.fill();ctx.strokeStyle=sur;ctx.lineWidth=1.5*devicePixelRatio;ctx.stroke();
        ctx.fillStyle=ink;ctx.textAlign='center';ctx.lineJoin='round';
        ctx.font=`600 ${10.5*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
        ctx.strokeStyle=sur;ctx.lineWidth=3*devicePixelRatio;
        const dy=(i%2?15:-11)*devicePixelRatio;
        ctx.strokeText(n.name,P[0],P[1]+dy);
        ctx.fillText(n.name,P[0],P[1]+dy);});});
  }
  if(on.has('occupations')){const G=d.groups.occupations;
    G.nodes.forEach(n=>{const P=proj(n.mu); if(on.has('spread')) ring(rings(n.mu,n.S,1.5),css('--occ'),0.14);
      nodes.push({x:P[0],y:P[1],name:n.name,group:'occupations'});
      push(P[2]+1e5,()=>{ctx.beginPath();ctx.arc(P[0],P[1],4.6*devicePixelRatio,0,6.2832);
        ctx.fillStyle=css('--occ');ctx.fill();ctx.strokeStyle=sur;
        ctx.lineWidth=1.4*devicePixelRatio;ctx.stroke();
        ctx.fillStyle=muted;ctx.textAlign='center';ctx.lineJoin='round';
        ctx.font=`${10*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
        ctx.strokeStyle=sur;ctx.lineWidth=3*devicePixelRatio;
        ctx.strokeText(n.name,P[0],P[1]+16*devicePixelRatio);
        ctx.fillText(n.name,P[0],P[1]+16*devicePixelRatio);});});}
  if(on.has('assistant')){const O=proj([0,0,0]);
    push(O[2]+2e5,()=>{ctx.beginPath();ctx.arc(O[0],O[1],6*devicePixelRatio,0,6.2832);
      ctx.fillStyle=css('--blue');ctx.fill();ctx.strokeStyle=sur;
      ctx.lineWidth=1.6*devicePixelRatio;ctx.stroke();
      ctx.fillStyle=ink;ctx.textAlign='center';ctx.lineJoin='round';
      ctx.font=`600 ${11*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
      ctx.strokeStyle=sur;ctx.lineWidth=3*devicePixelRatio;
      ctx.strokeText('assistant',O[0],O[1]+19*devicePixelRatio);
      ctx.fillText('assistant',O[0],O[1]+19*devicePixelRatio);});}
  prim.sort((a,b)=>a.z-b.z).forEach(p=>p.f());
  const k=d.keep;
  document.getElementById('keep').textContent =
    `axes keep: age ${k.age} · era ${k.era} · occupations ${k.occupations} of each group's spread`;
}
let drag=null;
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];spinning=false;
  document.getElementById('spin').setAttribute('aria-pressed','false');
  cv.classList.add('drag');cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('pointermove',e=>{
  if(drag){yaw+=(e.clientX-drag[0])*0.0095;
    pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*0.0095));
    drag=[e.clientX,e.clientY];draw();return;}
  const R=cv.getBoundingClientRect(),mx=(e.clientX-R.left)*devicePixelRatio,
        my=(e.clientY-R.top)*devicePixelRatio;
  let best=null,bd=15*devicePixelRatio;
  for(const n of nodes){const q=Math.hypot(n.x-mx,n.y-my);if(q<bd){bd=q;best=n;}}
  if(best){tip.textContent=`${best.name} (${best.group})`;tip.style.opacity=1;
    tip.style.left=Math.min(R.width-140,best.x/devicePixelRatio+14)+'px';
    tip.style.top=(best.y/devicePixelRatio-8)+'px';}
  else tip.style.opacity=0;});
cv.addEventListener('pointerleave',()=>tip.style.opacity=0);
cv.addEventListener('wheel',e=>{e.preventDefault();
  zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*0.0013)));draw();},{passive:false});
function mkBtns(box,items,get,set){
  const el=document.getElementById(box);
  items.forEach(v=>{const b=document.createElement('button');
    b.textContent=v;b.dataset.v=v;b.onclick=()=>{set(v);mark();fit();draw();};el.appendChild(b);});
  function mark(){el.querySelectorAll('button').forEach(b=>
    b.setAttribute('aria-pressed',get(b.dataset.v)));}
  return mark;}
const markV=mkBtns('views',VIEWS,v=>v===V,v=>V=v);
const markL=mkBtns('layers',DATA.layers,v=>v===L,v=>L=v);
const markG=mkBtns('groups',['assistant','age','era','occupations','spread'],
  v=>on.has(v),v=>{on.has(v)?on.delete(v):on.add(v);if(!on.size)on.add(v);});
document.getElementById('reset').onclick=()=>{yaw=-0.9;pitch=0.3;zoom=1;draw();};
const sb=document.getElementById('spin');
sb.onclick=()=>{spinning=!spinning;sb.setAttribute('aria-pressed',spinning);
  if(spinning)requestAnimationFrame(tick);};
function tick(){if(!spinning)return;yaw+=0.004;draw();requestAnimationFrame(tick);}
document.getElementById('rampAge').style.background=
  `linear-gradient(90deg,${css('--age0')},${css('--age1')})`;
document.getElementById('rampEra').style.background=
  `linear-gradient(90deg,${css('--era0')},${css('--era1')})`;
addEventListener('resize',()=>{fit();draw();});
markV();markL();markG();fit();draw();
</script>
"""
open("attr_analysis/spline.html","w").write(HTML.replace("__DATA__", DATA))
print(f"spline.html {len(HTML)+len(DATA)} bytes")
