import json
d = json.load(open("attr_analysis/steer_viz.json"))
# JSON numbers stringify differently in JS than the dict keys do; key everything by the key
d["add_alphas"] = list(d["P"][d["personas"][0]][d["layers"][0]]["add_total"])
d["pin_scales"] = list(d["P"][d["personas"][0]][d["layers"][0]]["pin"])
for f in ("best_add", "best_pin"):
    d[f] = {k: (repr(v) if v != int(v) else f"{v:.1f}") for k, v in d[f].items()}
V = json.dumps(d)
HTML = r"""<title>Additive steering vs pinned steering</title>
<style>
:root{--surface:#fbfaf7;--raised:#f4f2ec;--ink:#14120e;--muted:#6d6a61;--line:#e2dfd5;
 --asst:#2a6fc9;--target:#1f7a4d;--add:#a3231a;--pin:#7b4fbf;--mono:ui-monospace,"SF Mono",Menlo,monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;--line:#2d2b26;
 --asst:#63a2ee;--target:#4ec98a;--add:#f0736a;--pin:#b78cf0;}}
:root[data-theme="dark"]{--surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;
 --line:#2d2b26;--asst:#63a2ee;--target:#4ec98a;--add:#f0736a;--pin:#b78cf0;}
*{box-sizing:border-box}
body{background:var(--surface);color:var(--ink);margin:0;font:15px/1.6 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:23px;margin:0 0 6px;font-weight:650;letter-spacing:-.015em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 18px;max-width:72ch}
.bar{display:flex;gap:7px;align-items:center;margin-bottom:9px;flex-wrap:wrap}
.lab{color:var(--muted);font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase}
button{font-family:var(--mono);font-size:11.5px;padding:5px 10px;border:1px solid var(--line);
 background:var(--raised);color:var(--muted);border-radius:6px;cursor:pointer}
button:hover{color:var(--ink)}
button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.stage{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;touch-action:none}
canvas{display:block;width:100%;cursor:grab}canvas.drag{cursor:grabbing}
.key{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;color:var(--muted);font-size:12.5px}
.k{display:flex;align-items:center;gap:7px}
.sw{width:20px;height:3px;border-radius:2px}.dot{width:9px;height:9px;border-radius:50%}
.card{border:1px solid var(--line);border-radius:12px;padding:14px;margin-top:14px;
 font-family:var(--mono);font-size:12px;color:var(--muted)}
.card b{color:var(--ink);font-size:14px}
.note{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:15px;
 margin-top:26px;max-width:76ch}.note b{color:var(--ink)}
</style>
<div class="wrap">
<h1>Additive steering vs pinned steering</h1>
<p class="sub">Both methods aim at the same persona, along the same direction, and end up in
very different places. The assistant is at the origin, the persona's own centroid is the green
dot. Drag to rotate. Everything is drawn in the three directions that carry the persona's
displacement across layers 26 to 30, so all layers share one frame. The red path takes one
step per layer: each push is carried forward by the residual stream, so the steps add up and
bend away from the direction the last layer actually asks for.</p>

<div class="bar">
  <span class="lab">persona</span><span id="who"></span>
  <span class="lab" style="margin-left:10px">layer</span><span id="layers"></span>
  <span style="flex:1"></span><button id="reset">reset</button>
</div>
<div class="stage"><canvas id="c"></canvas></div>
<div class="key">
  <div class="k"><span class="dot" style="background:var(--asst)"></span>assistant</div>
  <div class="k"><span class="dot" style="background:var(--target)"></span>persona centroid</div>
  <div class="k"><span class="sw" style="background:var(--add)"></span>additive, one push per layer</div>
  <div class="k"><span class="sw" style="background:var(--pin)"></span>pin, where it places the state</div>
</div>
<div class="card" id="nums"></div>

<p class="note"><b>Why the two lines differ so much.</b> Additive adds its push at every layer
and the residual stream carries each one forward, so by layer 30 it has added about 3.7 times
the persona's own distance. Pin sets the position on the direction instead of adding to it, so
it lands exactly where you ask. Both work, and both need to end up far past the centroid: the
useful range is roughly 80 to 105 units out, and past about 130 the model stops making sense.
The best multiplier therefore differs per persona only because personas start at different
distances.</p>
</div>
<script>
const V=__V__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let P=V.personas[0],L=V.layers[V.layers.length-1],yaw=-.9,pitch=.3,zoom=1,W=0,H=0,sc=1,ctr=[0,0,0],drag=null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const D=()=>V.P[P][L];
function proj(q){const p=[q[0]-ctr[0],q[1]-ctr[1],q[2]-ctr[2]];
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const x=p[0]*cy-p[2]*sy,z=p[0]*sy+p[2]*cy,y=p[1]*cp-z*sp,d=p[1]*sp+z*cp;
 return [W/2+x*sc*zoom,H/2-y*sc*zoom,d];}
function pts(){const d=D(),o=[[0,0,0],d.target];
 for(const a of V.add_alphas)o.push(d.add_total[a]);
 for(const s of V.pin_scales)o.push(d.pin[s]);return o;}
function fit(){const o=pts();
 ctr=[0,1,2].map(i=>o.reduce((s,p)=>s+p[i],0)/o.length);
 W=cv.width=cv.clientWidth*devicePixelRatio;
 H=cv.height=Math.round(cv.clientWidth*0.62)*devicePixelRatio;
 cv.style.height=Math.round(cv.clientWidth*0.62)+'px';
 let m=1;for(const p of o)m=Math.max(m,Math.hypot(p[0]-ctr[0],p[1]-ctr[1],p[2]-ctr[2]));
 sc=Math.min(W,H)/(m*2.15);}
function line(items,A,B,col,w,dash){const a=proj(A),b=proj(B);
 items.push({z:(a[2]+b[2])/2,f:()=>{ctx.beginPath();ctx.setLineDash(dash||[]);
  ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=col;
  ctx.lineWidth=w*devicePixelRatio;ctx.lineCap='round';ctx.stroke();ctx.setLineDash([]);}});}
function mark(items,A,col,r,label,dy){const p=proj(A);
 items.push({z:p[2]+1e5,f:()=>{ctx.beginPath();ctx.arc(p[0],p[1],r*devicePixelRatio,0,6.2832);
  ctx.fillStyle=col;ctx.fill();ctx.strokeStyle=css('--surface');
  ctx.lineWidth=1.4*devicePixelRatio;ctx.stroke();
  if(label){ctx.fillStyle=css('--ink');ctx.textAlign='center';ctx.lineJoin='round';
   ctx.font=`600 ${10.5*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
   ctx.strokeStyle=css('--surface');ctx.lineWidth=3*devicePixelRatio;
   ctx.strokeText(label,p[0],p[1]+dy*devicePixelRatio);
   ctx.fillText(label,p[0],p[1]+dy*devicePixelRatio);}}});}
function draw(){const d=D(),items=[];ctx.clearRect(0,0,W,H);
 const bA=V.best_add[P],bS=V.best_pin[P];
 // every setting, faint
 for(const a of V.add_alphas) if(a!==bA) line(items,[0,0,0],d.add_total[a],css('--add'),1.2,[4,4]);
 for(const s of V.pin_scales) if(s!==bS) line(items,[0,0,0],d.pin[s],css('--pin'),1.2,[4,4]);
 // the additive path: one push per layer, carried forward by the residual stream
 let prev=[0,0,0];
 for(const l of V.layers){const q=V.P[P][l].add_total[bA];
  line(items,prev,q,css('--add'),3.2);
  if(l!==L)mark(items,q,css('--add'),3.4,null,0);
  prev=q; if(l===L)break;}
 line(items,[0,0,0],d.pin[bS],css('--pin'),3.2);
 line(items,[0,0,0],d.target,css('--target'),2.4);
 for(const a of V.add_alphas) mark(items,d.add_total[a],css('--add'),a===bA?5.5:3,
      a===bA?`add ${a}`:null,-11);
 for(const s of V.pin_scales) mark(items,d.pin[s],css('--pin'),s===bS?5.5:3,
      s===bS?`pin ${s}`:null,15);
 mark(items,d.target,css('--target'),6,'persona',-12);
 mark(items,[0,0,0],css('--asst'),6,'assistant',18);
 items.sort((a,b)=>a.z-b.z).forEach(i=>i.f());
 document.getElementById('nums').innerHTML =
  `at layer ${L}: persona centroid sits <b>${d.norm}</b> from the assistant &nbsp;·&nbsp; `+
  `additive at &alpha;=${bA} has accumulated <b>${d.add_total_norm[bA]}</b> &nbsp;·&nbsp; `+
  `pin at s=${bS} places it at <b>${d.pin_norm[bS]}</b>`;}
function mk(box,items,get,set){const el=document.getElementById(box);
 items.forEach(v=>{const b=document.createElement('button');b.textContent=v;b.dataset.v=v;
  b.onclick=()=>{set(v);m();fit();draw();};el.appendChild(b);});
 function m(){el.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',get(b.dataset.v)));}
 return m;}
const mW=mk('who',V.personas,v=>v===P,v=>P=v);
const mL=mk('layers',V.layers,v=>v===L,v=>L=v);
document.getElementById('reset').onclick=()=>{yaw=-.9;pitch=.3;zoom=1;draw();};
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.classList.add('drag');cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.0095;
 pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.0095));drag=[e.clientX,e.clientY];draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*Math.exp(-e.deltaY*.0013)));draw();},{passive:false});
addEventListener('resize',()=>{fit();draw();});
mW();mL();fit();draw();
</script>
"""
open("attr_analysis/steer_compare.html","w").write(HTML.replace("__V__",V))
print(f"steer_compare.html {len(HTML)+len(V)} bytes")
