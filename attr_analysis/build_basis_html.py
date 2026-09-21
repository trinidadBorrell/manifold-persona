import json
DATA = open("attr_analysis/basis_data.json").read()
HTML = r"""<title>Persona coordinates</title>
<style>
:root{--surface:#fbfaf7;--raised:#f4f2ec;--ink:#14120e;--muted:#6d6a61;--line:#e2dfd5;
 --blue:#2a6fc9;--pos:#8c3b12;--neg:#2a5c7a;--grid:#ebe8e0;
 --mono:ui-monospace,"SF Mono",Menlo,monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;--line:#2d2b26;
 --blue:#63a2ee;--pos:#e08a4e;--neg:#5fa8d0;--grid:#232120;}}
:root[data-theme="dark"]{--surface:#131210;--raised:#1c1a17;--ink:#f2f0ea;--muted:#9a968c;
 --line:#2d2b26;--blue:#63a2ee;--pos:#e08a4e;--neg:#5fa8d0;--grid:#232120;}
*{box-sizing:border-box}
body{background:var(--surface);color:var(--ink);margin:0;font:15px/1.55 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:30px 20px 60px}
h1{font-size:22px;margin:0 0 5px;font-weight:650;letter-spacing:-.015em}
h2{font-size:14px;margin:26px 0 4px;font-weight:650}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 18px;max-width:72ch}
.bar{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:12px}
.lab{color:var(--muted);font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase}
button,select{font-family:var(--mono);font-size:11.5px;padding:5px 10px;border:1px solid var(--line);
 background:var(--raised);color:var(--muted);border-radius:6px;cursor:pointer}
button:hover{color:var(--ink)}
button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
select{color:var(--ink);max-width:230px}
.card{border:1px solid var(--line);border-radius:12px;padding:15px;margin-top:12px}
.formula{font-family:var(--mono);font-size:13px;line-height:1.9;word-break:break-word}
.formula .p{color:var(--pos);font-weight:650}
.formula .n{color:var(--neg);font-weight:650}
.stat{display:flex;gap:26px;flex-wrap:wrap;margin-top:10px}
.stat div{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.stat b{display:block;font-size:19px;color:var(--ink);font-weight:650;font-variant-numeric:tabular-nums}
.note{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;
 margin-top:26px;max-width:76ch}
.note b{color:var(--ink)}
canvas{display:block;width:100%}
.coordrow{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;margin-bottom:2px}
.coordrow span{width:118px;text-align:right;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.coordrow .track{flex:1;height:12px;position:relative;background:linear-gradient(90deg,transparent 49.7%,var(--line) 49.7%,var(--line) 50.3%,transparent 50.3%)}
.coordrow i{position:absolute;top:2px;height:8px;border-radius:2px;display:block}
.coordrow b{width:46px;text-align:right;font-variant-numeric:tabular-nums;font-weight:500}
</style>
<div class="wrap">
<h1>Persona coordinates</h1>
<p class="sub">Two ways to write a persona down, across all 300 personas in the paper's role
set. <b>Persona basis</b>: greedily pick the personas that explain the most, orthogonalise,
continue to 99% &mdash; which takes 160 of them at layer 28. <b>Concept basis</b>: named
contrast axes, orthogonalised in order, which reach only 78%. One is exact, the other is
readable, and the gap between them is the price of naming things.</p>

<div class="bar">
  <span class="lab">layer</span><span id="layers"></span>
  <span class="lab" style="margin-left:12px">basis</span><span id="bases"></span>
  <span class="lab" style="margin-left:12px">persona</span>
  <select id="pick"></select>
</div>

<div class="card">
  <div class="lab">how much of persona space each basis captures</div>
  <canvas id="curve"></canvas>
  <div class="stat" id="stats"></div>
</div>

<h2 id="ftitle">written in this basis</h2>
<div class="card"><div class="formula" id="formula"></div></div>

<h2>coordinates</h2>
<div class="card" id="coords"></div>

<p class="note"><b>The region is thin, but its tail is heavy.</b> 23 directions carry 90% of the
variation among 300 personas in 2048 dimensions. Going from 90% to 99% costs about another
hundred. The concept axes stall around 78% because human-nameable concepts are not
geometrically distinct &mdash; the closest pairs sit at cosine 0.88&ndash;0.91, so several
axes add almost nothing once orthogonalised.</p>
</div>
<script>
const D=__DATA__;
let LAY=D.layers[1], BAS='personas', SEL='vampire';
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const LD=()=>D.L[LAY];
function mkb(box,items,get,set){const el=document.getElementById(box);
  items.forEach(v=>{const b=document.createElement('button');b.textContent=v;b.dataset.v=v;
    b.onclick=()=>{set(v);render();};el.appendChild(b);});
  return()=>el.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',get(b.dataset.v)));}
const markL=mkb('layers',D.layers,v=>v===LAY,v=>LAY=v);
const markB=mkb('bases',['personas','concepts'],v=>v===BAS,v=>BAS=v);
const sel=document.getElementById('pick');
D.personas.forEach(p=>{const o=document.createElement('option');o.value=o.textContent=p;sel.appendChild(o);});
sel.value=SEL; sel.onchange=()=>{SEL=sel.value;render();};

function curve(){
  const c=document.getElementById('curve'), x=c.getContext('2d');
  const w=c.width=c.clientWidth*devicePixelRatio, h=c.height=220*devicePixelRatio;
  x.clearRect(0,0,w,h);
  const P=48, L=LD(), series=[
    {d:L.persona_curve,col:css('--pos'),name:'persona basis'},
    {d:L.concept_curve,col:css('--blue'),name:'concept basis'},
    {d:L.pca_curve,col:css('--muted'),name:'PCA (optimal)',dash:[4,4]}];
  const nmax=Math.max(...series.map(s=>s.d.length));
  const X=i=>P+ (w-P*1.4)*i/(nmax-1), Y=v=>h-26*devicePixelRatio-(h-52*devicePixelRatio)*v;
  x.strokeStyle=css('--grid');x.lineWidth=1*devicePixelRatio;
  x.fillStyle=css('--muted');x.font=`${9.5*devicePixelRatio}px ${css('--mono')}`;
  [0.5,0.75,0.9,0.99].forEach(v=>{x.beginPath();x.moveTo(P,Y(v));x.lineTo(w-P*0.4,Y(v));x.stroke();
    x.textAlign='right';x.fillText(v.toFixed(2),P-6*devicePixelRatio,Y(v)+3*devicePixelRatio);});
  series.forEach(s=>{x.beginPath();x.setLineDash((s.dash||[]).map(v=>v*devicePixelRatio));
    s.d.forEach((v,i)=>{const px=X(i),py=Y(v);i?x.lineTo(px,py):x.moveTo(px,py);});
    x.strokeStyle=s.col;x.lineWidth=2.2*devicePixelRatio;x.stroke();x.setLineDash([]);
    const li=s.d.length-1;x.beginPath();x.arc(X(li),Y(s.d[li]),3.4*devicePixelRatio,0,6.3);
    x.fillStyle=s.col;x.fill();
    x.textAlign='left';x.fillStyle=s.col;
    x.font=`600 ${10*devicePixelRatio}px ui-sans-serif,system-ui,sans-serif`;
    x.fillText(s.name,X(li)+7*devicePixelRatio,Y(s.d[li])+3*devicePixelRatio);});
  x.fillStyle=css('--muted');x.textAlign='center';
  x.font=`${9.5*devicePixelRatio}px ${css('--mono')}`;
  [1,20,40,60,80,100,120,140,160,180,200].filter(v=>v<=nmax).forEach(v=>
    x.fillText(v,X(v-1),h-9*devicePixelRatio));
}
function render(){
  markL();markB();curve();
  const L=LD(), isP=BAS==='personas';
  const basis=isP?L.persona_basis:L.concept_basis;
  const coord=(isP?L.coordP:L.coordC)[SEL]||[];
  const fit=(isP?L.fitP:L.fitC)[SEL];
  document.getElementById('stats').innerHTML=
    `<div>axes<b>${basis.length}</b></div>`+
    `<div>space captured<b>${(isP?L.persona_curve.at(-1):L.concept_explained).toFixed(3)}</b></div>`+
    `<div>${SEL} rebuilt${isP?' (without itself)':''}<b>${((isP?L.looFit[SEL]:fit)*100).toFixed(1)}%</b></div>`+
    `<div>&#8214;${SEL}&#8214;<b>${L.norms[SEL]}</b></div>`;
  const pairs = isP ? (L.looP[SEL]||[]).slice().sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]))
                    : basis.map((b,i)=>[b,coord[i]||0]).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  document.getElementById('ftitle').textContent =
    isP ? `${SEL} rebuilt from the other basis personas` : `${SEL} in concept coordinates`;
  document.getElementById('formula').innerHTML= SEL+' &asymp; '+
    pairs.slice(0,7).filter(p=>Math.abs(p[1])>0.02).map(([b,v])=>
      `<span class="${v>=0?'p':'n'}">${v>=0?'+':'&minus;'}${Math.abs(v).toFixed(2)}</span>&#183;${b}`
    ).join(' ')+' &hellip;';
  const mx=Math.max(...pairs.map(p=>Math.abs(p[1])),1e-6);
  document.getElementById('coords').innerHTML=pairs.slice(0,18).map(([b,v])=>{
    const wd=Math.abs(v)/mx*46, left=v>=0?50:50-wd;
    return `<div class="coordrow"><span title="${b}">${b}</span>`+
      `<div class="track"><i style="left:${left}%;width:${wd}%;background:${v>=0?'var(--pos)':'var(--neg)'}"></i></div>`+
      `<b>${v>=0?'+':'−'}${Math.abs(v).toFixed(2)}</b></div>`;}).join('');
}
addEventListener('resize',render);
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',render);
render();
</script>
"""
open("attr_analysis/basis.html","w").write(HTML.replace("__DATA__",DATA))
print(f"basis.html {len(HTML)+len(DATA)} bytes")
