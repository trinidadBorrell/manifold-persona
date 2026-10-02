import json
V = open("attr_analysis/steer_examples.json").read()
HTML = r"""<title>Steered Persona Transcripts</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
/* instrument readout: fixed control rail, transcripts stacked as one column of specimens */
:root{
  --paper:#f7f6f3; --panel:#efeee9; --edge:#dedbd2; --ink:#16181a; --dim:#6a6d6b;
  --accent:#1c6b5c; --warn:#9a5a16; --bad:#8f2f28; --good:#1c6b5c;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --read:"Newsreader",Georgia,serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#111312; --panel:#191c1b; --edge:#2b2f2d; --ink:#eceae4; --dim:#92968f;
  --accent:#5fcab0; --warn:#d99a4e; --bad:#e2766c; --good:#5fcab0; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#111312; --panel:#191c1b; --edge:#2b2f2d; --ink:#eceae4; --dim:#92968f;
  --accent:#5fcab0; --warn:#d99a4e; --bad:#e2766c; --good:#5fcab0; color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.6 var(--read);-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding-block:30px 70px;padding-left:20px;padding-right:20px}
h1{font-size:clamp(25px,4vw,34px);font-weight:500;margin:0 0 8px;letter-spacing:-.02em;text-wrap:balance}
.lede{color:var(--dim);margin:0 0 6px;max-width:66ch;font-size:15.5px}
.meta{font-family:var(--mono);font-size:11.5px;color:var(--dim);margin:0 0 24px}
.rail{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;
  border:1px solid var(--edge);background:var(--panel);border-radius:3px;padding:15px;margin-bottom:12px}
.grp{display:flex;flex-direction:column;gap:7px;min-width:0}
.cap{font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim)}
.chips{display:flex;flex-wrap:wrap;gap:5px}
button{font-family:var(--mono);font-size:11px;padding:4px 9px;border:1px solid var(--edge);
  background:transparent;color:var(--dim);border-radius:2px;cursor:pointer;transition:.12s}
button:hover{color:var(--ink);border-color:var(--dim)}
button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.q{font-family:var(--mono);font-size:12px;color:var(--dim);border:1px solid var(--edge);
  background:var(--panel);border-radius:3px;padding:11px 14px;margin-bottom:22px}
.q b{color:var(--ink);font-weight:500}
.recipe{font-family:var(--mono);font-size:11px;color:var(--dim);line-height:1.8;
  padding:11px 14px;border-left:2px solid var(--accent);margin-bottom:24px}
.recipe b{color:var(--ink);font-weight:500}
.spec{border-top:1px solid var(--edge);padding-block:18px;display:grid;
  grid-template-columns:168px minmax(0,1fr);gap:22px}
.spec:last-child{border-bottom:1px solid var(--edge)}
.side{min-width:0}
.arm{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--ink);margin-bottom:5px}
.sub{font-family:var(--mono);font-size:10.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:9px}
.tag{font-family:var(--mono);font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;
  padding:2px 6px;border-radius:2px;border:1px solid currentColor}
.hit{color:var(--good)}.miss{color:var(--dim)}.nons{color:var(--bad)}.plain{color:var(--warn)}
.body{font-size:16px;line-height:1.72;min-width:0}
.body.none{color:var(--dim);font-style:italic}
.foot{color:var(--dim);font-size:13px;border-top:1px solid var(--edge);margin-top:34px;
  padding-top:18px;max-width:72ch}
.foot b{color:var(--ink);font-weight:500}
@media(max-width:620px){.spec{grid-template-columns:1fr;gap:10px}}
</style>
<div class="wrap">
<h1>Steered Persona Transcripts</h1>
<p class="lede">The same question put to Qwen2.5-3B with no system prompt, once for each way of
pushing its activations toward a character. Every arm sets one coordinate at a single layer to the
same distance, so only the direction differs.</p>
<p class="meta">12 personas &middot; 10 questions &middot; 5 arms &middot; 4 strengths &middot; 2400 answers, labelled blind</p>

<div class="rail">
  <div class="grp"><span class="cap">persona</span><div class="chips" id="who"></div></div>
  <div class="grp"><span class="cap">question</span><div class="chips" id="qs"></div></div>
  <div class="grp"><span class="cap">strength</span><div class="chips" id="st"></div></div>
</div>
<div class="q" id="qt"></div>
<div class="recipe" id="rc"></div>
<div id="out"></div>

<p class="foot"><b>How to read the labels.</b> <span class="hit">named</span> means the blind judge
picked this persona from a list of twelve. <span class="miss">missed</span> means it named a
different one, or none. <span class="plain">assistant</span> means the answer stayed ordinary
helpful-AI prose. <span class="nons">degenerate</span> means repetition or word salad. An answer can
be in character and still be missed, and it can name the right character while sounding like an
assistant, which is why both labels are shown.</p>
</div>
<script>
const V=__V__;
let P=V.personas[0], Q="0", MODE="best";
const el=id=>document.getElementById(id);

function chips(box,items,get,set,fmt){
  const b=el(box);
  items.forEach(v=>{const x=document.createElement("button");
    x.textContent=fmt?fmt(v):v; x.dataset.v=v;
    x.onclick=()=>{set(v);draw();};b.appendChild(x);});
}
function marks(){
  [["who",v=>v===P],["qs",v=>v===Q],["st",v=>v===MODE]].forEach(([id,f])=>
    el(id).querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",f(b.dataset.v))));
}
function tagsFor(d,t){
  const out=[];
  if(d.r==="nonsense") out.push(["degenerate","nons"]);
  else if(d.r==="assistant") out.push(["assistant","plain"]);
  if(d.c===t) out.push(["named "+t,"hit"]);
  else if(d.r!=="nonsense") out.push([d.c&&d.c!=="none"?"read as "+d.c:"no character","miss"]);
  return out;
}
function draw(){
  marks();
  el("qt").innerHTML="<b>Q.</b> "+V.questions[Q];
  const r8=(V.recipes[P]||[]).map(([n,w])=>`${w>=0?"+":""}${w.toFixed(2)} ${n}`).join("  ");
  el("rc").innerHTML=`<b>${P}</b> rebuilt from 8 other personas:<br>${r8}`;
  const o=el("out"); o.innerHTML="";
  for(const a of V.arms){
    const s = MODE==="best" ? V.best[a+"|"+P] : MODE;
    const d = ((V.data[P]||{})[a]||{})[s]||{};
    const row=d[Q];
    const div=document.createElement("div"); div.className="spec";
    const tags=row?tagsFor(row,P).map(([txt,c])=>`<span class="tag ${c}">${txt}</span>`).join(""):"";
    div.innerHTML=`<div class="side"><div class="arm">${V.label[a]}</div>
        <div class="sub">strength s = ${s}</div>
        <div class="tags">${tags}</div></div>
      <div class="body${row?"":" none"}">${row?row.t:"not run at this strength"}</div>`;
    o.appendChild(div);
  }
}
chips("who",V.personas,0,v=>P=v);
chips("qs",Object.keys(V.questions),0,v=>Q=v,v=>"Q"+(+v+1));
chips("st",["best"].concat(V.strengths),0,v=>MODE=v,v=>v==="best"?"best per arm":"s="+v);
draw();
</script>
"""
open("attr_analysis/steer_examples.html","w").write(HTML.replace("__V__",V))
print("built", (len(HTML)+len(V))/1e6, "MB")
