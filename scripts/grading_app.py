"""Multi-grader ERIN hand-labelling app — cluster-internal only (NHS IG: report
text never leaves the institutional network). Stdlib-only HTTP server + SQLite.

v2 (2026-10-02): one SECTION per screen, keyboard-only, timer, CSV export.
v2.2 (2026-10-06): sessions of 20, daily goal and day-streak removed — one continuous queue per grader; items may carry
highlight_found=null (no highlighting requested) in which case no warning is shown. Items come from
hand_grading/items_<grader>.json (built by analysis/erin_handlabel/code/build_items.py);
graders without an item file fall back to the legacy per-report flow at /legacy.
No jury/model/rule grade is ever sent to the browser.

Run:  env GRADING_PASS=... python grading_app.py [port]
Data: /mnt/scratche/slow/fmlab/zuberi01/hand_grading/{cases.json,items_<grader>.json,grading.db}
Export: /sexport?pw=...  (also written to hand_grading/labels_<grader>.csv on every save)
"""
import csv, datetime as dt, io, json, os, sqlite3, sys, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

H = os.environ.get("GRADING_HOME", "/mnt/scratche/slow/fmlab/zuberi01/hand_grading")
PASS = os.environ.get("GRADING_PASS", "change-me")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8471
CASES = json.load(open(H + "/cases.json"))
CORE = [c for c in CASES if c["block"] == "core"]
POOL = [c for c in CASES if c["block"] == "pool"]
BY_CASE = {c["case"]: c for c in CASES}
GRADES = ["NDBE", "IND", "LGD", "HGD", "CANCER", "NORMAL_OTHER", "CANT_GRADE"]
SUBTYPES = ["adenocarcinoma", "squamous", "signet_ring", "post_neoadjuvant_tx_effect", "other_cancer"]
SGRADES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
SITES = ["OESOPHAGUS", "GOJ", "STOMACH", "DUODENUM", "COLORECTAL", "OTHER"]

db = sqlite3.connect(H + "/grading.db", check_same_thread=False, timeout=15, isolation_level=None)  # autocommit: no implicit transaction can be left open
db.execute("""CREATE TABLE IF NOT EXISTS labels
  (grader TEXT, case_name TEXT, payload TEXT, note TEXT,
   ts DATETIME DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (grader, case_name))""")
db.execute("""CREATE TABLE IF NOT EXISTS assignments
  (grader TEXT, case_name TEXT, ord INTEGER, PRIMARY KEY (grader, case_name))""")
db.execute("""CREATE TABLE IF NOT EXISTS section_labels
  (grader TEXT, item_id TEXT, case_id TEXT, section TEXT, grades TEXT, ambiguous INTEGER, imc INTEGER, p53 INTEGER,
   site TEXT, site_source TEXT, note TEXT, seconds REAL, session_no INTEGER, ts TEXT, PRIMARY KEY (grader, item_id))""")
db.commit()
lock = threading.Lock()

def items_for(grader):
    p = f"{H}/items_{grader}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def slabels(grader):
    rows = db.execute("SELECT item_id, grades, ambiguous, imc, p53, site, site_source, note, seconds, session_no, ts "
                      "FROM section_labels WHERE grader=?", (grader,)).fetchall()
    return {r[0]: {"grades": json.loads(r[1]), "ambiguous": bool(r[2]), "imc": bool(r[3]), "p53": bool(r[4]), "site": r[5],
                   "site_source": r[6], "note": r[7], "seconds": r[8], "session_no": r[9], "ts": r[10]} for r in rows}

def stats_for(grader):
    """Total labels, today's count, median seconds per section."""
    rows = db.execute("SELECT substr(ts,1,10), COUNT(*) FROM section_labels WHERE grader=? GROUP BY 1", (grader,)).fetchall()
    per_day = {d: n for d, n in rows}
    secs = [r[0] for r in db.execute("SELECT seconds FROM section_labels WHERE grader=? AND seconds IS NOT NULL", (grader,)).fetchall()]
    secs.sort(); med = secs[len(secs) // 2] if secs else None
    return {"today": per_day.get(dt.date.today().isoformat(), 0), "total_labels": sum(per_day.values()), "median_seconds_all": med}

def write_csv(grader):
    rows = db.execute("SELECT item_id, case_id, section, grades, ambiguous, imc, p53, site, site_source, note, ts, seconds, session_no "
                      "FROM section_labels WHERE grader=? ORDER BY ts", (grader,)).fetchall()
    with open(f"{H}/labels_{grader}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["section_id", "case_id", "section", "grades", "ambiguous", "imc", "p53", "site", "site_source", "note", "timestamp", "seconds", "session"])
        for r in rows: w.writerow([r[0], r[1], r[2], "|".join(json.loads(r[3]))] + list(r[4:]))
    os.chmod(f"{H}/labels_{grader}.csv", 0o600)

def assign(grader):
    with lock:
        rows = db.execute("SELECT case_name FROM assignments WHERE grader=? ORDER BY ord", (grader,)).fetchall()
        if rows: return [r[0] for r in rows]
        used = {r[0] for r in db.execute("SELECT DISTINCT case_name FROM assignments")}
        fresh = [c["case"] for c in POOL if c["case"] not in used][:20]
        order = [c["case"] for c in CORE] + fresh
        db.executemany("INSERT OR IGNORE INTO assignments VALUES (?,?,?)", [(grader, c, i) for i, c in enumerate(order)])
        db.commit()
        return order

# ---------------------------------------------------------------- keyboard reducer (pure; unit-tested with node)
REDUCER_JS = r"""
const SGRADES=["NORMAL_OTHER","NDBE","IND","LGD","HGD","CANCER"];
const SITES=["OESOPHAGUS","GOJ","STOMACH","DUODENUM","COLORECTAL","OTHER"];
function freshSel(item){return {grades:[],ambiguous:false,imc:false,p53:false,site:item.site_prefill,site_source:item.site_source,note:""}}
// reduce(sel, key, inNote) -> {sel, action}  action: null | "confirm" | "undo" | "note" | "blur" | "reject"
function reduce(sel, key, inNote){
  sel=JSON.parse(JSON.stringify(sel));
  if(inNote){ if(key==="Escape") return {sel,action:"blur"}; if(key==="Enter+mod") return {sel,action:"confirm"}; return {sel,action:null}; }
  if(/^[1-6]$/.test(key)){const g=SGRADES[+key-1];const i=sel.grades.indexOf(g);if(i>=0)sel.grades.splice(i,1);else sel.grades.push(g);return {sel,action:null}}
  if(key==="a"){sel.ambiguous=!sel.ambiguous;return {sel,action:null}}
  if(key==="i"){sel.imc=!sel.imc;return {sel,action:null}}
  if(key==="p"){sel.p53=!sel.p53;return {sel,action:null}}
  if(key==="s"){sel.site=SITES[(SITES.indexOf(sel.site)+1)%SITES.length];sel.site_source="grader";return {sel,action:null}}
  if(key==="n"){return {sel,action:"note"}}
  if(key==="z"){return {sel,action:"undo"}}
  if(key==="Enter"){ if(sel.grades.length||sel.ambiguous) return {sel,action:"confirm"}; return {sel,action:"reject"}; }
  return {sel,action:null};
}
if(typeof module!=="undefined") module.exports={reduce,freshSel,SGRADES,SITES};
"""

PAGE = """<!doctype html><meta charset="utf-8"><title>ERIN grading</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d23;--mute:#667085;--line:#e4e7ec;--acc:#2563eb;--ok:#059669;--mark:#fff3bf}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:14px 18px 40px}
.top{display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between;padding:8px 0 12px;border-bottom:1px solid var(--line)}
.bars{flex:1 1 320px;min-width:260px}.bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden;margin:4px 0}.bar i{display:block;height:100%;background:var(--acc)}.bar.all i{background:#94a3b8}
.stat{font-size:13px;color:var(--mute)}.stat b{color:var(--ink)}
.goal{font-size:13px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:var(--card)}.goal.done{background:#ecfdf5;border-color:var(--ok);color:var(--ok)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;margin-top:14px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.rep{max-height:72vh;overflow-y:auto}.rep h4{margin:12px 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
.rep pre{white-space:pre-wrap;font-family:Georgia,"Times New Roman",serif;font-size:15.5px;line-height:1.55;margin:0}
mark{background:var(--mark);border-left:4px solid #f59e0b;padding:1px 4px;display:inline-block}
.hdr{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}.hdr .sec{font-size:22px;font-weight:700}.hdr .mute{color:var(--mute);font-size:13px}
.keys{position:sticky;top:12px}
.k{display:flex;align-items:center;gap:10px;padding:7px 10px;border:1px solid var(--line);border-radius:9px;margin:5px 0;cursor:pointer;user-select:none}
.k:hover{border-color:var(--acc)}.k.on{background:#eff6ff;border-color:var(--acc);font-weight:600}
kbd{font:12px ui-monospace,Menlo,monospace;background:#f1f5f9;border:1px solid var(--line);border-radius:5px;padding:1px 7px;min-width:26px;text-align:center}
.flag.on{background:#fff7ed;border-color:#f59e0b}
textarea{width:100%;min-height:58px;font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px}
.confirm{width:100%;margin-top:10px;padding:12px;font-size:16px;font-weight:600;color:#fff;background:var(--ok);border:0;border-radius:10px;cursor:pointer}
.confirm:disabled{background:#cbd5e1}
.help{font-size:12px;color:var(--mute);margin-top:10px;line-height:1.7}
.end{max-width:560px;margin:60px auto;text-align:center}.end h2{font-size:26px;margin:0 0 6px}.end .big{font-size:48px;font-weight:700;margin:14px 0}
.end button{font-size:18px;padding:14px 30px;border-radius:12px;border:0;background:var(--acc);color:#fff;cursor:pointer;margin:6px}
.end button.secondary{background:#fff;color:var(--ink);border:1px solid var(--line)}
.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1a1d23;color:#fff;padding:8px 14px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .15s}.toast.show{opacity:.95}
input{padding:8px;border:1px solid var(--line);border-radius:8px;font-size:14px}
</style>
<div class="wrap" id="app"></div><div class="toast" id="toast"></div>
<script>__REDUCER__</script>
<script>
let S={name:localStorage.getItem("g_name")||"",pw:localStorage.getItem("g_pw")||"",items:[],labels:{},stats:{},idx:0,sel:null,inNote:false,shownAt:0,sessStart:0,sessTimes:[],view:"login"};
const app=document.getElementById("app");
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function toast(m){const t=document.getElementById("toast");t.textContent=m;t.classList.add("show");clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove("show"),1400)}
function fmt(s){s=Math.max(0,Math.round(s));return s>=3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`:`${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`}
function median(a){if(!a.length)return null;const b=[...a].sort((x,y)=>x-y);return b[Math.floor(b.length/2)]}
function login(msg=""){S.view="login";app.innerHTML=`<div class="card" style="max-width:520px;margin:60px auto"><h2 style="margin-top:0">ERIN section grading</h2>
<p class="stat">One section per screen. Keys: <kbd>1</kbd>–<kbd>6</kbd> grade, <kbd>a</kbd> ambiguous, <kbd>i</kbd> IMC, <kbd>p</kbd> p53, <kbd>s</kbd> site, <kbd>n</kbd> note, <kbd>Enter</kbd> confirm, <kbd>z</kbd> back.</p>
<p><input id="nm" placeholder="Your name" value="${esc(S.name)}"> <input id="pw" type="password" placeholder="Lab passphrase" value="${esc(S.pw)}"> <button onclick="start()">Start / resume</button></p><p class="stat" id="err">${esc(msg)}</p></div>`;document.getElementById("pw").addEventListener("keydown",e=>{if(e.key==="Enter")start()})}
async function start(){S.name=document.getElementById("nm").value.trim().toLowerCase();S.pw=document.getElementById("pw").value;if(!S.name)return;
 const r=await fetch(`/items?name=${encodeURIComponent(S.name)}&pw=${encodeURIComponent(S.pw)}`);
 if(r.status===403){document.getElementById("err").textContent="Wrong passphrase";return}
 if(r.status===404){location.href="/legacy";return}
 const d=await r.json();S.items=d.items;S.labels=d.labels;S.stats=d.stats;localStorage.setItem("g_name",S.name);localStorage.setItem("g_pw",S.pw);
 S.idx=S.items.findIndex(it=>!(it.item_id in S.labels));if(S.idx<0)S.idx=S.items.length;
 S.sessStart=Date.now();S.sessTimes=[];                     // timer = this sitting
 for(const b of loadQ()){S.labels[b.item_id]=b}            // unsynced labels from a previous visit count as done
 S.idx=S.items.findIndex(it=>!(it.item_id in S.labels));if(S.idx<0)S.idx=S.items.length;
 pump();
 if(S.idx>=S.items.length){allDone();return}
 show()}
function show(){S.view="item";const it=S.items[S.idx];const saved=S.labels[it.item_id];
 S.sel=saved?{grades:[...saved.grades],ambiguous:saved.ambiguous,imc:saved.imc,p53:saved.p53,site:saved.site,site_source:saved.site_source,note:saved.note||""}:freshSel(it);
 const draft=localStorage.getItem("draft_"+it.item_id);if(!saved&&draft){try{S.sel=JSON.parse(draft)}catch(e){}}
 S.shownAt=Date.now();S.inNote=false;S._lastReport=null;render();
 if(!document.querySelector(".rep mark")){const r=document.querySelector(".rep");if(r)r.scrollTop=0}}
function render(){const it=S.items[S.idx];if(!S._lastReport||S._lastReport!==it.item_id){S._lastReport=it.item_id;renderAll()}else renderPanel()}
function renderAll(){const it=S.items[S.idx],done=Object.keys(S.labels).length,left=S.items.length-S.idx;
 const med=median(S.sessTimes)??S.stats.median_seconds_all;const eta=med?fmt(left*med):"–";
 app.innerHTML=`<div class="top">
  <div class="bars"><div class="stat"><b>${done}</b> of ${S.items.length} labelled — <b>${left}</b> left</div>
   <div class="bar all"><i style="width:${100*done/S.items.length}%"></i></div></div>
  <div class="stat">this sitting <b id="timer">${fmt((Date.now()-S.sessStart)/1000)}</b> · median <b>${med?Math.round(med)+"s":"–"}</b>/section · ETA to finish <b>${eta}</b></div>
  <div class="stat">today <b>${S.stats.today}</b> · <span id="sync"></span></div></div>
 <div class="grid"><div class="card">
  <div class="hdr"><div><span class="sec">Section ${esc(it.section)}</span> <span class="mute">(${it.pos_in_case} of ${it.sections_in_case.length} in this report: ${it.sections_in_case.join(", ")})</span></div><div class="mute">${esc(it.case_id)}${S.labels[it.item_id]?" · previously saved":""}</div></div>
  ${it.highlight_found===false?'<div class="stat" style="color:#b45309">No lettered section '+esc(it.section)+' found in the text — read the whole report.</div>':""}
  <div class="rep">${it.html}</div></div>
 <div class="card keys" id="panel"></div></div>`;
 renderPanel();syncBadge();
 const r0=document.querySelector(".rep mark");if(r0)r0.scrollIntoView({block:"center"});
 clearInterval(S._t);S._t=setInterval(()=>{const t=document.getElementById("timer");if(t)t.textContent=fmt((Date.now()-S.sessStart)/1000)},1000)}
function renderPanel(){const it=S.items[S.idx];const panel=document.getElementById("panel");if(!panel)return;
 panel.innerHTML=`
  ${SGRADES.map((g,i)=>`<div class="k ${S.sel.grades.includes(g)?"on":""}" onclick="key('${i+1}')"><kbd>${i+1}</kbd>${g.replace("_"," / ")}</div>`).join("")}
  <div style="height:6px"></div>
  <div class="k flag ${S.sel.ambiguous?"on":""}" onclick="key('a')"><kbd>a</kbd>ambiguous</div>
  <div class="k flag ${S.sel.imc?"on":""}" onclick="key('i')"><kbd>i</kbd>intramucosal (IMC) note</div>
  <div class="k flag ${S.sel.p53?"on":""}" onclick="key('p')"><kbd>p</kbd>p53 mentioned</div>
  <div class="k" onclick="key('s')"><kbd>s</kbd>site: <b>${esc(S.sel.site)}</b> <span class="stat">(${S.sel.site_source})</span></div>
  <div class="k" onclick="key('n')"><kbd>n</kbd>note</div>
  <textarea id="note" placeholder="free-text note (Esc to leave, ⌘/Ctrl+Enter to confirm)">${esc(S.sel.note)}</textarea>
  <button class="confirm" ${S.sel.grades.length||S.sel.ambiguous?"":"disabled"} onclick="key('Enter')">Confirm &amp; next <kbd style="color:#000">Enter</kbd></button>
  <div class="help"><kbd>z</kbd> back one · grades toggle, several allowed · site cycles with <kbd>s</kbd></div>`;
 const ta=document.getElementById("note");ta.addEventListener("focus",()=>{S.inNote=true});ta.addEventListener("blur",()=>{S.inNote=false;S.sel.note=ta.value;saveDraft()});ta.addEventListener("input",()=>{S.sel.note=ta.value;saveDraft()});
 const done=Object.keys(S.labels).length;const h=document.querySelector(".bars .stat");if(h)h.innerHTML=`<b>${done}</b> of ${S.items.length} labelled — <b>${S.items.length-S.idx}</b> left`;const b=document.querySelector(".bar i");if(b)b.style.width=(100*done/S.items.length)+"%";}

function saveDraft(){try{localStorage.setItem("draft_"+S.items[S.idx].item_id,JSON.stringify(S.sel))}catch(e){}}
function key(k){if(S.view!=="item")return;const r=reduce(S.sel,k,S.inNote);S.sel=r.sel;
 if(r.action==="note"){document.getElementById("note").focus();return}
 if(r.action==="blur"){document.getElementById("note").blur();render();return}
 if(r.action==="undo"){if(S.idx>0){S.idx--;show();toast("back one")}return}
 if(r.action==="reject"){toast("pick a grade (1–6) or mark ambiguous (a)");return}
 if(r.action==="confirm"){confirm_();return}
 saveDraft();render()}
function confirm_(){const it=S.items[S.idx];const secs=(Date.now()-S.shownAt)/1000;
 const body={name:S.name,pw:S.pw,item_id:it.item_id,case_id:it.case_id,section:it.section,grades:[...S.sel.grades],ambiguous:S.sel.ambiguous,imc:S.sel.imc,p53:S.sel.p53,site:S.sel.site,site_source:S.sel.site_source,note:S.sel.note,seconds:Math.round(secs*10)/10,session_no:1};
 S.labels[it.item_id]=body;S.sessTimes.push(secs);localStorage.removeItem("draft_"+it.item_id);
 enqueue(body);                                  // save in the background; UI never waits on the network
 S.idx++;if(S.idx>=S.items.length){allDone();return}show()}
// ---- background save queue: persisted in localStorage, retried with backoff, flushed on load ----
function qkey(){return "queue_"+S.name}
function loadQ(){try{return JSON.parse(localStorage.getItem(qkey())||"[]")}catch(e){return []}}
function saveQ(q){try{localStorage.setItem(qkey(),JSON.stringify(q))}catch(e){}}
function enqueue(body){const q=loadQ().filter(b=>b.item_id!==body.item_id);q.push(body);saveQ(q);pump()}
let pumping=false,backoff=500;
async function pump(){if(pumping)return;pumping=true;
 while(true){const q=loadQ();if(!q.length)break;const body=q[0];
  try{const ctl=new AbortController();const to=setTimeout(()=>ctl.abort(),8000);
   const r=await fetch("/slabel",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal:ctl.signal});clearTimeout(to);
   if(r.status===400||r.status===403){console.warn("rejected",await r.text());saveQ(loadQ().filter(b=>b.item_id!==body.item_id));continue}
   if(!r.ok)throw new Error("http "+r.status);
   const d=await r.json();S.stats=d.stats;saveQ(loadQ().filter(b=>b.item_id!==body.item_id));backoff=500;syncBadge()}
  catch(e){syncBadge();await new Promise(r=>setTimeout(r,backoff));backoff=Math.min(backoff*2,15000)}}
 pumping=false;syncBadge()}
function syncBadge(){const n=loadQ().length;const el=document.getElementById("sync");if(el){el.textContent=n?`saving ${n}…`:"saved";el.style.color=n?"#b45309":"var(--ok)"}}
function allDone(){S.view="done";app.innerHTML=`<div class="card end"><h2>All ${S.items.length} sections labelled.</h2><p class="stat">Thank you — export: labels_${esc(S.name)}.csv on the cluster.</p><p><button onclick="localStorage.removeItem('g_name');location.href='/?switch'">Switch grader</button></p></div>`}
document.addEventListener("keydown",e=>{
 if(S.view!=="item")return;
 const inNote=document.activeElement&&document.activeElement.id==="note";
 if(inNote){if(e.key==="Escape"){e.preventDefault();key("Escape")}else if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)){e.preventDefault();S.inNote=true;const r=reduce(S.sel,"Enter+mod",true);if(r.action==="confirm"){document.activeElement.blur();S.inNote=false;if(S.sel.grades.length||S.sel.ambiguous)confirm_();else toast("pick a grade first")}}return}
 if(e.metaKey||e.ctrlKey||e.altKey)return;
 if(/^[1-6]$|^[aipsnz]$|^Enter$/.test(e.key)){e.preventDefault();key(e.key)}});
if(S.name&&S.pw&&!/[?&]switch/.test(location.search)){login();start()}else login();
</script>"""

LEGACY_PAGE = """<!doctype html><meta charset="utf-8"><title>ERIN Lab Grading (legacy)</title>
<style>body{font-family:-apple-system,sans-serif;background:#f7f7f8;margin:0;padding:24px;display:flex;justify-content:center;color:#1a1a1a}.wrap{max-width:860px;width:100%}.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:22px}pre{white-space:pre-wrap;font-family:Georgia,serif;font-size:15px;line-height:1.55;max-height:50vh;overflow-y:auto}button{font-size:14px;padding:10px 16px;border-radius:8px;border:1px solid #d1d5db;background:#fff;cursor:pointer;margin:3px}button.sel{background:#2563eb;color:#fff}input,textarea{padding:8px;border:1px solid #d1d5db;border-radius:8px;font-size:14px}.bar{height:8px;background:#e5e7eb;border-radius:4px;margin:12px 0}.fill{height:100%;background:#2563eb;border-radius:4px}.mut{color:#666;font-size:13px}</style>
<div class="wrap" id="app"></div>
<script>
const GRADES=["NDBE","IND","LGD","HGD","CANCER","NORMAL_OTHER","CANT_GRADE"];const SUBTYPES=["adenocarcinoma","squamous","signet_ring","post_neoadjuvant_tx_effect","other_cancer"];const SECTIONS=["A","B","C","D","E","F","G","H","whole_report"];
let S={name:localStorage.getItem("g_name")||"",pw:localStorage.getItem("g_pw")||"",cases:[],labels:{},idx:0,rows:[]};const app=document.getElementById("app");
function login(){app.innerHTML=`<div class="card"><h2>ERIN report grading (per section)</h2><p class="mut">Reports contain lettered specimen sections (A, B, C...). For EACH section, tick every grade present. If CANCER is present, also tick its subtype(s). Use "whole report" only when the report has no lettered sections.</p><p><input id="nm" placeholder="Your name" value="${S.name}"> <input id="pw" type="password" placeholder="Lab passphrase" value="${S.pw}"> <button onclick="start()">Start / Resume</button></p><p class="mut" id="err"></p></div>`}
async function start(){S.name=document.getElementById("nm").value.trim().toLowerCase();S.pw=document.getElementById("pw").value;if(!S.name)return;const r=await fetch(`/session?name=${encodeURIComponent(S.name)}&pw=${encodeURIComponent(S.pw)}`);if(!r.ok){document.getElementById("err").textContent="Wrong passphrase";return}localStorage.setItem("g_name",S.name);localStorage.setItem("g_pw",S.pw);const d=await r.json();S.cases=d.cases;S.labels=d.labels;S.idx=S.cases.findIndex(c=>!(c.case in S.labels));if(S.idx<0)S.idx=0;load();render()}
function load(){const c=S.cases[S.idx];S.rows=(S.labels[c.case]&&JSON.parse(JSON.stringify(S.labels[c.case])))||[{section:"A",grades:[],subtypes:[]}];}
function rowHtml(r,i){const cancer=r.grades.includes("CANCER");return `<div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px;margin:8px 0"><b>Section</b> <select onchange="S.rows[${i}].section=this.value">${SECTIONS.map(x=>`<option ${r.section===x?"selected":""}>${x}</option>`).join("")}</select><button style="float:right;color:#b91c1c" onclick="S.rows.splice(${i},1);render()">remove</button><br>${GRADES.map(g=>`<label style="margin-right:10px"><input type="checkbox" ${r.grades.includes(g)?"checked":""} onchange="tog(${i},'grades','${g}',this.checked)"> ${g.replace(/_/g," ")}</label>`).join("")}${cancer?`<div style="margin-top:6px"><i>cancer subtype(s):</i> ${SUBTYPES.map(t=>`<label style="margin-right:10px"><input type="checkbox" ${r.subtypes.includes(t)?"checked":""} onchange="tog(${i},'subtypes','${t}',this.checked)"> ${t.replace(/_/g," ")}</label>`).join("")}</div>`:""}</div>`}
function tog(i,f,v,on){const a=S.rows[i][f];const j=a.indexOf(v);if(on&&j<0)a.push(v);if(!on&&j>=0)a.splice(j,1);render()}
function render(){const c=S.cases[S.idx],done=Object.keys(S.labels).length;app.innerHTML=`<div class="card"><div class="mut">${S.name} &middot; report ${S.idx+1}/${S.cases.length} (${c.case})</div><div class="bar"><div class="fill" style="width:${100*done/S.cases.length}%"></div></div><pre>${c.text.replace(/</g,"&lt;")}</pre><div id="rows">${S.rows.map(rowHtml).join("")}</div><button onclick="S.rows.push({section:'A',grades:[],subtypes:[]});render()">+ add section</button><textarea id="note" placeholder="optional note" style="width:100%;box-sizing:border-box;margin-top:8px"></textarea><p><button onclick="move(-1)">&larr; Prev</button> <button style="background:#059669;color:#fff;border-color:#059669" onclick="saveNext()">Save &amp; next &rarr;</button> <button onclick="move(1)">skip &rarr;</button> <span class="mut">${done}/${S.cases.length} saved</span></p></div>`}
async function saveNext(){const c=S.cases[S.idx];const rows=S.rows.filter(r=>r.grades.length>0);if(!rows.length){alert("Tick at least one grade in at least one section.");return}const r=await fetch("/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:S.name,pw:S.pw,case:c.case,sections:rows,note:document.getElementById("note").value})});if(!r.ok){alert("save failed — retry");return}S.labels[c.case]=rows;if(S.idx<S.cases.length-1)S.idx++;load();render()}
function move(d){S.idx=Math.min(S.cases.length-1,Math.max(0,S.idx+d));load();render()}
login();
</script>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _html(self, s):
        b = s.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        try: self._get()
        except Exception as e:
            try: db.rollback()
            except Exception: pass
            print("GET error:", repr(e), flush=True); self._json({"err": "server error"}, 500)
    def do_POST(self):
        try: self._post()
        except Exception as e:
            try: db.rollback()
            except Exception: pass
            print("POST error:", repr(e), flush=True); self._json({"err": "server error"}, 500)
    def _get(self):
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/": return self._html(PAGE.replace("__REDUCER__", REDUCER_JS))
        if u.path == "/legacy": return self._html(LEGACY_PAGE)
        if u.path in ("/items", "/session", "/export", "/sexport") and q.get("pw") != PASS: return self._json({"err": "bad pw"}, 403)
        name = q.get("name", "").strip().lower()
        if u.path == "/items":
            items = items_for(name)
            if items is None: return self._json({"err": "no item queue for this grader"}, 404)
            return self._json({"items": items, "labels": slabels(name), "stats": stats_for(name)})
        if u.path == "/session":
            if not name: return self._json({"err": "no name"}, 400)
            order = assign(name)
            labels = {c: json.loads(pl) for c, pl in db.execute("SELECT case_name, payload FROM labels WHERE grader=?", (name,)).fetchall()}
            return self._json({"cases": [{"case": c, "text": BY_CASE[c]["text"]} for c in order], "labels": labels})
        if u.path == "/export":
            rows = db.execute("SELECT grader, case_name, payload, note, ts FROM labels").fetchall()
            return self._json({"n": len(rows), "labels": rows})
        if u.path == "/sexport":
            rows = db.execute("SELECT item_id, grader, case_id, section, grades, ambiguous, imc, p53, site, site_source, note, ts, seconds, session_no FROM section_labels ORDER BY grader, ts").fetchall()
            buf = io.StringIO(); w = csv.writer(buf)
            w.writerow(["section_id", "grader", "case_id", "section", "grades", "ambiguous", "imc", "p53", "site", "site_source", "note", "timestamp", "seconds", "session"])
            for r in rows: w.writerow([r[0], r[1], r[2], r[3], "|".join(json.loads(r[4]))] + list(r[5:]))
            b = buf.getvalue().encode(); self.send_response(200); self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=labels.csv"); self.send_header("Content-Length", str(len(b))); self.end_headers(); return self.wfile.write(b)
        self._json({"err": "not found"}, 404)
    def _post(self):
        n = int(self.headers.get("Content-Length", 0)); d = json.loads(self.rfile.read(n) or b"{}")
        if d.get("pw") != PASS: return self._json({"err": "bad pw"}, 403)
        name = str(d.get("name", "")).strip().lower()
        if self.path == "/slabel":
            items = items_for(name) or []
            ids = {it["item_id"]: it for it in items}
            if d.get("item_id") not in ids: return self._json({"err": "unknown item"}, 400)
            grades = d.get("grades") or []
            if not isinstance(grades, list) or any(g not in SGRADES for g in grades) or (not grades and not d.get("ambiguous")): return self._json({"err": "bad grades"}, 400)
            site = d.get("site") if d.get("site") in SITES else ids[d["item_id"]]["site_prefill"]
            with lock:
                db.execute("INSERT OR REPLACE INTO section_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (name, d["item_id"], ids[d["item_id"]]["case_id"], ids[d["item_id"]]["section"], json.dumps(grades),
                            int(bool(d.get("ambiguous"))), int(bool(d.get("imc"))), int(bool(d.get("p53"))), site, str(d.get("site_source", ""))[:20],
                            str(d.get("note", ""))[:1000], float(d.get("seconds") or 0), int(d.get("session_no") or 0), dt.datetime.now().isoformat(timespec="seconds")))
                db.commit()
            try: write_csv(name)
            except Exception as e: print("csv write failed:", repr(e), flush=True)
            return self._json({"ok": True, "stats": stats_for(name)})
        if self.path == "/label":
            secs = d.get("sections")
            if d.get("case") not in BY_CASE or not isinstance(secs, list) or not secs: return self._json({"err": "bad"}, 400)
            for sec in secs:
                if not isinstance(sec.get("grades"), list) or any(g not in GRADES for g in sec["grades"]) or any(t not in SUBTYPES for t in sec.get("subtypes", [])):
                    return self._json({"err": "bad section"}, 400)
            with lock:
                db.execute("INSERT OR REPLACE INTO labels (grader, case_name, payload, note) VALUES (?,?,?,?)", (name, d["case"], json.dumps(secs), str(d.get("note", ""))[:500])); db.commit()
            return self._json({"ok": True})
        self._json({"err": "not found"}, 404)

print(f"grading app v2 on port {PORT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
