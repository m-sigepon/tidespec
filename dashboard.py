"""TideSpec live throughput dashboard.

Single-file stdlib web server: serves a dark dashboard page and proxies
streaming chat completions to the sglang server on :30000, so the browser can
measure prefill (TTFT) and decode throughput in real time. A second endpoint
tails the serving container's own log lines as first-party evidence.

Usage:  python dashboard.py   ->  open http://<host>:8800
"""
import json
import subprocess
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SGLANG = "http://localhost:30000"
PORT = 8800
_CID = None


def server_logs():
    """Tail the sglang container's own log lines (the evidence pane)."""
    global _CID
    try:
        if not _CID:
            _CID = subprocess.check_output(
                ["docker", "ps", "-q", "--filter", "publish=30000"],
                text=True, timeout=10,
            ).split()[0]
        out = subprocess.check_output(
            ["docker", "logs", "--tail", "60", _CID],
            text=True, stderr=subprocess.STDOUT, timeout=10, encoding="utf-8",
            errors="replace",
        )
        keep = [
            l for l in out.splitlines()
            if ("Decode batch" in l or "Prefill batch" in l or "ngram-fusion" in l)
        ]
        return keep[-14:]
    except Exception as e:
        _CID = None
        return [f"(log unavailable: {e})"]


PAGE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>TideSpec Live</title>
<style>
  /* dark tokens from the validated reference palette (dark column) */
  :root {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
    --series:#3987e5;            /* categorical slot 1, dark step */
    --target:#ec835a;            /* status: serious (labelled) */
    --good:#0ca30c;              /* status: good */
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--page); color:var(--ink);
         font-family:system-ui,-apple-system,"Segoe UI",sans-serif; padding:20px; }
  header { display:flex; align-items:baseline; gap:14px; margin-bottom:16px; }
  h1 { font-size:19px; font-weight:700; letter-spacing:.02em; }
  h1 b { color:var(--series); font-weight:800; }
  .sub { color:var(--muted); font-size:12px; }
  .card { background:var(--surface); border:1px solid var(--ring); border-radius:14px; padding:16px; }
  .layout { display:grid; grid-template-columns:280px 1fr; gap:14px;
            height:calc(100vh - 330px); min-height:500px; }
  .layout > .card { height:100%; overflow:hidden; }
  .layout > .card:nth-child(1) { overflow-y:auto; }
  .layout > .card:nth-child(2) { display:flex; flex-direction:column; }
  button { display:block; width:100%; text-align:left; background:transparent;
           color:var(--ink-2); border:1px solid var(--ring); border-radius:10px;
           padding:11px 13px; margin:0 0 8px; cursor:pointer; font-size:13.5px;
           font-family:inherit; transition:background .12s,border-color .12s; }
  button:hover { background:rgba(255,255,255,.05); border-color:rgba(255,255,255,.22); }
  button.primary { border-color:var(--series); color:var(--ink); }
  button.primary::before { content:"▶ "; color:var(--series); }
  .lbl { font-size:11px; letter-spacing:.14em; color:var(--muted); margin:2px 0 10px; }
  .hero { display:flex; align-items:baseline; gap:18px; flex-wrap:wrap; }
  .hero .n { font-size:76px; font-weight:800; line-height:1;
             font-variant-numeric:tabular-nums; letter-spacing:-.01em; }
  .hero .u { color:var(--ink-2); font-size:15px; }
  .badge { font-size:12px; font-weight:700; border-radius:999px; padding:5px 12px;
           border:1px solid var(--ring); color:var(--muted); align-self:center; }
  .badge.flood { color:var(--good); border-color:var(--good); }
  .metrics { display:flex; gap:34px; margin-top:14px; flex-wrap:wrap; }
  .metric .v { font-size:24px; font-weight:700; }
  .metric .l { font-size:11px; color:var(--muted); margin-top:2px; }
  canvas { width:100%; height:132px; display:block; margin-top:14px; }
  .passes { margin-top:14px; }
  .prow { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:12.5px; }
  .prow .name { width:74px; color:var(--ink-2); }
  .prow .track { flex:1; height:14px; position:relative; }
  .prow .fill { position:absolute; inset:0 auto 0 0; background:var(--series);
                border-radius:0 4px 4px 0; transition:width .25s; }
  .prow .val { width:96px; color:var(--ink); font-weight:600; }
  #out { margin-top:14px; font:11.5px/1.55 Consolas,ui-monospace,monospace; color:var(--ink-2);
         flex:1; min-height:0; overflow-y:auto; white-space:pre-wrap; background:var(--page);
         border:1px solid var(--ring); border-radius:10px; padding:10px; }
  .console { margin-top:14px; }
  #srvlog { font:11px/1.65 Consolas,ui-monospace,monospace; color:var(--ink-2);
            background:var(--page); border:1px solid var(--ring); border-radius:10px;
            padding:10px; height:200px; overflow-y:auto; white-space:pre; }
  #srvlog em { color:var(--series); font-style:normal; font-weight:700; }
  #status { font-size:13px; color:var(--ink-2); min-height:18px; margin-top:2px; }
</style></head><body>
<header>
  <h1>Tide<b>Spec</b> LIVE</h1>
  <div class="sub">Qwen3.8-27B NVFP4 · RTX 5090 · draft-free copy mode (fusion v4.6) · greedy · reasoning_effort=low</div>
</header>
<div class="layout">
  <div class="card">
    <div class="lbl">TEST CASES</div>
    <button class="primary" onclick="runEdit()">LONG EDIT ×3 — 300超を長く見る</button>
    <button class="primary" onclick="runMarathon()">MARATHON ×2 — 定型2600tok</button>
    <button class="primary" onclick="runRepeat()">REPEAT ×5 — 潮を育てる</button>
    <button onclick="runCase('math')">math</button>
    <button onclick="runCase('ts')">ts-codegen</button>
    <button onclick="runCase('json')">json-out</button>
    <button onclick="runCase('boiler')">boilerplate</button>
    <div class="lbl" style="margin-top:14px">RUN</div>
    <div id="status">idle</div>
    <div class="passes" id="passes"></div>
  </div>
  <div class="card">
    <div class="hero">
      <span class="n" id="rate">0</span><span class="u">tok/s decode (live)</span>
      <span class="badge" id="tide">— 待機</span>
    </div>
    <div class="metrics">
      <span class="metric"><div class="v" id="ttft">–</div><div class="l">prefill / TTFT (s)</div></span>
      <span class="metric"><div class="v" id="toks">0</div><div class="l">tokens (est)</div></span>
      <span class="metric"><div class="v" id="avg">–</div><div class="l">pass avg tok/s（サーバー実測）</div></span>
      <span class="metric"><div class="v" id="peak">0</div><div class="l">peak tok/s (live)</div></span>
    </div>
    <canvas id="chart" width="1000" height="132"></canvas>
    <div id="out"></div>
  </div>
</div>
<div class="card console">
  <div class="lbl">SERVER CONSOLE — sglang本体ログ（accept len / gen throughput がサーバー側の実測値）</div>
  <div id="srvlog"></div>
</div>
<script>
const CASES = {
  math: "A factory produces widgets on 3 lines. Line A makes 120/hr with 2% defects, B makes 200/hr with 3.5% defects, C makes 150/hr with 1.5% defects. All run 16 hours/day. Compute total good widgets per day, overall defect rate, and how many hours line B alone would need to replace one full day of C's good output. Show your work.",
  ts: "Write a TypeScript function that parses a cron expression into its five fields, validates ranges, and returns a structured object. Include error handling and unit tests.",
  json: "Extract the following into strict JSON with keys name, version, deps (array), scripts (object): 'The project foo-cli v2.3.1 depends on chalk, commander and zod. It has scripts: build runs tsc, test runs vitest run, lint runs eslint src.' Output only JSON.",
  boiler: "Generate Pydantic v2 models and FastAPI CRUD endpoints (create/read/update/delete/list with pagination) for entities User, Team, and Project. Follow the exact same structure for each entity.",
};
function makeFetchFn(name) {
  return `def fetch_${name}(session, ${name}_id, retries=3):\n    for attempt in range(retries):\n        try:\n            resp = session.get(f"/api/${name}s/{${name}_id}", timeout=5)\n            resp.raise_for_status()\n            return resp.json()\n        except TransientError:\n            if attempt == retries - 1:\n                raise\n            time.sleep(2 ** attempt)\n\n`;
}
const ENTITIES = ['user','team','project','order','invoice','session_rec','report','webhook'];
const EDIT_PROMPT = "Here is a Python module:\n\n```python\n" + ENTITIES.map(makeFetchFn).join('') +
  "```\n\nRewrite the COMPLETE module changing only the timeout from 5 to 10 seconds. Keep all " +
  ENTITIES.length + " functions. Output the full code, no explanations.";
const MARATHON_PROMPT = "Generate Pydantic v2 models and FastAPI CRUD endpoints (create/read/update/delete/list with pagination) for entities User, Team, Project, Order, Invoice, and Webhook. Follow the exact same structure for each entity. Output code only.";

const cv = document.getElementById('chart');
const cx = cv.getContext('2d');
const css = getComputedStyle(document.documentElement);
const C = (n) => css.getPropertyValue(n).trim();
let hist = [];
function draw() {
  const W = cv.width, H = cv.height, MAX = 600, PADL = 34;
  cx.clearRect(0,0,W,H);
  cx.font = '10px system-ui';
  for (const g of [100,200,300,400,500]) {
    const y = H - g/MAX*(H-14) - 7;
    cx.strokeStyle = C('--grid'); cx.lineWidth = 1;
    if (g === 300) { cx.strokeStyle = C('--target'); cx.setLineDash([5,4]); }
    cx.beginPath(); cx.moveTo(PADL,y); cx.lineTo(W,y); cx.stroke(); cx.setLineDash([]);
    cx.fillStyle = g === 300 ? C('--target') : C('--muted');
    cx.fillText(g === 300 ? '300 目標' : String(g), 2, y+3);
  }
  if (hist.length > 1) {
    const n = Math.max(hist.length, 300);
    const X = (i) => PADL + i/n*(W-PADL);
    const Y = (v) => H - Math.min(v,MAX)/MAX*(H-14) - 7;
    cx.beginPath();
    hist.forEach((v,i)=> i ? cx.lineTo(X(i),Y(v)) : cx.moveTo(X(i),Y(v)));
    cx.strokeStyle = C('--series'); cx.lineWidth = 2; cx.stroke();
    cx.lineTo(X(hist.length-1), H-7); cx.lineTo(PADL, H-7); cx.closePath();
    cx.fillStyle = 'rgba(57,135,229,0.12)'; cx.fill();
  }
}
function estTokens(s){ return s.length / 3.6; }

async function runOne(prompt, maxTokens, label) {
  const st=document.getElementById('status'); st.textContent = label + ' …';
  const out=document.getElementById('out'); out.textContent='';
  hist=[]; let text='', t0=performance.now(), tFirst=0, peak=0, win=[];
  const res = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({prompt, max_tokens: maxTokens})});
  const rd = res.body.getReader(); const dec = new TextDecoder(); let buf='', usage=null;
  while (true) {
    const {done, value} = await rd.read(); if (done) break;
    buf += dec.decode(value, {stream:true});
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const evt = buf.slice(0, idx); buf = buf.slice(idx+2);
      for (const line of evt.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') continue;
        try {
          const j = JSON.parse(payload);
          if (j.usage) usage = j.usage;
          const d = j.choices && j.choices[0] && j.choices[0].delta;
          if (!d) continue;
          const piece = (d.content||'') + (d.reasoning_content||'');
          if (piece) {
            if (!tFirst) { tFirst = performance.now();
              document.getElementById('ttft').textContent = ((tFirst-t0)/1000).toFixed(2); }
            text += piece;
            const now = performance.now(), tk = estTokens(text);
            win.push([now, tk]);
            while (win.length && now - win[0][0] > 700) win.shift();
            let rate = 0;
            if (win.length > 1 && now - win[0][0] > 250) rate = (tk - win[0][1]) / ((now - win[0][0])/1000);
            rate = Math.min(rate, 900);
            peak = Math.max(peak, rate);
            hist.push(rate); if (hist.length > 300) hist.shift();
            document.getElementById('rate').textContent = rate.toFixed(0);
            const tide = document.getElementById('tide');
            if (rate >= 300) { tide.textContent = '満ち潮 ≥300'; tide.className='badge flood'; }
            else { tide.textContent = rate >= 150 ? '巡航' : '引き潮'; tide.className='badge'; }
            document.getElementById('toks').textContent = tk.toFixed(0);
            document.getElementById('peak').textContent = peak.toFixed(0);
            out.textContent = text.slice(-1200); out.scrollTop = out.scrollHeight;
            draw();
          }
        } catch(e){}
      }
    }
  }
  const tEnd = performance.now();
  let avg = null;
  if (usage && tFirst) avg = usage.completion_tokens / ((tEnd - tFirst)/1000);
  document.getElementById('avg').textContent = avg ? avg.toFixed(1) : '–';
  st.textContent = label + ' 完了';
  return avg;
}
function addPass(label, avg) {
  const p = document.getElementById('passes');
  const row = document.createElement('div'); row.className = 'prow';
  row.innerHTML = '<span class="name"></span><span class="track"><span class="fill" style="width:0"></span></span><span class="val"></span>';
  row.querySelector('.name').textContent = label;
  row.querySelector('.val').textContent = avg.toFixed(1) + ' tok/s';
  p.appendChild(row);
  requestAnimationFrame(()=>{ row.querySelector('.fill').style.width = Math.min(avg/600*100,100) + '%'; });
}
async function runEdit() {
  document.getElementById('passes').innerHTML='';
  for (let i=1;i<=3;i++) { const a = await runOne(EDIT_PROMPT, 1800, 'LONG EDIT '+i+'/3'); if (a) addPass('edit '+i, a); }
}
async function runMarathon() {
  document.getElementById('passes').innerHTML='';
  for (let i=1;i<=2;i++) { const a = await runOne(MARATHON_PROMPT, 2600, 'MARATHON '+i+'/2'); if (a) addPass('mara '+i, a); }
}
async function runRepeat() {
  document.getElementById('passes').innerHTML='';
  for (let i=1;i<=5;i++) { const a = await runOne(CASES.ts, 700, 'REPEAT '+i+'/5'); if (a) addPass('pass '+i, a); }
}
async function runCase(k) {
  document.getElementById('passes').innerHTML='';
  const a = await runOne(CASES[k], 800, k); if (a) addPass(k, a);
}
const seenLogs = new Set();
async function pollLogs() {
  try {
    const r = await fetch('/api/logs'); const j = await r.json();
    const el = document.getElementById('srvlog');
    for (const l of j.lines) {
      if (seenLogs.has(l)) continue;
      seenLogs.add(l);
      const html = l
        .replace(/(gen throughput \(token\/s\): [0-9.]+)/, '<em>$1</em>')
        .replace(/(accept len: [0-9.]+)/, '<em>$1</em>');
      el.insertAdjacentHTML('afterbegin', html + '\n');
    }
    while (el.childNodes.length > 400) el.removeChild(el.lastChild);
  } catch(e){}
}
setInterval(pollLogs, 1000); pollLogs();
draw();
</script></body></html>
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/logs":
            body = json.dumps({"lines": server_logs()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/run":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        payload = json.dumps({
            "model": "RadixArk/Qwen3.8-27B-NVFP4",
            "messages": [{"role": "user", "content": req["prompt"]}],
            "max_tokens": int(req.get("max_tokens", 700)),
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }).encode()
        up = urllib.request.Request(
            SGLANG + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with urllib.request.urlopen(up, timeout=600) as r:
                while True:
                    chunk = r.read(512)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as e:
            try:
                self.wfile.write(f"data: {{\"error\": \"{e}\"}}\n\n".encode())
            except Exception:
                pass


if __name__ == "__main__":
    print(f"TideSpec dashboard: http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
