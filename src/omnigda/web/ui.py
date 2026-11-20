"""Embedded HTML/CSS/JS frontend for OmniGDA-Manip Interactive Studio."""

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OmniGDA-Manip Interactive Robotics Studio</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel2: #1f2633; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --green: #3fb950;
    --orange: #d29922; --red: #f85149; --purple: #bc8cff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); font-size: 13px; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--border); }
  header h1 { margin: 0; font-size: 17px; display: flex; align-items: center; gap: 10px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; background: #1f3b5c; color: var(--accent); border: 1px solid #305a8a; }
  .badge.green { background: #163322; color: var(--green); border-color: #245c39; }
  .badge.purple { background: #2c1e42; color: var(--purple); border-color: #543782; }
  .layout { display: grid; grid-template-columns: 1.35fr 1fr; gap: 14px; padding: 14px; max-width: 1600px; margin: 0 auto; }
  @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 14px; }
  .card h2 { margin: 0 0 10px 0; font-size: 14px; color: var(--text); display: flex; justify-content: space-between; align-items: center; }
  .tabs { display: flex; gap: 6px; }
  .tab { padding: 6px 12px; border-radius: 6px; background: var(--panel2); color: var(--muted); border: 1px solid var(--border); cursor: pointer; font-weight: 600; }
  .tab.active { background: #1f3b5c; color: #fff; border-color: var(--accent); }
  .viewport { position: relative; background: #000; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); min-height: 380px; display: flex; align-items: center; justify-content: center; }
  .viewport img { width: 100%; height: auto; display: block; object-fit: contain; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
  button { padding: 7px 12px; border-radius: 6px; border: 1px solid var(--border); background: var(--panel2); color: var(--text); cursor: pointer; font-weight: 600; }
  button:hover:not(:disabled) { border-color: var(--accent); background: #263144; }
  button.primary { background: #1f6feb; border-color: #388bfd; color: #fff; }
  button.success { background: #238636; border-color: #2ea043; color: #fff; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  input[type=text], input[type=number] { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; font-size: 13px; }
  .cmd-input { flex: 1; min-width: 200px; }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0 10px; }
  .chip { font-size: 11px; padding: 4px 9px; border-radius: 14px; background: var(--panel2); border: 1px solid var(--border); cursor: pointer; color: #c9d1d9; }
  .chip:hover { border-color: var(--accent); color: #fff; }
  .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
  .metric { background: var(--panel2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
  .metric .label { font-size: 11px; color: var(--muted); }
  .metric .val { font-size: 16px; font-weight: 700; margin-top: 2px; color: var(--accent); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  .bar-wrap { display: flex; height: 8px; width: 110px; background: #21262d; border-radius: 4px; overflow: hidden; }
  .bar-2d { background: #58a6ff; } .bar-3d { background: #3fb950; } .bar-txt { background: #bc8cff; }
  .log-box { background: #0a0d12; border: 1px solid var(--border); border-radius: 8px; padding: 10px; height: 150px; overflow-y: auto; font-family: ui-monospace, monospace; font-size: 11.5px; line-height: 1.5; white-space: pre-wrap; color: #c9d1d9; }
  .status-pill { font-size: 12px; color: var(--green); font-weight: 600; }
  .status-pill.busy { color: var(--orange); }
  .playback-bar { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .playback-bar input[type=range] { flex: 1; }
  .reg-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
</style>
</head>
<body>
<header>
  <h1>
    <span>OmniGDA-Manip</span>
    <span class="badge" id="hdr-backend">Backend: ...</span>
    <span class="badge green" id="hdr-cam">Camera: base_camera</span>
    <span class="badge purple" id="hdr-classes">Registered: 0</span>
  </h1>
  <div class="status-pill" id="status-pill">Ready</div>
</header>
<div class="layout">
  <div>
    <div class="card">
      <h2>
        <span>Live Perception &amp; Manipulation Viewport</span>
        <div class="tabs">
          <button class="tab active" onclick="setView('annotated')" id="tab-annotated">Annotated 2D+3D+Grasp</button>
          <button class="tab" onclick="setView('heatmap')" id="tab-heatmap">Confidence &amp; Entropy</button>
          <button class="tab" onclick="setView('playback')" id="tab-playback">Franka Playback (<span id="pb-count">0</span>)</button>
        </div>
      </h2>
      <div class="viewport"><img id="main-img" alt="Live Viewport" /></div>
      <div class="playback-bar" id="playback-controls" style="display:none;">
        <button onclick="togglePlay()" id="btn-play">Pause</button>
        <input type="range" id="pb-slider" min="0" max="0" value="0" oninput="seekFrame(this.value)" />
        <span id="pb-label">0 / 0</span>
      </div>
      <div class="metrics-grid">
        <div class="metric"><div class="label">2D SegmentGDA+</div><div class="val" id="m-seg2d">-- ms</div></div>
        <div class="metric"><div class="label">3D Multimodal GMDA</div><div class="val" id="m-gmda3d">-- ms</div></div>
        <div class="metric"><div class="label">6DoF Grasp Plan</div><div class="val" id="m-grasp">-- ms</div></div>
        <div class="metric"><div class="label">GPU VRAM / Views</div><div class="val" id="m-vram">--</div></div>
      </div>
    </div>
    <div class="card">
      <h2>
        <span>Detected 3D Objects &amp; Modality Weights</span>
        <span style="font-size:11px;color:var(--muted);"><span style="color:#58a6ff;">■ 2D</span> &nbsp; <span style="color:#3fb950;">■ 3D</span> &nbsp; <span style="color:#bc8cff;">■ Text</span></span>
      </h2>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>ID</th><th>Class</th><th>Conf</th><th>Entropy</th><th>OOD</th><th>Centroid (m)</th><th>Modality Fusion</th></tr></thead>
          <tbody id="det-tbody"><tr><td colspan="7" style="color:var(--muted);text-align:center;">Click "Register Objects" then "Run 2D+3D Perception"</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>
  <div>
    <div class="card">
      <h2><span>1. Simulation &amp; Camera Controls</span></h2>
      <div class="row">
        <label>Seed: <input type="number" id="inp-seed" value="42" style="width:72px;" /></label>
        <label>Objects: <input type="number" id="inp-num" value="6" min="3" max="7" style="width:60px;" /></label>
        <label><input type="checkbox" id="chk-novel" checked /> Include Novel Objects</label>
      </div>
      <div class="row">
        <button onclick="resetSim()">Reset Scene</button>
        <button onclick="switchCamera()">Switch Camera (Base ↔ NBV)</button>
        <button class="primary" onclick="runPerceive()">Run 2D+3D Perception</button>
      </div>
    </div>
    <div class="card">
      <h2><span>2. Online Few-Shot &amp; Zero-Shot Registration</span></h2>
      <div class="row">
        <button class="success" onclick="registerPreset('base')">+ Register Base Objects (2-shot)</button>
        <button class="success" onclick="registerPreset('novel')">+ Register Novel Unseen (cyan_t_tool, magenta_cube)</button>
      </div>
      <div class="row">
        <input type="text" id="zs-name" placeholder="Class name" style="width:130px;" />
        <input type="text" id="zs-prompt" placeholder="Zero-shot text prompt (e.g. a red ceramic mug)" class="cmd-input" />
        <button onclick="registerZeroShot()">+ Zero-Shot Text</button>
      </div>
      <div class="reg-tags" id="reg-tags"><span style="color:var(--muted);">No classes registered yet.</span></div>
    </div>
    <div class="card">
      <h2><span>3. Bilingual Multimodal Manipulation (EN / 中文)</span></h2>
      <div class="row">
        <input type="text" id="inp-cmd" class="cmd-input" value="pick up the cyan t tool" placeholder="输入中文或英文指令" />
        <button class="primary" onclick="runExecute()">Execute Grasp</button>
      </div>
      <div class="chips">
        <span class="chip" onclick="setCmd('pick up the cyan t tool')">pick up the cyan t tool (Novel)</span>
        <span class="chip" onclick="setCmd('抓起最左边的物体')">抓起最左边的物体</span>
        <span class="chip" onclick="setCmd('把绿色盒子放到蓝色瓶子旁边')">把绿色盒子放到蓝色瓶子旁边</span>
        <span class="chip" onclick="setCmd('找到洋红方块')">找到洋红方块</span>
        <span class="chip" onclick="setCmd('pick up the red mug')">pick up the red mug</span>
      </div>
      <div class="row" style="margin-bottom:0;">
        <label><input type="checkbox" id="chk-nbv" checked /> Enable NBV Active Perception</label>
        <label><input type="checkbox" id="chk-force-nbv" /> Force Multi-View NBV Demo</label>
      </div>
    </div>
    <div class="card">
      <h2><span>3D Semantic Scene Graph &amp; Execution Log</span></h2>
      <div class="log-box" id="log-box">Studio initialized. Ready for few-shot registration &amp; grasping.\n</div>
    </div>
  </div>
</div>
<script>
let currentView='annotated', latestImages={annotated:'',heatmap:''}, playbackFrames=[], pbIndex=0, pbTimer=null, pbPlaying=true;
function log(m){const b=document.getElementById('log-box');b.textContent+=`[${new Date().toLocaleTimeString()}] ${m}\n`;b.scrollTop=b.scrollHeight;}
function setBusy(b,l){const p=document.getElementById('status-pill');p.textContent=l||(b?'Running...':'Ready');p.className='status-pill'+(b?' busy':'');document.querySelectorAll('button').forEach(x=>{if(!x.classList.contains('tab')&&x.id!=='btn-play')x.disabled=b;});}
function setCmd(t){document.getElementById('inp-cmd').value=t;}
function setView(v){currentView=v;['annotated','heatmap','playback'].forEach(t=>document.getElementById('tab-'+t).classList.toggle('active',t===v));document.getElementById('playback-controls').style.display=(v==='playback'&&playbackFrames.length>0)?'flex':'none';renderViewport();}
function renderViewport(){const i=document.getElementById('main-img');if(currentView==='playback'&&playbackFrames.length>0){i.src='data:image/jpeg;base64,'+playbackFrames[pbIndex];document.getElementById('pb-slider').value=pbIndex;document.getElementById('pb-label').textContent=`${pbIndex+1} / ${playbackFrames.length}`;}else if(currentView==='heatmap'&&latestImages.heatmap){i.src='data:image/jpeg;base64,'+latestImages.heatmap;}else if(latestImages.annotated){i.src='data:image/jpeg;base64,'+latestImages.annotated;}}
function startPlayback(){if(pbTimer)clearInterval(pbTimer);pbPlaying=true;document.getElementById('btn-play').textContent='Pause';pbTimer=setInterval(()=>{if(!pbPlaying||!playbackFrames.length)return;pbIndex=(pbIndex+1)%playbackFrames.length;if(currentView==='playback')renderViewport();},110);}
function togglePlay(){pbPlaying=!pbPlaying;document.getElementById('btn-play').textContent=pbPlaying?'Pause':'Play';}
function seekFrame(v){pbIndex=parseInt(v,10)||0;renderViewport();}
function updateUI(d){
  if(d.backend)document.getElementById('hdr-backend').textContent='Backend: '+d.backend;
  if(d.camera)document.getElementById('hdr-cam').textContent='Camera: '+d.camera;
  if(d.registered_classes){document.getElementById('hdr-classes').textContent='Registered: '+d.registered_classes.length;document.getElementById('reg-tags').innerHTML=d.registered_classes.length?d.registered_classes.map(c=>`<span class="badge green">${c}</span>`).join(''):'<span style="color:var(--muted);">No classes registered yet.</span>';}
  if(d.annotated_jpg)latestImages.annotated=d.annotated_jpg;
  if(d.heatmap_jpg)latestImages.heatmap=d.heatmap_jpg;
  if(d.metrics){document.getElementById('m-seg2d').textContent=d.metrics.seg2d_ms!==undefined?`${d.metrics.seg2d_ms.toFixed(1)} ms`:'-- ms';document.getElementById('m-gmda3d').textContent=d.metrics.gmda3d_ms!==undefined?`${d.metrics.gmda3d_ms.toFixed(1)} ms`:'-- ms';document.getElementById('m-grasp').textContent=d.metrics.grasp_ms!==undefined?`${d.metrics.grasp_ms.toFixed(1)} ms`:'-- ms';document.getElementById('m-vram').textContent=(d.metrics.vram_mb!==undefined?`${d.metrics.vram_mb.toFixed(0)} MB`:'--')+(d.metrics.num_views!==undefined?` / ${d.metrics.num_views}v`:'');}
  if(d.detections){const tb=document.getElementById('det-tbody');tb.innerHTML=d.detections.length?d.detections.map(x=>{const w2=Math.round((x.w_2d||0)*100),w3=Math.round((x.w_3d||0)*100),wt=Math.max(0,100-w2-w3);return `<tr><td>#${x.instance_id}</td><td><b>${x.class_name}</b></td><td>${(x.confidence*100).toFixed(1)}%</td><td>${x.entropy.toFixed(2)}</td><td>${x.is_ood?`<span style="color:var(--red);font-weight:700;">OOD (${x.ood_score.toFixed(1)})</span>`:x.ood_score.toFixed(1)}</td><td>(${x.centroid.map(v=>v.toFixed(2)).join(', ')})</td><td><div class="bar-wrap" title="2D:${w2}% 3D:${w3}% Text:${wt}%"><div class="bar-2d" style="width:${w2}%"></div><div class="bar-3d" style="width:${w3}%"></div><div class="bar-txt" style="width:${wt}%"></div></div></td></tr>`;}).join(''):'<tr><td colspan="7" style="color:var(--muted);text-align:center;">No objects detected yet.</td></tr>';}
  if(d.playback_frames&&d.playback_frames.length){playbackFrames=d.playback_frames;pbIndex=0;document.getElementById('pb-count').textContent=playbackFrames.length;document.getElementById('pb-slider').max=playbackFrames.length-1;setView('playback');startPlayback();}else{renderViewport();}
}
async function api(u,b){const r=await fetch(u,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{method:'GET'});const d=await r.json();if(!r.ok)throw new Error(d.error||r.statusText);return d;}
async function init(){setBusy(true,'Loading Studio...');try{const d=await api('/api/state');updateUI(d);log(`Connected. Scene objects: ${(d.scene_objects||[]).join(', ')}`);}catch(e){log('ERROR: '+e.message);}setBusy(false,'Ready');}
async function resetSim(){setBusy(true,'Resetting ManiSkill 3...');try{const d=await api('/api/reset',{seed:parseInt(document.getElementById('inp-seed').value,10),num_objects:parseInt(document.getElementById('inp-num').value,10),include_novel:document.getElementById('chk-novel').checked});playbackFrames=[];document.getElementById('pb-count').textContent='0';setView('annotated');updateUI(d);log(`Reset scene (seed=${d.seed}): ${(d.scene_objects||[]).join(', ')}`);}catch(e){log('ERROR: '+e.message);}setBusy(false,'Ready');}
async function switchCamera(){setBusy(true,'Switching camera...');try{const d=await api('/api/switch_camera',{});setView('annotated');updateUI(d);log(`Switched active camera -> ${d.camera}`);}catch(e){log('ERROR: '+e.message);}setBusy(false,'Ready');}
async function registerPreset(g){setBusy(true,`Registering ${g} (2-shot)...`);try{const d=await api('/api/register_preset',{group:g,num_shots:2});updateUI(d);log(`Registered ${g} (${d.newly_registered.join(', ')}) in ${d.registration_ms.toFixed(1)} ms!`);}catch(e){log('ERROR: '+e.message);}setBusy(false,'Ready');}
async function registerZeroShot(){const c=document.getElementById('zs-name').value.trim(),p=document.getElementById('zs-prompt').value.trim();if(!c)return;setBusy(true,`Zero-shot registering ${c}...`);try{const d=await api('/api/register_zero_shot',{class_name:c,text_prompt:p||c});updateUI(d);log(`Zero-shot registered "${c}" in ${d.registration_ms.toFixed(1)} ms.`);}catch(e){log('ERROR: '+e.message);}setBusy(false,'Ready');}
async function runPerceive(){setBusy(true,'Running SegmentGDA+ & 3D GMDA...');try{const d=await api('/api/perceive',{});setView('annotated');updateUI(d);log(`Perceived ${d.detections.length} objects | Relations: ${d.relations_summary||'none'}`);}catch(e){log('ERROR: '+e.message);}setBusy(false,'Ready');}
async function runExecute(){const c=document.getElementById('inp-cmd').value.trim();if(!c)return;setBusy(true,'Executing closed-loop grasp...');log(`>>> Instruction: "${c}"`);try{const d=await api('/api/execute',{instruction:c,allow_nbv:document.getElementById('chk-nbv').checked,force_nbv:document.getElementById('chk-force-nbv').checked});updateUI(d);log(`Result [${d.success?'SUCCESS':'FAILED'}] target=${d.target_class_name||'none'} | views=${d.num_views_used} | ${d.message}`);if(d.grounding_reasoning)log(`Grounding: ${d.grounding_reasoning}`);}catch(e){log('ERROR: '+e.message);}setBusy(false,'Ready');}
window.addEventListener('DOMContentLoaded',init);
</script>
</body>
</html>
"""
