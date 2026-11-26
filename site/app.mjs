import {CHANNELS,S,R,extract,parseCSV,predict,profileDefinitions,validate} from './inference.mjs?v=0.1.0-1';
const $=s=>document.querySelector(s),profiles=profileDefinitions();
const labels={rms_single:'Bus 7 · RMS',rms_receiving:'Bus 8 · RMS',rms_double:'Both buses · RMS',phasor_double:'Both buses · phasors'};
const eventLabels={internal:'Internal line fault',external:'External disturbance',healthy:'Healthy operation'};
const names={AG:'A–ground fault',BG:'B–ground fault',CG:'C–ground fault',AB:'A–B fault',BC:'B–C fault',CA:'C–A fault',ABG:'A–B–ground fault',BCG:'B–C–ground fault',CAG:'C–A–ground fault',ABC:'Three-phase fault'};
let examples=[],evaluation=null,sequence=0,currentResult=null,currentReference=null;
const bundles={};
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=(v,d=1)=>v==null?'—':`${(v*100).toFixed(d)}%`;
const num=(v,d=2)=>v==null?'—':Number(v).toFixed(d);
const template=()=>({schema_version:1,benchmark_id:'ieee9-adapted-7-8-v1',line_id:'L7-8',bus_s:7,bus_r:8,line_length_km:120,base_kv:345,base_mva:100,frequency_hz:50,units:'per_unit_rms',synchronized:true,rms:{},phasors:{}});

function initTable(){
  $('#measurement-rows').innerHTML=['Va','Vb','Vc','Ia','Ib','Ic'].map((name,i)=>`<tr><td>${name.slice(0,1)}<sub>${name.slice(1)}</sub><span class="unit-label">${i<3?'V':'I'}</span></td>${[S[i],R[i]].map(k=>`<td><input type="number" step="any" min="0" id="${k}-mag" aria-label="${k} RMS per unit" placeholder="—"></td><td class="angle-col"><input class="angle-input" type="number" step="any" id="${k}-angle" aria-label="${k} phase degrees" placeholder="—"></td>`).join('')}</tr>`).join('');
}

function setRecord(record,reference=null){
  currentReference=reference;
  const hasPhase=Object.keys(record.phasors??{}).length>0;
  $('#representation').value=hasPhase?'phasor':'rms';
  const included=[...Object.keys(record.phasors??{}),...Object.keys(record.rms??{})];
  const hasS=S.some(k=>included.includes(k)),hasR=R.some(k=>included.includes(k));
  $('#terminal').value=hasS&&hasR?'both':hasR?'receiving':'sending';
  $('#synchronized').checked=record.synchronized===true;
  for(const k of CHANNELS){
    const z=record.phasors?.[k];
    $(`#${k}-mag`).value=z?Math.hypot(z.re,z.im).toPrecision(6):record.rms?.[k]??'';
    $(`#${k}-angle`).value=z?(Math.atan2(z.im,z.re)*180/Math.PI).toFixed(3):'';
  }
  updateAvailability();
}

function getRecord(){
  const record=template();record.synchronized=$('#synchronized').checked;
  const terminal=$('#terminal').value,withPhase=$('#representation').value==='phasor';
  for(const k of CHANNELS){
    if((terminal==='sending'&&k.startsWith('r_'))||(terminal==='receiving'&&k.startsWith('s_')))continue;
    const magRaw=$(`#${k}-mag`).value,angleRaw=$(`#${k}-angle`).value;
    if(magRaw==='')continue;
    const mag=Number(magRaw);
    if(!Number.isFinite(mag)||mag<0)throw Error(`${k}: enter a finite non-negative RMS magnitude`);
    if(withPhase&&angleRaw!==''){
      const angle=Number(angleRaw)*Math.PI/180;
      if(!Number.isFinite(angle))throw Error(`${k}: invalid phase angle`);
      record.phasors[k]={re:mag*Math.cos(angle),im:mag*Math.sin(angle)};
    }else record.rms[k]=mag;
  }
  return record;
}

function updateAvailability(){
  const terminal=$('#terminal').value,phase=$('#representation').value==='phasor';
  for(const k of CHANNELS){
    const disabled=(terminal==='sending'&&k.startsWith('r_'))||(terminal==='receiving'&&k.startsWith('s_'));
    $(`#${k}-mag`).disabled=disabled;$(`#${k}-angle`).disabled=disabled;
  }
  document.querySelectorAll('.angle-col').forEach(e=>e.hidden=!phase);
  $('#sending-heading').colSpan=phase?2:1;$('#receiving-heading').colSpan=phase?2:1;
  $('#synchronized').disabled=!phase||terminal!=='both';
  $('#input-error').hidden=true;
  try{
    const record=getRecord(),info=validate(record,profiles);
    const available=info.eligible;
    $('#capability-note').textContent=available.length?`${available.length} trained profile${available.length>1?'s':''} available · ${labels[available.at(-1)]}. ${!record.synchronized?'Phasor location requires confirmed time alignment.':''}`:'A trained profile needs all six V/I channels from at least one terminal. Missing values remain missing.';
    document.querySelectorAll('[name="task"]').forEach(e=>e.disabled=!available.length);
    const chosen=$('#profile').value==='auto'?available.at(-1):$('#profile').value;
    $('#family option[value="anfis"]').disabled=chosen!=='rms_single';
    if(chosen!=='rms_single'&&$('#family').value==='anfis')$('#family').value='forest';
    renderVoltage(info.features);
  }catch(e){$('#capability-note').textContent=e.message;}
}

function invalidate(){
  sequence++;currentResult=null;
  $('#result-status').textContent='Inputs changed';$('#result-status').className='chip neutral';
  $('#result-body').innerHTML='<div class="result-empty"><div class="empty-orbit">ϟ</div><h3>Ready for a new analysis.</h3><p>Run the model for the current measurements.</p></div>';
  updateAvailability();
}

function renderVoltage(f){
  const W=390,H=150,base=118,max=1.25;
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Sending and receiving phase RMS voltages">`;
  for(const value of [0,.5,1]){const y=base-value/max*92;svg+=`<line x1="30" y1="${y}" x2="380" y2="${y}" stroke="#eaf0e7" stroke-dasharray="3 4"/><text x="18" y="${y+3}" font-size="8" fill="#9ba998" text-anchor="end">${value.toFixed(1)}</text>`;}
  ['a','b','c'].forEach((phase,i)=>{
    const x=83+i*115;
    for(const [j,end] of ['s','r'].entries()){
      const value=f[`${end}_v${phase}`];
      if(value==null)continue;
      const h=Math.max(0,Math.min(value,max))/max*92;
      svg+=`<rect x="${x+j*25}" y="${base-h}" width="18" height="${h}" rx="3" fill="${j?'#bed8c3':'#329a83'}"/><text x="${x+j*25+9}" y="${base-h-5}" font-size="8" fill="#8a9c85" text-anchor="middle">${value.toFixed(2)}</text>`;
    }
    svg+=`<text x="${x+21}" y="139" font-size="9" fill="#85977e" text-anchor="middle">PHASE ${phase.toUpperCase()}</text>`;
  });$('#voltage-chart').innerHTML=svg+'</svg>';
}

async function loadBundle(profile){
  if(!bundles[profile]){
    const response=await fetch(`assets/model-${profile}.json`);
    if(!response.ok)throw Error('Could not load the trained model. Please retry.');
    bundles[profile]=await response.json();
  }
  return bundles[profile];
}

async function analyze(){
  const request=++sequence,button=$('#analyze');
  button.disabled=true;button.firstElementChild.textContent='Running trained model…';$('#input-error').hidden=true;
  try{
    const tasks=[...document.querySelectorAll('[name="task"]:checked')].map(e=>e.value);
    if(!tasks.length)throw Error('Select at least one output.');
    const record=getRecord(),info=validate(record,profiles);
    const profile=$('#profile').value==='auto'?info.eligible.at(-1):$('#profile').value;
    if(!profile)throw Error('No trained profile matches these measurements. Supply all six V/I channels at one terminal.');
    if(!info.eligible.includes(profile))throw Error('The requested profile is unavailable. Check missing channels and synchronization.');
    const bundle=await loadBundle(profile);
    if(request!==sequence)return;
    const start=performance.now();
    currentResult=predict(record,bundle,profiles,$('#family').value);
    renderResult(currentResult,tasks,performance.now()-start);
  }catch(e){
    if(request===sequence){$('#input-error').textContent=e.message;$('#input-error').hidden=false;$('#result-status').textContent='Check input';$('#result-status').className='chip warn';$('#result-body').innerHTML='<div class="result-empty"><h3>Input needs attention.</h3><p>The model has withheld a result.</p></div>';}
  }finally{button.disabled=false;button.firstElementChild.textContent='Analyze measurements';}
}

function renderResult(r,tasks,elapsed){
  const status=$('#result-status');status.textContent=r.quality==='accepted'?'Analysis complete':r.quality==='uncertain'?'Uncertain':'Outside domain';status.className='chip'+(r.quality==='accepted'?'':' warn');
  const event=r.event?eventLabels[r.event]:'Diagnosis withheld';
  const main=r.event==='internal'?(tasks.includes('type')?r.fault_type??'Uncertain type':'Line fault'):r.event==='healthy'?'Healthy':r.event==='external'?'External event':'Review input';
  let html=`<div class="diagnosis-main"><div><h3 class="fault-label">${esc(main)}</h3><p>${esc(r.fault_type&&tasks.includes('type')?names[r.fault_type]:event)}</p></div><span class="fault-icon">${r.event==='healthy'?'✓':r.event==='internal'?'ϟ':'○'}</span></div>`;
  if(tasks.includes('location')){
    if(r.location_km!=null){
      const [lo,hi]=r.interval_km;
      html+=`<div class="location-block"><span class="micro">ESTIMATED DISTANCE FROM BUS 7</span><div class="distance">${num(r.location_km,1)} <small>km</small></div><div class="location-sub">${num(r.location_km/120*100,1)}% along the monitored line</div><div class="line-track"><span class="line-band" style="left:${lo/120*100}%;width:${(hi-lo)/120*100}%"></span><span class="line-point" style="left:${r.location_km/120*100}%"></span></div><div class="line-labels"><span>BUS 7 · 0 km</span><span>BUS 8 · 120 km</span></div><p class="interval-note">Calibration band: ${num(lo,1)}–${num(hi,1)} km · 90% target<br>${r.location_method==='physics'?'Two-ended electrical model':'Learned RMS distance estimate'} · empirical test coverage is available under Evaluation.</p></div>`;
    }else html+='<div class="location-block"><span class="micro">FAULT LOCATION</span><div class="distance" style="font-size:23px">Not applicable / withheld</div><p class="interval-note">A distance is reported only for a sufficiently supported internal-line fault.</p></div>';
  }
  if(tasks.includes('type')&&r.type_probabilities){
    html+='<div class="probability-title"><span>FAULT TYPE · CALIBRATED MODEL SCORES</span><span>TOP 4</span></div>';
    html+=Object.entries(r.type_probabilities).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([c,p])=>`<div class="prob-row"><span>${esc(c)}</span><div class="prob-track"><div class="prob-fill" style="width:${p*100}%"></div></div><span>${percent(p,0)}</span></div>`).join('');
  }
  if(tasks.includes('event')&&r.event_probabilities)html+=`<div class="probability-title"><span>EVENT MODEL SCORE</span><span>${percent(Math.max(...Object.values(r.event_probabilities)),1)}</span></div>`;
  html+=r.messages.map(m=>`<p class="result-message">${esc(m)}</p>`).join('');
  if(currentReference){
    const t=currentReference;
    const referenceText=`True event: ${eventLabels[t.event]}. ${t.fault_type?'True type: '+t.fault_type+'. ':''}${t.location_km==null?'No monitored-line fault position.':'True distance: '+num(t.location_km,2)+' km from bus 7.'}`;
    const error=t.location_km!=null&&r.location_km!=null?`Absolute error for this example: ${num(Math.abs(t.location_km-r.location_km),2)} km.`:'';
    html+=`<details class="advanced reference"><summary>Show simulation reference</summary><p class="small muted">${esc(referenceText)} ${esc(error)} The reference is used only for this comparison; it is not a prediction input.</p></details>`;
  }
  html+=`<div class="result-meta"><span>${esc(labels[r.profile])} · ${r.family==='anfis'?'100-rule ANFIS':'Random Forest'}</span><span>${elapsed.toFixed(1)} ms inference*</span></div><p class="privacy-note">*Excludes measurement acquisition and model download. Synthetic benchmark.</p>`;
  $('#result-body').innerHTML=html;
}

function download(name,text,type='application/json'){
  const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
}

function navigate(){
  const tab=['diagnose','evaluation','method'].includes(location.hash.slice(1))?location.hash.slice(1):'diagnose';
  document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!==`view-${tab}`);
  document.querySelectorAll('[data-tab]').forEach(e=>{e.classList.toggle('active',e.dataset.tab===tab);if(e.dataset.tab===tab)e.setAttribute('aria-current','page');else e.removeAttribute('aria-current');});
}

function renderEvaluation(e){
  const best=e.profiles.phasor_double.forest,manifest=e.dataset;
  const cards=[['Measurement records',manifest.records.toLocaleString(),`${manifest.physical_scenarios.toLocaleString()} physical scenarios · ${manifest.groups} independent groups`],['Held-out records',manifest.partition_records.test.toLocaleString(),'Separate test groups · held-out distance bands'],['Two-ended location MAE',`${num(best.location_accepted_internal.mae_km)} km`,'Accepted internal test faults · all SNR levels'],['Location availability',percent(best.pipeline.location_availability_on_internal),'Internal test records receiving a distance']];
  let html='<div class="metric-grid">'+cards.map(([label,value,note])=>`<div class="metric-card"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('')+'</div>';
  html+='<section class="panel evaluation-panel"><h2>Same test set. Different measurements.</h2><p>Event macro-F1 uses all test records before abstention. Type macro-F1 assumes a true internal fault. Location errors use accepted internal cases; the availability column reports how often a distance was produced.</p><div class="table-scroll"><table class="eval-table"><thead><tr><th>Input / model</th><th>Event F1</th><th>Type F1</th><th>MAE</th><th>RMSE</th><th>Location availability</th><th>Band coverage</th></tr></thead><tbody>';
  for(const [profile,families] of Object.entries(e.profiles))for(const [family,m] of Object.entries(families)){
    html+=`<tr><td>${labels[profile]}<small>${family==='anfis'?'ANFIS reference':'Random Forest'+(profile==='phasor_double'?' + physics':'')}</small></td><td>${percent(m.event.macro_f1)}</td><td>${percent(m.fault_type_oracle_internal.macro_f1)}</td><td>${num(m.location_accepted_internal.mae_km)} km</td><td>${num(m.location_accepted_internal.rmse_km)} km</td><td>${percent(m.pipeline.location_availability_on_internal)}</td><td>${percent(m.pipeline.interval_empirical_test_coverage)}</td></tr>`;
  }
  html+='</tbody></table></div><p>Calibration bands target 90% coverage. Measured coverage can differ on held-out distance bands and other shifts. These results describe this synthetic benchmark and do not establish field performance.</p></section>';
  html+='<div class="chart-grid"><section class="panel"><h2>Fault-type confusion matrix</h2><img src="assets/figures/confusion.png" alt="Held-out fault classification confusion matrix for the synchronized two-ended profile" loading="lazy"><p>Actual class on rows; predicted class on columns. Counts use all true internal test cases before the final eligibility gates.</p></section><section class="panel"><h2>Location error under measurement noise</h2><img src="assets/figures/noise.png" alt="Measured mean location error by complex-phasor noise level" loading="lazy"><p>SNR is defined for complex phasor perturbations. It is not raw-waveform SNR. Curves use accepted internal faults and show all profile differences.</p></section></div>';
  html+=`<section class="panel evaluation-panel" style="margin-top:20px"><h2>Error management is part of the result</h2><p>The phasor pipeline withheld an event diagnosis on <strong>${best.pipeline.abstention_or_domain_rejection_count}</strong> of ${best.pipeline.total_records} test records. It assigned an in-line location to <strong>${best.pipeline.false_in_line_location_count}</strong> of ${best.pipeline.non_internal_records} healthy/external records. The event-confidence threshold is 0.65; type threshold 0.55. Out-of-domain inputs and inconsistent physical positions are rejected.</p><div class="resource-links"><a href="assets/evaluation.json" download>Download full evaluation JSON ↓</a><a href="https://github.com/Snow-Warrior07/GridFault-Lab/tree/main/results">Standalone figures and results ↗</a></div></section>`;
  $('#evaluation-content').innerHTML=html;
}

function renderMethod(e){
  const d=e.dataset,steps=[
    ['Define the electrical problem',`<p>The benchmark is a <b>three-phase nodal circuit on an adapted IEEE 9-bus topology</b>. The monitored line is bus 7 to bus 8, assigned a length of 120 km. Positive-sequence branch data originate from MATPOWER case9; source impedances, zero-sequence impedances, grounding, constant-admittance loads, and the length are documented research assumptions.</p><p>The simulator solves <code>Y · V = I</code> after inserting a physical fault connection. It generates fundamental-frequency steady-state phasors. This version makes no claim of modelling electromagnetic traveling waves, CT saturation, breaker transients, or field measurements.</p>`],
    ['Generate labelled data',`<p><b>${d.groups} independent operating-condition groups</b> produce ${d.physical_scenarios.toLocaleString()} physical scenarios. Each group contains ten internal fault types, one healthy case, and two external faults. Clean and noisy observations give <b>${d.records.toLocaleString()} measurement records</b>.</p><p>Loading varies from 0.65–1.35 of nominal, source strength from 0.75–1.25, voltage scale from 0.97–1.05, and line R/X by ±5%. Fault resistance spans 0.5–80 Ω; distance spans 5–95% of line length. Gaussian complex-phasor noise uses 20, 30, 40, or 50 dB SNR. AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, and ABC are covered.</p>`],
    ['Preprocess without inventing measurements',`<p>All measurements use consistent <b>per-unit RMS bases: 345 kV line-to-line and 100 MVA</b>. Phase-voltage base is 345/√3 kV; phase-current base is 100 MVA/(√3 × 345 kV). RMS magnitudes are calculated from complex phasors. Phases are preserved for sequence and current-differential features.</p><p>Four separately trained profiles use six sending-end RMS values, six receiving-end values, twelve two-ended values, or 52 phasor/sequence/physics features. ANFIS uses training-only mean/standard-deviation scaling. Random Forest needs no standardization. Missing terminal data selects a supported profile; absent values are never replaced with zero.</p>`],
    ['Train the model hierarchy',`<p><b>Random Forest</b> provides three-way event recognition (healthy/internal/external), ten-way fault classification, and RMS-based distance regression. Each forest has 40 trees, maximum depth 13, and minimum leaf size 3.</p><p>The paper-aligned reference is <b>100-rule Gaussian Sugeno ANFIS</b> for sending-end RMS inputs: k-means initializes premise centers; ridge least squares fits linear consequents; gradient updates refine centers and Gaussian widths. Detector, classifier, and locator are separate heads. It is an adapted implementation, not a MATLAB numerical replica of the paper.</p><p>With synchronized two-ended phasors, distance comes from a <b>least-squares electrical locator</b>, while Random Forest handles event/type recognition. Models are exported as JSON and executed in Python or the browser with numerical parity tests.</p>`],
    ['Keep training, calibration, and testing separate',`<p>Groups are assigned before creating events or noise variants: <b>60% training, 15% validation, 10% calibration, 15% testing</b>. Every related observation stays in its group. Fault-distance bands 20–25% and 70–75% are excluded from development partitions and explicitly represented in testing.</p><p>Training fits model parameters and scaling. Validation fits probability temperatures. Calibration estimates location-error bands. The test set supplies the reported results. Fault labels, true positions, true fault resistance, and noise settings are not prediction inputs.</p>`],
    ['Measure and manage errors',`<p>Classification uses macro-F1, per-class recall, confusion matrices, and calibration error. Distance uses MAE, RMSE, 95th-percentile absolute error, and error as a percentage of total line length.</p><div class="formula">MAE = mean(|predicted km − true km|)<br>RMSE = √mean((predicted km − true km)²)</div><p>Event scores below 0.65 cause abstention; type scores below 0.55 withhold the class. A training-support screen rejects out-of-domain inputs. The phasor locator checks electrical residuals and refuses positions outside the line. Healthy/external events have no in-line fault distance. Prediction bands use a finite-sample calibration-error quantile with a 90% target; actual test coverage is reported.</p><p>Noise robustness is evaluated under known injected noise. Without a clean reference, the app does not claim to recover exact true noise or SNR. Its input checks cover numeric validity, units, channel availability, declared synchronization, and training support.</p>`],
    ['Use the result within its tested scope',`<p>This version is a <b>research prototype for one registered circuit</b>. The interface reveals supported tasks, runs real learned models, and reports uncertainty. Receiving-end-only RMS location can be particularly coarse; compare the measured errors under Evaluation before interpreting a point estimate.</p><p>The next major validation step is independent Simulink/EMT or measured relay data. Extensions include transient windows, realistic CT/CVT models, broader topology generalization, and simulator-based control decisions. A clean numerical circuit test does not establish real-world protective-relay reliability.</p>`],
  ];
  $('#method-content').innerHTML=steps.map(([title,body],i)=>`<section class="walkthrough"><span class="step-number">${String(i+1).padStart(2,'0')}</span><div class="panel"><h2>${title}</h2>${body}</div></section>`).join('')+`<div class="resource-links"><a href="https://github.com/Snow-Warrior07/GridFault-Lab/blob/main/docs/PROJECT_GUIDE.md">Thorough project explanation ↗</a><a href="https://github.com/Snow-Warrior07/GridFault-Lab/blob/main/docs/DATA_CARD.md">Data and assumptions ↗</a><a href="https://doi.org/10.1016/j.epsr.2022.108031">Distribution review ↗</a><a href="https://ieeexplore.ieee.org/document/10305233">AI review ↗</a><a href="https://doi.org/10.1109/ACCESS.2021.3129838">Fault-location review ↗</a></div>`;
}

async function init(){
  initTable();navigate();window.addEventListener('hashchange',navigate);
  for(const id of ['representation','terminal','synchronized','profile','family'])$(`#${id}`).addEventListener('change',invalidate);
  $('#measurement-rows').addEventListener('input',()=>{currentReference=null;$('#example-select').value='';invalidate();});
  document.querySelectorAll('[name="task"]').forEach(e=>e.addEventListener('change',invalidate));
  $('#analyze').addEventListener('click',analyze);
  $('#example-select').addEventListener('change',()=>{const e=examples[Number($('#example-select').value)];if(e){setRecord(e.record,e.truth);invalidate();}});
  $('#download-input').addEventListener('click',()=>{try{download('gridfault-input.json',JSON.stringify(getRecord(),null,2));}catch(e){alert(e.message);}});
  $('#download-csv').addEventListener('click',()=>{
    const r=getRecord(),keys=['synchronized'],values=[String(r.synchronized)];
    for(const k of CHANNELS){if(r.phasors[k]){keys.push(k+'_re',k+'_im');values.push(r.phasors[k].re,r.phasors[k].im);}else if(k in r.rms){keys.push(k);values.push(r.rms[k]);}}
    download('gridfault-measurement.csv',keys.join(',')+'\n'+values.join(',')+'\n','text/csv');
  });
  $('#upload').addEventListener('change',async event=>{
    try{
      const file=event.target.files[0];if(!file)return;if(file.size>1000000)throw Error('Input file is too large; the limit is 1 MB.');
      const text=await file.text(),record=file.name.toLowerCase().endsWith('.csv')?parseCSV(text):JSON.parse(text);
      validate(record,profiles);setRecord(record);invalidate();$('#example-select').value='';
    }catch(e){$('#input-error').textContent=e.message;$('#input-error').hidden=false;}finally{event.target.value='';}
  });
  try{
    const responses=await Promise.all([fetch('assets/examples.json'),fetch('assets/evaluation.json')]);
    if(responses.some(r=>!r.ok))throw Error('Dataset assets could not be loaded.');
    [examples,evaluation]=await Promise.all(responses.map(r=>r.json()));
    $('#example-select').innerHTML=examples.map((e,i)=>`<option value="${i}">${esc(e.truth.event==='internal'?names[e.truth.fault_type]:eventLabels[e.truth.event])}</option>`).join('');
    const initial=Math.max(0,examples.findIndex(e=>e.truth.fault_type==='AG'));$('#example-select').value=String(initial);
    setRecord(examples[initial].record,examples[initial].truth);renderEvaluation(evaluation);renderMethod(evaluation);await analyze();
  }catch(e){$('#input-error').textContent=e.message;$('#input-error').hidden=false;$('#analyze').disabled=true;}
}
init();

