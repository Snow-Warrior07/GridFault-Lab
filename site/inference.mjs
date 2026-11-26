/* Portable inference. Models are learned in Python; this executes the same JSON in-browser. */
export const S=['s_va','s_vb','s_vc','s_ia','s_ib','s_ic'];
export const R=['r_va','r_vb','r_vc','r_ia','r_ib','r_ic'];
export const CHANNELS=[...S,...R];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1]];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const mul=(a,b)=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const abs=z=>Math.hypot(...z);
const norm=z=>Math.sqrt(z.reduce((s,a)=>s+a[0]**2+a[1]**2,0));
const matmul=(m,v)=>m.map(row=>row.reduce((sum,z,i)=>add(sum,mul(z,v[i])),[0,0]));

export function extract(phasors){
  const f={};
  for(const [k,z] of Object.entries(phasors)){f[k]=abs(z);f[k+'_re']=z[0];f[k+'_im']=z[1];}
  if(CHANNELS.every(k=>k in phasors)){
    const a=[-.5,Math.sqrt(3)/2],a2=[-.5,-Math.sqrt(3)/2];
    const groups=[];
    for(let i=0;i<12;i+=3){
      const [x,y,z]=CHANNELS.slice(i,i+3).map(k=>phasors[k]);groups.push([x,y,z]);
      const seq=[add(add(x,y),z),add(add(x,mul(a,y)),mul(a2,z)),add(add(x,mul(a2,y)),mul(a,z))];
      const name=['sv','si','rv','ri'][i/3];
      seq.forEach((v,j)=>f[`${name}_seq${j}`]=abs(v)/3);
    }
    const diff=groups[1].map((v,i)=>add(v,groups[3][i]));
    diff.forEach((v,i)=>f[['diff_a','diff_b','diff_c'][i]]=abs(v));
    f.diff_ratio=norm(diff)/(norm(groups[1])+norm(groups[3])+1e-12);
  }
  return f;
}

export function validate(record,profiles){
  if(!record||typeof record!=='object'||Array.isArray(record))throw Error('Input must be a JSON object');
  const required={schema_version:1,benchmark_id:'ieee9-adapted-7-8-v1',line_id:'L7-8',bus_s:7,bus_r:8,base_kv:345,base_mva:100,frequency_hz:50,line_length_km:120,units:'per_unit_rms'};
  for(const [k,v] of Object.entries(required))if(record[k]!==v)throw Error(`Unsupported or missing ${k}; this model requires ${v}`);
  if(typeof record.synchronized!=='boolean')throw Error('synchronized must explicitly be true or false');
  const supplied=record.phasors===undefined?{}:record.phasors,given=record.rms===undefined?{}:record.rms,phasors={},rms={};
  if(typeof supplied!=='object'||supplied===null||Array.isArray(supplied)||typeof given!=='object'||given===null||Array.isArray(given))throw Error('rms and phasors must be objects');
  for(const [k,z] of Object.entries(supplied)){
    if(!CHANNELS.includes(k))throw Error(`Unknown measurement channel: ${k}`);
    if(!z||typeof z!=='object'||Object.keys(z).sort().join(',')!=='im,re'||![z.re,z.im].every(v=>typeof v==='number'&&Number.isFinite(v)))throw Error(`Invalid phasor ${k}; finite numeric re and im required`);
    phasors[k]=[z.re,z.im];rms[k]=abs(phasors[k]);
  }
  for(const [k,v] of Object.entries(given)){
    if(!CHANNELS.includes(k))throw Error(`Unknown measurement channel: ${k}`);
    if(typeof v!=='number'||!Number.isFinite(v)||v<0)throw Error(`${k} must be a finite non-negative RMS magnitude`);
    if(k in rms&&Math.abs(v-rms[k])>Math.max(.001,.02*rms[k]))throw Error(`RMS and phasor disagree for ${k}`);
    rms[k]=v;
  }
  const features={...extract(phasors),...rms};
  const eligible=Object.entries(profiles).filter(([p,cols])=>cols.every(k=>k in features)&&(p!=='phasor_double'||record.synchronized)).map(([p])=>p);
  return {features,phasors,eligible};
}

export function rawPredict(model,x){
  if(model.kind==='forest'){
    let acc=null;
    for(const tree of model.trees){
      let n=0;
      while(tree[n][0]!==-1){const node=tree[n];n=Math.fround(x[node[0]])<=node[1]?node[2]:node[3];}
      const value=Array.isArray(tree[n][4])?tree[n][4]:[tree[n][4]];
      if(acc===null)acc=Array(value.length).fill(0);
      value.forEach((v,i)=>acc[i]+=v/model.trees.length);
    }
    return acc;
  }
  const xn=x.map((v,i)=>(v-model.mean[i])/model.scale[i]);
  const logw=model.centers.map((c,j)=>-.5*c.reduce((s,v,i)=>s+((xn[i]-v)/model.widths[j][i])**2,0));
  const max=Math.max(...logw),w=logw.map(v=>Math.exp(v-max)),total=w.reduce((s,v)=>s+v,0);
  const extended=[1,...xn],outputs=Array(model.coefficients[0][0].length).fill(0);
  for(let k=0;k<w.length;k++)for(let j=0;j<extended.length;j++)for(let o=0;o<outputs.length;o++)outputs[o]+=w[k]/total*extended[j]*model.coefficients[k][j][o];
  return outputs;
}

export function probabilities(model,x){
  let scores=rawPredict(model,x);
  if(model.kind==='forest')scores=scores.map(v=>Math.log(Math.max(v,1e-12)));
  scores=scores.map(v=>v/(model.temperature??1));
  const max=Math.max(...scores),e=scores.map(v=>Math.exp(v-max)),s=e.reduce((a,b)=>a+b,0);
  return e.map(v=>v/s);
}

export function physicsLocation(p,metadata){
  const vs=S.slice(0,3).map(k=>p[k]),is=S.slice(3).map(k=>p[k]);
  const vr=R.slice(0,3).map(k=>p[k]),ir=R.slice(3).map(k=>p[k]);
  const z=metadata.nominal_z_re.map((row,i)=>row.map((v,j)=>[v,metadata.nominal_z_im[i][j]]));
  const a=matmul(z,is.map((v,i)=>add(v,ir[i])));
  const zir=matmul(z,ir),b=vs.map((v,i)=>add(sub(v,vr[i]),zir[i]));
  const den=norm(a)**2;
  if(den<1e-12)return {location_pu:null,residual:null};
  const m=a.reduce((sum,v,i)=>sum+v[0]*b[i][0]+v[1]*b[i][1],0)/den;
  const res=b.map((v,i)=>sub(v,[m*a[i][0],m*a[i][1]]));
  return {location_pu:m,residual:norm(res)/(norm(b)+1e-12)};
}

export function predict(record,bundle,profiles,family='forest'){
  const {features:f,phasors,eligible}=validate(record,profiles);
  if(!eligible.includes(bundle.profile))throw Error(`Requested profile ${bundle.profile} is unavailable; check channels and synchronization`);
  const model=bundle.families[family];
  if(!model)throw Error(`${family} is not trained for ${bundle.profile}`);
  const x=bundle.features.map(k=>f[k]);
  const outside=bundle.features.filter((k,i)=>x[i]<bundle.bounds[i][0]||x[i]>bundle.bounds[i][1]);
  const r={profile:bundle.profile,family,eligible_profiles:eligible,event:null,fault_type:null,location_km:null,interval_km:null,quality:'accepted',messages:[],model_version:bundle.version};
  if(outside.length)return {...r,quality:'out_of_domain',messages:['Outside training support: '+outside.slice(0,6).join(', ')]};
  const ep=probabilities(model.event,x),ei=ep.indexOf(Math.max(...ep));
  r.event_probabilities=Object.fromEntries(model.event.classes.map((c,i)=>[c,ep[i]]));
  if(ep[ei]<.65)return {...r,quality:'uncertain',messages:['Event confidence below the validated decision threshold; diagnosis withheld.']};
  r.event=model.event.classes[ei];
  if(r.event!=='internal')return {...r,messages:['No in-line fault location applies to this event.']};
  const tp=probabilities(model.type,x),ti=tp.indexOf(Math.max(...tp));
  r.type_probabilities=Object.fromEntries(model.type.classes.map((c,i)=>[c,tp[i]]));
  if(tp[ti]>=.55)r.fault_type=model.type.classes[ti];else r.messages.push('Fault type is uncertain; type label withheld.');
  let m;
  if(model.location.kind==='physics'){
    const p=physicsLocation(phasors,bundle.benchmark);m=p.location_pu;r.physics_residual=p.residual;
    if(m===null||p.residual>model.location.max_residual){r.messages.push('Electrical consistency check failed; location withheld.');return r;}
  }else m=rawPredict(model.location,x)[0];
  if(!Number.isFinite(m)||m<0||m>1){r.messages.push('Predicted position is outside the monitored line; location withheld.');return r;}
  const d=m*120,q=model.location.interval_radius_km;
  r.location_km=d;r.interval_km=[Math.max(0,d-q),Math.min(120,d+q)];r.location_method=model.location.kind;
  if(!record.synchronized)r.messages.push('Phasor-based location is disabled; using the trained RMS fallback.');
  return r;
}

export function profileDefinitions(){
  return {rms_single:S,rms_receiving:R,rms_double:CHANNELS,phasor_double:[...CHANNELS,...CHANNELS.flatMap(k=>[k+'_re',k+'_im']),...['sv','si','rv','ri'].flatMap(g=>[0,1,2].map(i=>`${g}_seq${i}`)),'diff_a','diff_b','diff_c','diff_ratio']};
}

export function parseCSV(text){
  // Strict single-record numeric format; no spreadsheet formula evaluation.
  const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(v=>v.trim());
  if(lines.length!==2)throw Error('CSV must contain a header and exactly one measurement row. Use the downloadable template.');
  const keys=lines[0].split(',').map(v=>v.trim()),values=lines[1].split(',').map(v=>v.trim());
  if(keys.length!==values.length||new Set(keys).size!==keys.length)throw Error('Invalid or duplicate CSV columns');
  const record={schema_version:1,benchmark_id:'ieee9-adapted-7-8-v1',line_id:'L7-8',bus_s:7,bus_r:8,line_length_km:120,base_kv:345,base_mva:100,frequency_hz:50,units:'per_unit_rms',synchronized:false,rms:{},phasors:{}};
  const raw=Object.fromEntries(keys.map((k,i)=>[k,values[i]]));
  const allowed=new Set(['synchronized',...CHANNELS,...CHANNELS.flatMap(k=>[k+'_re',k+'_im'])]);
  for(const k of keys)if(!allowed.has(k))throw Error(`Unknown CSV column: ${k}`);
  for(const k of CHANNELS){
    if(raw[k]!==undefined&&raw[k]!==''){
      const v=Number(raw[k]);if(!Number.isFinite(v))throw Error(`Non-numeric ${k}`);record.rms[k]=v;
    }
    const re=raw[k+'_re'],im=raw[k+'_im'];
    if((re!==undefined&&re!=='')||(im!==undefined&&im!=='')){
      if(re===undefined||im===undefined||re===''||im==='')throw Error(`Both real and imaginary components are needed for ${k}`);
      const z={re:Number(re),im:Number(im)};
      if(!Number.isFinite(z.re)||!Number.isFinite(z.im))throw Error(`Non-numeric phasor ${k}`);
      record.phasors[k]=z;
    }
  }
  if(raw.synchronized!==undefined){
    if(!['true','false'].includes(raw.synchronized))throw Error('synchronized must be true or false');
    record.synchronized=raw.synchronized==='true';
  }
  return record;
}

