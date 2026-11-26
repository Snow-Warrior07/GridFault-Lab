import fs from 'node:fs';
import assert from 'node:assert/strict';
import {predict,profileDefinitions,parseCSV} from '../site/inference.mjs';

const fixtures=JSON.parse(fs.readFileSync(new URL('../tests/fixtures/parity.json',import.meta.url),'utf8'));
const profiles=profileDefinitions(),cache={};
let checked=0;
function compare(a,b,path=''){
  if(typeof a==='number'&&typeof b==='number')assert.ok(Math.abs(a-b)<1e-8,`${path}: ${a} != ${b}`);
  else if(Array.isArray(a))a.forEach((v,i)=>compare(v,b[i],`${path}[${i}]`));
  else if(a&&typeof a==='object')for(const [k,v] of Object.entries(a))compare(v,b[k],`${path}.${k}`);
  else assert.equal(a,b,path);
}
for(const f of fixtures){
  cache[f.profile]??=JSON.parse(fs.readFileSync(new URL(`../site/assets/model-${f.profile}.json`,import.meta.url),'utf8'));
  const got=predict(f.record,cache[f.profile],profiles,f.family);
  compare(f.expected,got);checked++;
}
assert.throws(()=>parseCSV('s_va,s_va\n1,2'),/duplicate/);
assert.throws(()=>parseCSV('s_va\n=1+1'),/Non-numeric/);
assert.throws(()=>parseCSV('s_va_re\n1'),/Both real/);
console.log(`PASS: ${checked} Python/browser prediction fixtures; strict CSV rejection checks.`);

