"""Generate scientific plots, human-readable metrics, and independent MATLAB cases."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gridfault.config import FAULT_TYPES
from gridfault.simulator import simulate,to_record

ROOT=Path(__file__).resolve().parents[1]
e=json.loads((ROOT/'results/evaluation.json').read_text())
figures=ROOT/'results/figures';figures.mkdir(parents=True,exist_ok=True)
site=ROOT/'site/assets/figures';site.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#304c43','text.color':'#304c43','xtick.color':'#758574','ytick.color':'#758574','axes.edgecolor':'#dce5d8','figure.facecolor':'white','axes.facecolor':'white'})

def save(fig,name):
    fig.savefig(figures/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(figures/(name+'.svg'),bbox_inches='tight')
    fig.savefig(site/(name+'.png'),dpi=150,bbox_inches='tight')
    plt.close(fig)

m=e['profiles']['phasor_double']['forest']['fault_type_oracle_internal']
cm=np.array(m['confusion_matrix'])
fig,ax=plt.subplots(figsize=(5.4,4.8))
ax.imshow(cm,cmap='BuGn',vmin=0,vmax=cm.max())
ax.set_xticks(range(10),m['classes'],rotation=45,ha='right');ax.set_yticks(range(10),m['classes'])
ax.set_xlabel('Predicted fault type');ax.set_ylabel('True fault type')
for i in range(10):
    for j in range(10):
        if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>.5*cm.max() else '#285c49')
ax.set_title('Synchronized phasors: held-out internal faults',pad=18,fontsize=11)
save(fig,'confusion')

fig,ax=plt.subplots(figsize=(6.0,4.4))
labels={'rms_single':'Sending-end RMS · forest','rms_receiving':'Receiving-end RMS · forest','rms_double':'Both-end RMS · forest','phasor_double':'Both-end phasors + physics'}
colors=['#8eb3a2','#c79e67','#458b70','#134e43']
for (profile,label),color in zip(labels.items(),colors):
    buckets=e['profiles'][profile]['forest']['noise_location']
    keys=[k for k in ['20.0','30.0','40.0','50.0','clean'] if k in buckets]
    ax.plot(range(len(keys)),[buckets[k]['mae_km'] for k in keys],marker='o',markersize=4,label=label,color=color,lw=1.6)
ax.set_xticks(range(5),['20 dB','30 dB','40 dB','50 dB','Clean'])
ax.set_ylabel('Mean absolute location error (km)');ax.set_xlabel('Complex-phasor measurement SNR')
ax.grid(axis='y',color='#edf1e9');ax.legend(frameon=False,fontsize=8,loc='upper right')
ax.set_title('Accepted internal test faults; availability reported separately',fontsize=10,pad=16)
save(fig,'noise')

fig,ax=plt.subplots(figsize=(6.4,3.9))
names=[];mae=[];rmse=[]
for profile,families in e['profiles'].items():
    for family,metrics in families.items():
        names.append(profile.replace('rms_','RMS ').replace('phasor_double','Phasors')+'\n'+family)
        mae.append(metrics['location_accepted_internal']['mae_km']);rmse.append(metrics['location_accepted_internal']['rmse_km'])
x=np.arange(len(names));ax.bar(x-.18,mae,.34,color='#2f8a70',label='MAE');ax.bar(x+.18,rmse,.34,color='#bcd6bc',label='RMSE')
ax.set_xticks(x,names,fontsize=8);ax.set_ylabel('Location error (km)');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
ax.set_axisbelow(True);save(fig,'model_comparison')

header='| Profile | Family | Event macro-F1 | Type macro-F1 (true internal) | MAE km | RMSE km | Internal location availability | Interval test coverage | False non-internal locations |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
lines=[]
for p,fs in e['profiles'].items():
    for family,m in fs.items():
        q=m['pipeline'];l=m['location_accepted_internal']
        lines.append(f"| {p} | {family} | {m['event']['macro_f1']:.4f} | {m['fault_type_oracle_internal']['macro_f1']:.4f} | {l['mae_km']:.3f} | {l['rmse_km']:.3f} | {q['location_availability_on_internal']:.2%} | {q['interval_empirical_test_coverage']:.2%} | {q['false_in_line_location_count']} / {q['non_internal_records']} |")
(ROOT/'results/SUMMARY.md').write_text('# Held-out results\n\n'+header+'\n'.join(lines)+'\n\nSynthetic fundamental-frequency benchmark only. Location errors are conditional on accepted internal faults; availability and false locations are shown to prevent selective reporting. Type scores use oracle internal-event membership. No field/EMT performance claim. Calibration intervals target 90%; reported coverage is empirical.\n',encoding='utf-8')

cases=[]
for fault in FAULT_TYPES:
    for d in [.05,.25,.5,.75,.95]:
        s=simulate(fault_type=fault,location_pu=d,resistance_ohm=15)
        cases.append({'fault_type':fault,'truth_pu':d,'record':to_record(s['phasors']),'z_re':s['actual_z'].real.tolist(),'z_im':s['actual_z'].imag.tolist()})
(ROOT/'tests/fixtures/matlab_cases.json').write_text(json.dumps(cases),encoding='utf-8')
print('Generated standalone PNG/SVG figures, results/SUMMARY.md, and 50 independent MATLAB equation-check cases.')

