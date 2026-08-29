import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12,
 'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.6,
 'figure.dpi':200,'savefig.dpi':200,'savefig.bbox':'tight','legend.frameon':False})
NAVY='#1F3B63'; ACC='#C4622D'; TEAL='#2E7D74'; GREY='#6B7280'; PURP='#6B4E9B'; RED='#B3282D'
R=json.load(open('analysis/repro_results.json'))
T=json.load(open('analysis/tabular_results.json'))
A=json.load(open('analysis/alpha_errors.json'))
ev=pd.read_csv('analysis/val_rows_914.csv')
F='analysis/figs/'

# ── F1: dataset timeline & image coverage ────────────────────────────────────
df=pd.read_csv('combined_dataset.csv'); df['timestamp']=pd.to_datetime(df.timestamp)
df=df[df.timestamp<'2026-08-16'].sort_values('timestamp').reset_index(drop=True)
m=df.set_index('timestamp').resample('ME').agg(rows=('ghi','size'),
        imgs=('image_path',lambda s: s.notna().sum()))
fig,ax=plt.subplots(figsize=(10,3.2))
ax.bar(m.index,m.rows,width=22,color='#D8DEE8',label='Hourly rows in dataset')
ax.bar(m.index,m.imgs,width=22,color=NAVY,label='Rows with a matched satellite frame')
ax.set_ylim(0,395)
for x,lab,c in [(pd.Timestamp('2025-11-02'),'train | val',ACC),
                (pd.Timestamp('2026-03-25'),'val | test',RED)]:
    ax.axvline(x,color=c,ls='--',lw=1.6)
    ax.text(x,388,'  '+lab,color=c,fontsize=9,va='top')
ax.annotate('imagery stops 2026-06-07\n→ the test split keeps only 18 imaged rows',
            xy=(pd.Timestamp('2026-06-10'),30),xytext=(pd.Timestamp('2025-11-20'),200),
            fontsize=9,color=RED,arrowprops=dict(arrowstyle='->',color=RED,lw=1.2))
ax.set_ylabel('rows per month')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=2,fontsize=9)
ax.set_title('Dataset coverage: weather rows ran ahead of the satellite archive',loc='left')
plt.savefig(F+'f1_coverage.png'); plt.close()

# ── F2: main results grouped bars (n=914) ────────────────────────────────────
order=[('Smart persistence',None),('Linear Regression',None),('Random Forest',None),
       ('LightGBM',None),('Physics-Gated (small)','d'),('Physics-Gated (large)','d'),
       ('CNN-only (image + temporal query)','d'),('Physics-Gated (no ghi_lag1)','d'),
       ('Naive concat fusion','d'),('LSTM-only','d')]
def per_h(name):
    if name in R['val']['models']:
        return [p['MAE'] for p in R['val']['models'][name]['per_horizon']]
    v=T['val_imaged'][name]
    return [p['MAE'] for p in v['per_horizon']]
labels=[n.replace(' (image + temporal query)','*') for n,_ in order]
vals=np.array([per_h(n) for n,_ in order])
x=np.arange(len(order)); w=0.26
fig,ax=plt.subplots(figsize=(11,4.2))
for i,(c,lab) in enumerate([(NAVY,'t+1h'),(TEAL,'t+2h'),(ACC,'t+3h')]):
    ax.bar(x+(i-1)*w,vals[:,i],w,color=c,label=lab)
ax.axvline(3.5,color=GREY,lw=1,ls='-')
ax.text(1.5,168,'baselines  (tabular / persistence)',ha='center',fontsize=9,color=GREY)
ax.text(6.9,168,'deep fusion variants  (15.5 M parameters)',ha='center',fontsize=9,color=GREY)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=18,ha='right',fontsize=9)
ax.set_ylabel('MAE (W/m²)'); ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(0.52,0.94),fontsize=10)
ax.set_title('MAE by horizon — all models scored on the identical 914 imaged samples '
             '(2025-11-03 → 2026-02-03)',loc='left',fontsize=11)
ax.set_ylim(0,178)
plt.savefig(F+'f2_main_results.png'); plt.close()

# ── F3: skill vs smart persistence by horizon ────────────────────────────────
sp=[p['MAE'] for p in T['val_imaged']['Smart persistence']['per_horizon']]
fig,ax=plt.subplots(figsize=(6.4,3.9))
series=[('LightGBM',[p['MAE'] for p in T['val_imaged']['LightGBM']['per_horizon']],ACC,'-o'),
        ('Random Forest',[p['MAE'] for p in T['val_imaged']['Random Forest']['per_horizon']],'#8A5A2B','-s'),
        ('LSTM-only (deep)',per_h('LSTM-only'),NAVY,'-^'),
        ('Physics-Gated (deep)',per_h('Physics-Gated (large)'),TEAL,'-v'),
        ('Linear Regression',[p['MAE'] for p in T['val_imaged']['Linear Regression']['per_horizon']],GREY,'--x')]
for nm,v,c,st in series:
    sk=[100*(1-a/b) for a,b in zip(v,sp)]
    ax.plot([1,2,3],sk,st,color=c,label=nm,lw=1.8,ms=5)
ax.axhline(0,color=RED,lw=1.2)
ax.text(3.02,1,'smart persistence',color=RED,fontsize=8.5,va='bottom',ha='right')
ax.set_xticks([1,2,3]); ax.set_xticklabels(['t+1h','t+2h','t+3h'])
ax.set_ylabel('skill score vs smart persistence (%)'); ax.set_xlabel('forecast horizon')
ax.legend(fontsize=8.5,loc='upper left'); ax.set_title('Skill grows with horizon',loc='left')
plt.savefig(F+'f3_skill.png'); plt.close()

# ── F4: sample-size effect — ranking flip ────────────────────────────────────
names=['LSTM-only','Naive concat fusion','Physics-Gated (no ghi_lag1)',
       'CNN-only (image + temporal query)','Physics-Gated (large)']
v914=[R['val']['models'][n]['global']['MAE'] for n in names]
v18=[R['test']['models'][n]['global']['MAE'] for n in names]
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
for ax,vals_,ttl,c in [(axs[0],v18,'reported test split — n = 18 (2 days)',RED),
                       (axs[1],v914,'imaged validation rows — n = 914 (93 days)',NAVY)]:
    o=np.argsort(vals_)
    ax.barh([names[i].replace(' (image + temporal query)','*') for i in o],
            [vals_[i] for i in o],color=c,height=.6)
    for k,i in enumerate(o): ax.text(vals_[i]+1.5,k,f'{vals_[i]:.1f}',va='center',fontsize=9)
    ax.set_title(ttl,loc='left',fontsize=10.5); ax.set_xlabel('MAE (W/m²)')
    ax.invert_yaxis(); ax.tick_params(labelsize=8.5)
axs[0].set_xlim(0,175); axs[1].set_xlim(0,100)
fig.suptitle('The ranking of the deep variants inverts with sample size',
             x=0.02,ha='left',fontsize=11.5)
plt.tight_layout(); plt.savefig(F+'f4_rank_flip.png'); plt.close()

# ── F5: calibration ──────────────────────────────────────────────────────────
mods=['LSTM-only','Naive concat fusion','CNN-only (image + temporal query)',
      'Physics-Gated (no ghi_lag1)','Physics-Gated (large)','Physics-Gated (small)']
p914=[R['val']['models'][n]['global']['PICP'] for n in mods]
p18=[R['test']['models'][n]['global']['PICP'] for n in mods]
w914=[R['val']['models'][n]['global']['PINAW'] for n in mods]
fig,axs=plt.subplots(1,2,figsize=(10.5,3.6))
x=np.arange(len(mods))
axs[0].bar(x-0.2,p914,0.4,color=NAVY,label='n = 914')
axs[0].bar(x+0.2,p18,0.4,color=RED,label='n = 18')
axs[0].axhline(90,color='k',ls='--',lw=1.2); axs[0].text(len(mods)-0.4,91,'nominal 90%',fontsize=8.5,ha='right')
axs[0].set_ylabel('PICP (%)'); axs[0].set_ylim(0,105); axs[0].legend(fontsize=8.5)
axs[0].set_title('Coverage of the 90% interval',loc='left',fontsize=10.5)
axs[1].bar(x,w914,0.55,color=TEAL)
axs[1].set_ylabel('mean interval width (W/m²)')
axs[1].set_title('Interval width (n = 914); mean GHI = 477 W/m²',loc='left',fontsize=10.5)
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels([m.replace(' (image + temporal query)','*')
        .replace('Physics-Gated','PG') for m in mods],rotation=20,ha='right',fontsize=8)
plt.tight_layout(); plt.savefig(F+'f5_calibration.png'); plt.close()

# ── F6: forecast vs actual with band ─────────────────────────────────────────
d=np.load('analysis/preds_val_best_model_lstm.npz')
mu,sg,tg=d['mu'],d['sigma'],d['target']
gm,gs=476.8901672363281,258.44403176171875
tg_r=tg*gs+gm if tg.max()<10 else tg
s=slice(300,420)
xs=np.arange(s.stop-s.start)
fig,ax=plt.subplots(figsize=(10.5,3.4))
lo=mu[s,0]-1.645*sg[s,0]; hi=mu[s,0]+1.645*sg[s,0]
ax.fill_between(xs,lo,hi,color=NAVY,alpha=.17,label='90% predictive interval')
ax.plot(xs,mu[s,0],color=NAVY,lw=1.6,label='forecast μ (t+1h)')
ax.scatter(xs,tg[s,0],s=11,color=ACC,zorder=5,label='observed GHI')
ax.set_ylabel('GHI (W/m²)'); ax.set_xlabel('consecutive daylight hours in the validation period')
ax.legend(ncol=3,fontsize=9,loc='upper left'); ax.set_ylim(-30,1150)
ax.set_title('LSTM-only, t+1h — 120 consecutive validation samples (PICP 89.9%)',loc='left')
plt.savefig(F+'f6_forecast.png'); plt.close()

# ── F7: gate alpha ───────────────────────────────────────────────────────────
al=np.load('analysis/preds_val_best_model.npz')['alpha']
fig,axs=plt.subplots(1,2,figsize=(10.5,3.5))
axs[0].scatter(ev.clearsky_ratio,al,s=7,alpha=.35,color=NAVY,edgecolors='none')
axs[0].set_xlabel('clear-sky index $k_t$ (gate input 1)'); axs[0].set_ylabel(r'gate output $\alpha$')
axs[0].set_ylim(0,1); axs[0].axhline(0.5,color=RED,ls='--',lw=1)
axs[0].text(0.02,0.53,'α = 0.5 (equal blend)',color=RED,fontsize=8.5)
axs[0].set_title(r'$\alpha$ vs $k_t$   (r = +0.90, range 0.43–0.56)',loc='left',fontsize=10.5)
hb=A['alpha_by_hour']
axs[1].plot([int(k) for k in hb],list(hb.values()),'-o',color=TEAL,lw=1.8)
axs[1].set_ylim(0,1); axs[1].axhline(0.5,color=RED,ls='--',lw=1)
axs[1].set_xlabel('hour (SGT)'); axs[1].set_ylabel(r'mean $\alpha$')
axs[1].set_title(r'$\alpha$ by hour — full 0–1 range shown for scale',loc='left',fontsize=10.5)
plt.tight_layout(); plt.savefig(F+'f7_gate.png'); plt.close()

# ── F8: error analysis ───────────────────────────────────────────────────────
fig,axs=plt.subplots(1,2,figsize=(10.5,3.5))
labs=['<0.3','0.3–0.5','0.5–0.7','0.7–0.9','>0.9']
for nm,c,st in [('LSTM-only',NAVY,'-o'),('Physics-Gated',TEAL,'-v'),
                ('LightGBM',ACC,'-s'),('Smart persistence',GREY,'--x')]:
    y=[A['error_by_kt'][nm][b]['MAE_t1'] for b in labs]
    axs[0].plot(labs,y,st,color=c,label=nm,lw=1.7,ms=5)
ns=[A['error_by_kt']['LSTM-only'][b]['n'] for b in labs]
for i,nn in enumerate(ns): axs[0].text(i,42,f'n={nn}',ha='center',fontsize=7.5,color=GREY)
axs[0].set_xlabel('clear-sky index $k_t$ at issue time'); axs[0].set_ylabel('MAE t+1h (W/m²)')
axs[0].legend(fontsize=8.5); axs[0].set_ylim(38,105)
axs[0].set_title('Error concentrates in partly-cloudy conditions',loc='left',fontsize=10.5)
bh=A['mae_by_hour_t1']
for nm,c,st in [('LSTM-only',NAVY,'-o'),('Physics-Gated',TEAL,'-v'),('LightGBM',ACC,'-s')]:
    hs=sorted(int(k) for k in bh[nm]); axs[1].plot(hs,[bh[nm][str(h)] for h in hs],st,color=c,label=nm,lw=1.7,ms=5)
axs[1].set_xlabel('hour of day (SGT)'); axs[1].set_ylabel('MAE t+1h (W/m²)')
axs[1].legend(fontsize=8.5); axs[1].set_title('Afternoon convection is the hardest regime',loc='left',fontsize=10.5)
plt.tight_layout(); plt.savefig(F+'f8_errors.png'); plt.close()

# ── F9: training curves (parsed from the notebook log) ───────────────────────
import re, json as _j
nb=_j.load(open('solar_pv_main.ipynb'))
txt=''.join([(o.get('text') if isinstance(o.get('text'),str) else ''.join(o.get('text',[])))
             for o in nb['cells'][23]['outputs']])
rows=re.findall(r'^\s*(\d+) \|\s+(-?[\d.]+)\s+([\d.]+)W/m² \|\s+(-?[\d.]+)\s+([\d.]+)W/m²',txt,re.M)
ep=[int(r[0]) for r in rows]; trn=[float(r[1]) for r in rows]; van=[float(r[3]) for r in rows]
trm=[float(r[2]) for r in rows]; vam=[float(r[4]) for r in rows]
fig,axs=plt.subplots(1,2,figsize=(10.5,3.4))
axs[0].plot(ep,trn,color=NAVY,label='train'); axs[0].plot(ep,van,color=ACC,label='validation')
axs[0].axvline(20,color=GREY,ls=':',lw=1.4); axs[0].text(20.6,max(trn)*.75,'CNN unfrozen\n(lr/10)',fontsize=8,color=GREY)
axs[0].set_xlabel('epoch'); axs[0].set_ylabel('Gaussian NLL'); axs[0].legend(fontsize=9)
axs[0].set_title('Loss',loc='left',fontsize=10.5)
axs[1].plot(ep,trm,color=NAVY,label='train'); axs[1].plot(ep,vam,color=ACC,label='validation')
axs[1].axvline(20,color=GREY,ls=':',lw=1.4)
i=int(np.argmin(van)); axs[1].scatter([ep[i]],[vam[i]],s=45,color=RED,zorder=6)
axs[1].annotate(f'best val NLL @ ep {ep[i]}\nval MAE {vam[i]:.1f} W/m²',(ep[i],vam[i]),
                xytext=(ep[i]+4,vam[i]+28),fontsize=8.5,color=RED,
                arrowprops=dict(arrowstyle='->',color=RED,lw=1))
axs[1].set_xlabel('epoch'); axs[1].set_ylabel('MAE t+1h (W/m²)'); axs[1].legend(fontsize=9)
axs[1].set_title('MAE — validation flattens while training keeps falling',loc='left',fontsize=10.5)
plt.tight_layout(); plt.savefig(F+'f9_training.png'); plt.close()

# ── F10: clear-sky convention (max k_t by hour) ──────────────────────────────
dcc=df.copy(); dcc['kt']=dcc.ghi/(dcc.ghi_clearsky+1e-6); dcc['hour']=dcc.timestamp.dt.hour
mx=dcc.groupby('hour').kt.max()
fig,ax=plt.subplots(figsize=(6.4,3.4))
ax.bar(mx.index,mx.values,color=[NAVY if v<=1.1 else RED for v in mx.values],width=.7)
ax.axhline(1.0,color='k',ls='--',lw=1.2); ax.text(8,1.03,'physical ceiling ≈ 1.0–1.1',fontsize=8.5)
for h,v in mx.items(): ax.text(h,v+0.02,f'{v:.2f}',ha='center',fontsize=8)
ax.set_xlabel('hour (SGT)'); ax.set_ylabel('max observed $k_t$ = GHI / GHI$_{clear}$')
ax.set_ylim(0,1.6)
ax.set_title('Monotone drift 0.69 → 1.43: hourly-mean GHI divided by instantaneous clear sky',
             loc='left',fontsize=10.5)
plt.savefig(F+'f10_clearsky.png'); plt.close()
print('figures written')

# ── F11: horizon degradation, compact ────────────────────────────────────────
fig,ax=plt.subplots(figsize=(6.2,3.5))
sp=[p['MAE'] for p in T['val_imaged']['Smart persistence']['per_horizon']]
pe=[p['MAE'] for p in T['val_imaged']['Persistence']['per_horizon']]
series=[('Persistence',pe,GREY,'--x'),
        ('Smart persistence',sp,RED,'--s'),
        ('Physics-Gated (deep)',per_h('Physics-Gated (large)'),TEAL,'-v'),
        ('LSTM-only (deep)',per_h('LSTM-only'),NAVY,'-^'),
        ('LightGBM',[p['MAE'] for p in T['val_imaged']['LightGBM']['per_horizon']],ACC,'-o')]
for nm,v,c,st in series:
    ax.plot([1,2,3],v,st,color=c,label=nm,lw=2.0,ms=6)
    if v[2] > 140:
        ax.text(3.06,v[2],f'{v[2]:.0f}',color=c,fontsize=9.5,va='center')
ax.text(3.06,92,'90–95',color=INK,fontsize=9.5,va='center')
ax.set_xticks([1,2,3]); ax.set_xticklabels(['t+1h','t+2h','t+3h'],fontsize=11)
ax.set_ylabel('MAE (W/m²)',fontsize=11); ax.set_xlim(0.9,3.35); ax.set_ylim(40,340)
ax.legend(fontsize=9.5,loc='upper left')
ax.tick_params(labelsize=10)
ax.set_title('Errors converge as the horizon grows',loc='left',fontsize=12)
plt.savefig(F+'f11_horizon.png'); plt.close()
print('f11 written')
