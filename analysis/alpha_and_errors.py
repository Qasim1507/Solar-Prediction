"""Gate-behaviour and error analysis on the 914 imaged validation rows."""
import numpy as np, pandas as pd, json, sys
from sklearn.metrics import mean_absolute_error
sys.path.insert(0,'.')
df = pd.read_csv('combined_dataset.csv'); df['timestamp']=pd.to_datetime(df.timestamp)
df = df.sort_values('timestamp').reset_index(drop=True)
df = df[df.timestamp < '2026-08-16'].reset_index(drop=True)
df['clearsky_ratio']=df.ghi/(df.ghi_clearsky+1e-6)
df['hour']=df.timestamp.dt.hour; df['month']=df.timestamp.dt.month
df['sin_hour']=np.sin(2*np.pi*df.hour/24); df['cos_hour']=np.cos(2*np.pi*df.hour/24)
df['sin_month']=np.sin(2*np.pi*df.month/12); df['cos_month']=np.cos(2*np.pi*df.month/12)
F=["clearsky_ratio","cloud_cover","temperature_2m","rain","wind_speed_10m",
   "relative_humidity_2m","sin_hour","cos_hour","sin_month","cos_month",
   "ghi_clearsky","ghi_lag1","ghi_lag2","ghi_lag3"]
for l in (1,2,3): df[f'ghi_lag{l}']=df.ghi.shift(l)
for h in (1,2,3):
    df[f'ghi_t{h}h']=df.ghi.shift(-h); df[f'cs_t{h}h']=df.ghi_clearsky.shift(-h)
dc=df.dropna(subset=F+['ghi_t1h','ghi_t2h','ghi_t3h','cs_t1h','cs_t2h','cs_t3h']).copy()
n=len(dc); t_end=int(n*.70); v_end=int(n*.85)
va=dc.iloc[t_end:v_end].reset_index(drop=True)
vi=[i for i in range(23,len(va)-3) if isinstance(va.image_path.iloc[i],str)]
ev=va.iloc[vi].reset_index(drop=True)
ev.to_csv('analysis/val_rows_914.csv', index=False)
a=np.load('analysis/preds_val_best_model.npz')
alpha=a['alpha']; mu=a['mu']; tg=a['target']
print('alpha stats: mean %.4f sd %.4f min %.4f max %.4f range %.4f'%(
    alpha.mean(),alpha.std(),alpha.min(),alpha.max(),alpha.max()-alpha.min()))
for col in ['clearsky_ratio','cloud_cover','hour','ghi']:
    print(f'  corr(alpha, {col}) = {np.corrcoef(alpha, ev[col].values)[0,1]:+.3f}')
out={'alpha_mean':float(alpha.mean()),'alpha_sd':float(alpha.std()),
     'alpha_min':float(alpha.min()),'alpha_max':float(alpha.max()),
     'corr':{c: float(np.corrcoef(alpha, ev[c].values)[0,1])
             for c in ['clearsky_ratio','cloud_cover','hour','ghi']}}
g=pd.DataFrame({'alpha':alpha,'hour':ev.hour,'cc':ev.cloud_cover,'cr':ev.clearsky_ratio})
out['alpha_by_hour']=g.groupby('hour').alpha.mean().round(4).to_dict()
g['ccdec']=pd.qcut(g.cc,10,duplicates='drop',labels=False)
out['alpha_by_cc_decile']=g.groupby('ccdec').alpha.mean().round(4).to_dict()
print(out['alpha_by_hour']); print(out['alpha_by_cc_decile'])

# error stratification for LSTM-only and Physics-Gated
tab=np.load('analysis/tabular_val_preds.npz')
lstm=np.load('analysis/preds_val_best_model_lstm.npz')['mu']
pg=mu; act=tg
ev['kt']=ev.clearsky_ratio
bins=[0,0.3,0.5,0.7,0.9,2.0]; lab=['<0.3','0.3–0.5','0.5–0.7','0.7–0.9','>0.9']
ev['ktbin']=pd.cut(ev.kt,bins=bins,labels=lab)
strat={}
for name,P in [('LSTM-only',lstm),('Physics-Gated',pg),('LightGBM',tab['LightGBM']),
               ('Smart persistence',tab['smart_persistence'])]:
    rows={}
    for b in lab:
        m=(ev.ktbin==b).values
        if m.sum()>5: rows[b]={'n':int(m.sum()),
            'MAE_t1':float(mean_absolute_error(act[m,0],P[m,0]))}
    strat[name]=rows
    print(name, {k:(v['n'],round(v['MAE_t1'],1)) for k,v in rows.items()})
out['error_by_kt']=strat
byhour={}
for name,P in [('LSTM-only',lstm),('Physics-Gated',pg),('LightGBM',tab['LightGBM'])]:
    byhour[name]={int(h): float(mean_absolute_error(act[(ev.hour==h).values,0],
                    P[(ev.hour==h).values,0])) for h in sorted(ev.hour.unique())}
out['mae_by_hour_t1']=byhour
print(json.dumps(byhour['LSTM-only'],indent=0))
json.dump(out, open('analysis/alpha_errors.json','w'), indent=2)
print('saved')
