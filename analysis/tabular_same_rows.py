"""
tabular_same_rows.py — score the tabular baselines on EXACTLY the rows the deep
models were scored on (the 914 imaged validation samples), so the two families
are comparable. Also runs Diebold-Mariano tests and a same-day-only recompute.
Read-only w.r.t. the repository.
"""
import os, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from scipy import stats as sps
sys.path.insert(0,'.')
from model import persistence_forecast, smart_persistence_forecast, skill_score

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
tr=dc.iloc[:t_end].copy(); va=dc.iloc[t_end:v_end].copy(); te=dc.iloc[v_end:].copy()
va_r=va.reset_index(drop=True); te_r=te.reset_index(drop=True)
vi_val=[i for i in range(23,len(va_r)-3) if isinstance(va_r.image_path.iloc[i],str)]
print(f'clean={n} train={len(tr)} val={len(va)} test={len(te)}  imaged val rows={len(vi_val)}')

def fit_and_predict(train_df, eval_df):
    Xtr=train_df[F].values; Xev=eval_df[F].values
    preds={}
    for name,mk in [('Linear Regression',lambda:LinearRegression()),
                    ('Random Forest',lambda:RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=42)),
                    ('LightGBM',lambda:lgb.LGBMRegressor(n_estimators=200,random_state=42,verbose=-1))]:
        P=np.zeros((len(eval_df),3))
        for h in (1,2,3):
            m=mk(); m.fit(Xtr, train_df[f'ghi_t{h}h'].values)
            P[:,h-1]=m.predict(Xev)
        preds[name]=P
    return preds

def report(actual, preds_dict, ref_mae, tag):
    out={}
    for name,P in preds_dict.items():
        rows=[dict(Horizon=f't+{h+1}h',
                   MAE=mean_absolute_error(actual[:,h],P[:,h]),
                   RMSE=float(np.sqrt(mean_squared_error(actual[:,h],P[:,h]))),
                   R2=r2_score(actual[:,h],P[:,h])) for h in range(3)]
        g=dict(MAE=mean_absolute_error(actual,P),
               RMSE=float(np.sqrt(mean_squared_error(actual,P))),
               R2=r2_score(actual,P))
        g['Skill_vs_SP']=skill_score(g['MAE'],ref_mae)
        out[name]={'global':g,'per_horizon':rows}
        print(f'  [{tag}] {name:20s} MAE={g["MAE"]:7.2f} R2={g["R2"]:+.3f} skill={g["Skill_vs_SP"]:+.3f}  '
              + ' '.join(f'{r["Horizon"]}={r["MAE"]:.1f}' for r in rows))
    return out

RES={}
# ---- A. full test split (what the deck's baseline CSV reports) --------------
act_te=np.stack([te[f'ghi_t{h}h'].values for h in (1,2,3)],1)
sp_te=smart_persistence_forecast(te.ghi.values, te.ghi_clearsky.values,
                                 np.stack([te[f'cs_t{h}h'].values for h in (1,2,3)],1))
pe_te=persistence_forecast(te.ghi.values)
ref_te=mean_absolute_error(act_te,sp_te)
print(f'\nA. FULL test split n={len(te)}  smart-persistence MAE={ref_te:.2f}')
RES['test_full']={'n':len(te),
    'Smart persistence':{'global':{'MAE':ref_te,'R2':r2_score(act_te,sp_te)},
      'per_horizon':[dict(Horizon=f't+{h+1}h',MAE=mean_absolute_error(act_te[:,h],sp_te[:,h]),
                          R2=r2_score(act_te[:,h],sp_te[:,h])) for h in range(3)]},
    'Persistence':{'global':{'MAE':mean_absolute_error(act_te,pe_te),'R2':r2_score(act_te,pe_te)},
      'per_horizon':[dict(Horizon=f't+{h+1}h',MAE=mean_absolute_error(act_te[:,h],pe_te[:,h]),
                          R2=r2_score(act_te[:,h],pe_te[:,h])) for h in range(3)]}}
RES['test_full'].update(report(act_te, fit_and_predict(tr,te), ref_te, 'test n=1437'))

# ---- B. the 914 imaged validation rows (same rows as the deep models) -------
ev=va_r.iloc[vi_val]
act_v=np.stack([ev[f'ghi_t{h}h'].values for h in (1,2,3)],1)
sp_v=smart_persistence_forecast(ev.ghi.values, ev.ghi_clearsky.values,
                                np.stack([ev[f'cs_t{h}h'].values for h in (1,2,3)],1))
pe_v=persistence_forecast(ev.ghi.values)
ref_v=mean_absolute_error(act_v,sp_v)
print(f'\nB. IMAGED VAL rows n={len(ev)}  smart-persistence MAE={ref_v:.2f}')
tab_v=fit_and_predict(tr,ev)
RES['val_imaged']={'n':len(ev),
    'Smart persistence':{'global':{'MAE':ref_v,'R2':r2_score(act_v,sp_v)},
      'per_horizon':[dict(Horizon=f't+{h+1}h',MAE=mean_absolute_error(act_v[:,h],sp_v[:,h]),
                          R2=r2_score(act_v[:,h],sp_v[:,h])) for h in range(3)]},
    'Persistence':{'global':{'MAE':mean_absolute_error(act_v,pe_v),'R2':r2_score(act_v,pe_v)},
      'per_horizon':[dict(Horizon=f't+{h+1}h',MAE=mean_absolute_error(act_v[:,h],pe_v[:,h]),
                          R2=r2_score(act_v[:,h],pe_v[:,h])) for h in range(3)]}}
RES['val_imaged'].update(report(act_v, tab_v, ref_v, 'val n=914'))
np.savez('analysis/tabular_val_preds.npz', actual=act_v,
         **{k.replace(' ','_'):v for k,v in tab_v.items()},
         smart_persistence=sp_v, persistence=pe_v)

# ---- C. Diebold-Mariano: deep vs tabular on the same 914 rows --------------
def dm_test(e1,e2,h):
    """DM on absolute-error loss, Newey-West with lag h-1, small-sample corrected."""
    d=np.abs(e1)-np.abs(e2); T=len(d); dbar=d.mean(); lag=h-1
    g0=np.mean((d-dbar)**2); s=g0
    for k in range(1,lag+1):
        gk=np.mean((d[k:]-dbar)*(d[:-k]-dbar)); s+=2*(1-k/(lag+1))*gk
    var=s/T
    if var<=0: return float('nan'), float('nan')
    stat=dbar/np.sqrt(var)
    stat*= np.sqrt((T+1-2*h+h*(h-1)/T)/T)          # Harvey-Leybourne-Newbold
    p=2*(1-sps.t.cdf(abs(stat), T-1))
    return float(stat), float(p)

deep=np.load('analysis/preds_val_best_model_lstm.npz')
mu_lstm=deep['mu']
deep_pg=np.load('analysis/preds_val_best_model.npz')['mu']
assert len(mu_lstm)==len(act_v)
print('\nC. Diebold-Mariano (abs-error loss), n=914')
DM={}
for a_name,a in [('LSTM-only',mu_lstm),('Physics-Gated (large)',deep_pg)]:
    for b_name,b in [('LightGBM',tab_v['LightGBM']),('Random Forest',tab_v['Random Forest']),
                     ('Smart persistence',sp_v)]:
        for h in (1,2,3):
            st,p=dm_test(act_v[:,h-1]-a[:,h-1], act_v[:,h-1]-b[:,h-1], h)
            DM[f'{a_name} vs {b_name} t+{h}h']={'DM':st,'p':p,
                'MAE_a':mean_absolute_error(act_v[:,h-1],a[:,h-1]),
                'MAE_b':mean_absolute_error(act_v[:,h-1],b[:,h-1])}
            print(f'  {a_name:22s} vs {b_name:18s} t+{h}h: DM={st:+6.2f} p={p:.2e} '
                  f'({DM[f"{a_name} vs {b_name} t+{h}h"]["MAE_a"]:.1f} vs '
                  f'{DM[f"{a_name} vs {b_name} t+{h}h"]["MAE_b"]:.1f})')
# LSTM vs Physics-Gated
for h in (1,2,3):
    st,p=dm_test(act_v[:,h-1]-mu_lstm[:,h-1], act_v[:,h-1]-deep_pg[:,h-1], h)
    DM[f'LSTM-only vs Physics-Gated t+{h}h']={'DM':st,'p':p}
    print(f'  LSTM-only vs Physics-Gated (large) t+{h}h: DM={st:+6.2f} p={p:.3f}')
RES['DM']=DM

# ---- D. overnight-gap contamination on the 914 rows ------------------------
print('\nD. cross-day horizon contamination')
tt=ev.timestamp.reset_index(drop=True)
full_ts=dc.timestamp.reset_index(drop=True)
pos=va.index[vi_val] if False else None
frac={}
for h in (1,2,3):
    # target timestamp = timestamp of row t+h in the *positional* index
    idx=np.array(vi_val)
    tgt=va_r.timestamp.values[idx+h]
    delta=(pd.to_datetime(tgt)-pd.to_datetime(va_r.timestamp.values[idx])).total_seconds()/3600
    frac[f't+{h}h']=float((delta!=h).mean()*100)
    same=(delta==h)
    print(f'  t+{h}h: {frac[f"t+{h}h"]:.1f}% cross-day | '
          f'LSTM MAE all={mean_absolute_error(act_v[:,h-1],mu_lstm[:,h-1]):.1f} '
          f'same-day={mean_absolute_error(act_v[same,h-1],mu_lstm[same,h-1]):.1f} | '
          f'LGBM all={mean_absolute_error(act_v[:,h-1],tab_v["LightGBM"][:,h-1]):.1f} '
          f'same-day={mean_absolute_error(act_v[same,h-1],tab_v["LightGBM"][same,h-1]):.1f} | '
          f'SP all={mean_absolute_error(act_v[:,h-1],sp_v[:,h-1]):.1f} '
          f'same-day={mean_absolute_error(act_v[same,h-1],sp_v[same,h-1]):.1f}')
    RES.setdefault('crossday',{})[f't+{h}h']={
        'pct_crossday':frac[f't+{h}h'],
        'LSTM_all':mean_absolute_error(act_v[:,h-1],mu_lstm[:,h-1]),
        'LSTM_sameday':mean_absolute_error(act_v[same,h-1],mu_lstm[same,h-1]),
        'LGBM_all':mean_absolute_error(act_v[:,h-1],tab_v['LightGBM'][:,h-1]),
        'LGBM_sameday':mean_absolute_error(act_v[same,h-1],tab_v['LightGBM'][same,h-1]),
        'SP_all':mean_absolute_error(act_v[:,h-1],sp_v[:,h-1]),
        'SP_sameday':mean_absolute_error(act_v[same,h-1],sp_v[same,h-1])}
json.dump(RES, open('analysis/tabular_results.json','w'), indent=2, default=float)
print('\nSaved analysis/tabular_results.json')
