"""
repro_eval.py — read-only reproduction of the notebook's data pipeline +
re-evaluation of every saved checkpoint on a larger, imaged split.

Nothing in the repository is modified. Outputs go to analysis/.
Reproduces solar_pv_main.ipynb cells 4/6/9/16/17/19/25 exactly, truncating the
dataset to the snapshot the training run saw (rows <= 2026-08-15 17:00).
"""
import os, json, time, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.insert(0, os.path.abspath('.'))
from model import (PhysicsGatedFusionModelV2, TABULAR_COLS,
                   persistence_forecast, smart_persistence_forecast, skill_score)

IMG_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMG_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)
SAT_DIR, NPY_DIR = 'data/satellite', 'data/satellite_npy'
ZEROS = torch.zeros((3,224,224), dtype=torch.float32)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── cell 4 ───────────────────────────────────────────────────────────────────
df = pd.read_csv('combined_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df = df[df['timestamp'] < '2026-08-16'].reset_index(drop=True)   # training snapshot
df['clearsky_ratio'] = df['ghi']/(df['ghi_clearsky']+1e-6)
df['hour']  = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['sin_hour']=np.sin(2*np.pi*df['hour']/24); df['cos_hour']=np.cos(2*np.pi*df['hour']/24)
df['sin_month']=np.sin(2*np.pi*df['month']/12); df['cos_month']=np.cos(2*np.pi*df['month']/12)
print(f'df: {df.shape}  {df.timestamp.min()} -> {df.timestamp.max()}')

# ── cell 6 ───────────────────────────────────────────────────────────────────
BASELINE_FEATURES = ["clearsky_ratio","cloud_cover","temperature_2m","rain",
    "wind_speed_10m","relative_humidity_2m","sin_hour","cos_hour","sin_month",
    "cos_month","ghi_clearsky","ghi_lag1","ghi_lag2","ghi_lag3"]
for l in (1,2,3): df[f'ghi_lag{l}'] = df['ghi'].shift(l)
for h in (1,2,3):
    df[f'ghi_t{h}h']=df['ghi'].shift(-h); df[f'cs_t{h}h']=df['ghi_clearsky'].shift(-h)
df_clean = df.dropna(subset=BASELINE_FEATURES+['ghi_t1h','ghi_t2h','ghi_t3h',
                                               'cs_t1h','cs_t2h','cs_t3h']).copy()
print(f'clean rows: {len(df_clean)}  (notebook printed 9574)')

# ── cell 16: image cache (lazy) ──────────────────────────────────────────────
# The notebook's .npy cache was grayscale-converted (cell 8 + cell 9): PNG ->
# L -> resize 224 -> /255 -> stacked to 3 identical channels. The .npy files in
# data/satellite_npy/ on this machine are the older RGB version, so frames are
# rebuilt from the source PNGs here to match training exactly.
from PIL import Image
GRAY_DIR = os.environ.get('GRAY_CACHE', 'analysis/gray_cache')
os.makedirs(GRAY_DIR, exist_ok=True)
def gray_path(p):
    return os.path.join(GRAY_DIR, os.path.relpath(p, SAT_DIR).replace('/','_')
                        ).replace('.png','.npy')
_cache = {}
def get_img(p):
    if not isinstance(p,str): return None
    if p in _cache: return _cache[p]
    g = gray_path(p); a = None
    if os.path.exists(g):
        a = np.load(g)
    elif os.path.exists(p):
        with Image.open(p) as im:
            arr = np.array(im.convert('L').resize((224,224), Image.BILINEAR),
                           dtype=np.float32)/255.0
        a = np.stack([arr,arr,arr], axis=-1)
        np.save(g, a)
    t = None
    if a is not None:
        a = (a-IMG_MEAN)/IMG_STD
        t = torch.from_numpy(a.transpose(2,0,1)).float()
    if len(_cache) < 3000: _cache[p] = t
    return t

# ── cell 17: optical flow ────────────────────────────────────────────────────
import cv2
FLOW_SIZE=64; FLOW_CENTER=slice(16,48)
def to_gray64(t):
    a = np.clip(t[0].numpy()*0.229+0.485,0,1); a=(a*255).astype(np.uint8)
    return cv2.resize(a,(FLOW_SIZE,FLOW_SIZE),interpolation=cv2.INTER_LINEAR)
all_paths = df_clean['image_path'].values
_flow = {}
def flow_for(i):
    p = all_paths[i]
    if not isinstance(p,str): return torch.zeros(2)
    if p in _flow: return _flow[p]
    cur = get_img(p); prev = get_img(all_paths[i-1]) if i>0 else None
    if cur is None or prev is None:
        v = torch.zeros(2)
    else:
        fl = cv2.calcOpticalFlowFarneback(to_gray64(prev), to_gray64(cur), None,
            pyr_scale=0.5, levels=3, winsize=11, iterations=3, poly_n=5,
            poly_sigma=1.1, flags=0)
        v = torch.tensor([float(fl[FLOW_CENTER,FLOW_CENTER,0].mean())/FLOW_SIZE,
                          float(fl[FLOW_CENTER,FLOW_CENTER,1].mean())/FLOW_SIZE])
    _flow[p]=v; return v

# ── cell 19: dataset + split ─────────────────────────────────────────────────
class GHIDS(Dataset):
    def __init__(self, dframe, offset, is_train, stats=None, feature_cols=None):
        self.fc = list(feature_cols or TABULAR_COLS)
        d = dframe.reset_index(drop=True)
        self.off = offset                   # row offset into df_clean (for flow)
        self.tab = d[self.fc].values.astype(np.float32)
        self.cs  = d['ghi_clearsky'].values.astype(np.float32)
        self.ghi = d['ghi'].values.astype(np.float32)
        self.paths = d['image_path'].values
        self.cr  = d['clearsky_ratio'].values.astype(np.float32)
        self.cc  = d['cloud_cover'].values.astype(np.float32)
        self.df  = d
        if is_train:
            self.mean=self.tab.mean(0); self.std=self.tab.std(0)+1e-6
            self.ghi_mean=float(self.ghi.mean()); self.ghi_std=float(self.ghi.std())+1e-6
        else:
            self.mean=np.asarray(stats['mean']); self.std=np.asarray(stats['std'])
            self.ghi_mean=stats['ghi_mean']; self.ghi_std=stats['ghi_std']
        self.tab=(self.tab-self.mean)/self.std
        self.valid_indices=[i for i in range(23,len(d)-3)
                            if isinstance(d['image_path'].iloc[i],str)]
    def _frame(self,i):
        if i<0: return ZEROS
        f = get_img(self.paths[i]); return f if f is not None else ZEROS
    def __len__(self): return len(self.valid_indices)
    def __getitem__(self,k):
        t=self.valid_indices[k]
        tab=torch.from_numpy(self.tab[t-23:t+1])
        fcs=torch.from_numpy(self.cs[t+1:t+4])
        tgt=torch.tensor((self.ghi[t+1:t+4]-self.ghi_mean)/self.ghi_std, dtype=torch.float32)
        f0,f1,f2=self._frame(t),self._frame(t-1),self._frame(t-2)
        multi=torch.cat([f0,f1,f2],0)
        roi=F.interpolate(f0[:,56:168,56:168].unsqueeze(0),size=(224,224),
                          mode='bilinear',align_corners=False).squeeze(0)
        fl=flow_for(self.off+t)
        gate=torch.tensor([float(self.cr[t]),float(self.cc[t])/100.0,
                           float(fl[0]),float(fl[1])],dtype=torch.float32)
        return {'tabular_seq':tab,'multi_frame':multi,'roi_image':roi,
                'future_clearsky':fcs,'targets':tgt,'gate_features':gate,'idx':t}

n=len(df_clean); t_end=int(n*.70); v_end=int(n*.85)
tr_df=df_clean.iloc[:t_end].copy(); va_df=df_clean.iloc[t_end:v_end].copy()
te_df=df_clean.iloc[v_end:].copy()
print(f'rows  train {len(tr_df)} val {len(va_df)} test {len(te_df)}   '
      f'(notebook printed 6701/1436/1437)')

def build(fcols=None):
    tr=GHIDS(tr_df,0,True,feature_cols=fcols)
    st={'mean':tr.mean,'std':tr.std,'ghi_mean':tr.ghi_mean,'ghi_std':tr.ghi_std}
    va=GHIDS(va_df,t_end,False,st,fcols); te=GHIDS(te_df,v_end,False,st,fcols)
    return tr,va,te,st
tr_ds,va_ds,te_ds,STATS = build()
print(f'deep  train {len(tr_ds)} val {len(va_ds)} test {len(te_ds)}   '
      f'(notebook printed 6675/914/18)')
saved=json.load(open('train_stats.json'))
print('stats match train_stats.json: ',
      np.allclose(STATS['mean'],saved['mean'],atol=1e-3),
      np.allclose(STATS['std'],saved['std'],atol=1e-3),
      round(STATS['ghi_mean'],2), round(saved['ghi_mean'],2))
for nm,d in (('train',tr_ds),('val',va_ds),('test',te_ds)):
    ts=d.df.iloc[d.valid_indices]['timestamp']
    print(f'  {nm}: n={len(d)}  {ts.min()} -> {ts.max()}')

# ── metrics ──────────────────────────────────────────────────────────────────
def crps_gaussian(y,mu,sig):
    from math import sqrt,pi
    from scipy.stats import norm
    z=(y-mu)/sig
    return float(np.mean(sig*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))))

def evaluate(ckpt, loader, ds, stats, tag):
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    cfg=json.load(open(ckpt+'.config.json'))
    kw={k:cfg[k] for k in ('backbone','lstm_hidden','num_heads','hidden_dim',
                           'dropout','n_tabular','ablation') if k in cfg}
    m=PhysicsGatedFusionModelV2(imagenet_init=False,**kw)
    m.load_state_dict(state); m.eval().to(device)
    mus,sigs,tgs,alphas=[],[],[],[]
    t0=time.time()
    with torch.no_grad():
        for b in loader:
            out=m(b['tabular_seq'].to(device),b['multi_frame'].to(device),
                  b['roi_image'].to(device),b['future_clearsky'].to(device),
                  b['gate_features'].to(device),return_attn=True)
            mu,sg,al,_=out
            mus.append(mu.float().cpu()); sigs.append(sg.float().cpu())
            tgs.append(b['targets']); alphas.append(al.float().cpu().view(-1))
    mu=torch.cat(mus).numpy(); sg=torch.cat(sigs).numpy(); tg=torch.cat(tgs).numpy()
    al=torch.cat(alphas).numpy()
    gm,gs=stats['ghi_mean'],stats['ghi_std']
    mu_r=mu*gs+gm; sg_r=sg*gs; tg_r=tg*gs+gm
    del m
    print(f'    {tag}: {len(mu_r)} samples in {time.time()-t0:.0f}s')
    return mu_r,sg_r,tg_r,al

def metrics(mu_r,sg_r,tg_r,ref_mae):
    rows=[]
    for h in range(3):
        lo=mu_r[:,h]-1.645*sg_r[:,h]; hi=mu_r[:,h]+1.645*sg_r[:,h]
        rows.append(dict(Horizon=f't+{h+1}h',
            MAE=mean_absolute_error(tg_r[:,h],mu_r[:,h]),
            RMSE=float(np.sqrt(mean_squared_error(tg_r[:,h],mu_r[:,h]))),
            R2=r2_score(tg_r[:,h],mu_r[:,h]),
            PICP=float(((tg_r[:,h]>=lo)&(tg_r[:,h]<=hi)).mean()*100),
            PINAW=float(np.mean(hi-lo)),
            CRPS=crps_gaussian(tg_r[:,h],mu_r[:,h],sg_r[:,h])))
    g=dict(MAE=mean_absolute_error(tg_r,mu_r),
           RMSE=float(np.sqrt(mean_squared_error(tg_r,mu_r))),
           R2=r2_score(tg_r,mu_r),
           PICP=float(((tg_r>=mu_r-1.645*sg_r)&(tg_r<=mu_r+1.645*sg_r)).mean()*100),
           PINAW=float(np.mean(2*1.645*sg_r)),
           CRPS=crps_gaussian(tg_r.ravel(),mu_r.ravel(),sg_r.ravel()))
    g['Skill_vs_SP']=skill_score(g['MAE'],ref_mae)
    return g,rows

def baselines(ds):
    idx=np.array(ds.valid_indices)
    ghi=ds.ghi[idx]; cs=ds.cs[idx]
    csf=np.stack([ds.cs[idx+h] for h in (1,2,3)],1)
    act=np.stack([ds.ghi[idx+h] for h in (1,2,3)],1)
    out={}
    for nm,pred in [('Persistence',persistence_forecast(ghi)),
                    ('Smart persistence',smart_persistence_forecast(ghi,cs,csf))]:
        out[nm]=dict(MAE=mean_absolute_error(act,pred),
            RMSE=float(np.sqrt(mean_squared_error(act,pred))),
            R2=r2_score(act,pred),
            per_h=[dict(Horizon=f't+{h+1}h',
                        MAE=mean_absolute_error(act[:,h],pred[:,h]),
                        RMSE=float(np.sqrt(mean_squared_error(act[:,h],pred[:,h]))),
                        R2=r2_score(act[:,h],pred[:,h])) for h in range(3)])
    return out,act,idx


# ── run ──────────────────────────────────────────────────────────────────────
NO_LAG = [c for c in TABULAR_COLS if c!='ghi_lag1']
CKPTS = [
    ('Physics-Gated (large)','best_model.pt',None),
    ('LSTM-only','best_model_lstm.pt',None),
    ('CNN-only (image + temporal query)','best_model_cnn.pt',None),
    ('Naive concat fusion','best_model_concat.pt',None),
    ('Physics-Gated (small)','best_model_small.pt',None),
    ('Physics-Gated (no ghi_lag1)','best_model_nolag.pt',NO_LAG),
]
BS=32
results={}
for split_name in ('val','test'):
    bl_ds = va_ds if split_name=='val' else te_ds
    bl,act,idx = baselines(bl_ds)
    ref = bl['Smart persistence']['MAE']
    print(f'\n=== {split_name} split (n={len(bl_ds)}) — smart persistence MAE {ref:.2f}')
    results[split_name]={'n':len(bl_ds),'baselines':bl,'models':{}}
    for label,ck,fcols in CKPTS:
        if fcols is not None:
            _,va2,te2,st2 = build(fcols)
            ds = va2 if split_name=='val' else te2
        else:
            ds, st2 = bl_ds, STATS
        dl=DataLoader(ds,batch_size=BS,shuffle=False,num_workers=0)
        mu,sg,tg,al = evaluate(ck,dl,ds,st2,f'{label}/{split_name}')
        g,rows = metrics(mu,sg,tg,ref)
        g['alpha_mean']=float(al.mean()); g['alpha_min']=float(al.min())
        g['alpha_max']=float(al.max()); g['alpha_std']=float(al.std())
        results[split_name]['models'][label]={'global':g,'per_horizon':rows}
        np.savez(f'analysis/preds_{split_name}_{ck.replace(".pt","")}.npz',
                 mu=mu,sigma=sg,target=tg,alpha=al)
        print(f'  {label:38s} MAE={g["MAE"]:7.2f}  R2={g["R2"]:+.3f}  '
              f'PICP={g["PICP"]:5.1f}  CRPS={g["CRPS"]:6.2f}  skill={g["Skill_vs_SP"]:+.3f}  '
              f'alpha={g["alpha_mean"]:.3f}[{g["alpha_min"]:.3f},{g["alpha_max"]:.3f}]')
    json.dump(results, open('analysis/repro_results.json','w'), indent=2, default=float)
print('\nSaved analysis/repro_results.json')
