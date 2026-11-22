"""Fixed Cand4 GBM plus seven inner-tuned families. Fold-local preprocessing."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['LOKY_MAX_CPU_COUNT']='8'
import argparse,time,warnings,json
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from round2_common import ROOT,FEATURES,LEAKAGE,BINARY,FAMILIES,SEEDS,grid12,preprocessing,load_cohort,dump,digest

COMPONENTS=['Age','NIHSS at admission','Hypertension','Diabetes','Atrial fibrillation']
def scores(x):
    age,nih,ht,dm,af=x.T
    return {'THRIVE_raw':np.where(age>=80,2,np.where(age>=60,1,0))+np.where(nih>=21,4,np.where(nih>=11,2,0))+ht+dm+af,
            'SPAN100_raw':(age+nih>=100).astype(float),'AgeNIHSS_raw':age+nih}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=8);a=ap.parse_args()
    out=ROOT/'outputs/stage_b_training'; run='primary'
    cp=out/'checkpoints'/run;cp.mkdir(parents=True,exist_ok=True)
    raw,y,_,_=load_cohort();X=raw[FEATURES].to_numpy();seeds=SEEDS
    cohort_hash=digest(ROOT/'outputs/stage_a_data/cohort_source_missingness.csv')
    code_hash={p.name:digest(p) for p in (ROOT/'src').glob('*.py') if p.name in ['stage_b_train.py','round2_common.py']}
    dump(out/(run+'_protocol.json'),{'seeds':seeds,'outer_folds':10,'inner_folds':3,'candidates_per_family':{n:len(g) for n,(_,g) in grid12(42).items()},'primary_selection':'fixed Cand4 after cohort exploration',
        'source_cohort_sha256':cohort_hash,'code_hashes':code_hash,'clinical_scores':'fold-imputed components; raw held-out scores and outer-training logistic calibration both saved',
        'inference':'patient bootstrap conditional on predictions; not a full retraining bootstrap',
        'primary_model':'GBM'})
    t0=time.time()
    for seed in seeds:
        for fold,(tr,va) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(X,y)):
            p=cp/f'{seed}_{fold}.csv';meta=cp/f'{seed}_{fold}.json'
            if p.exists() and meta.exists():
                old=json.loads(meta.read_text());assert old['cohort_hash']==cohort_hash and old.get('cache_compatible_code_hashes',old.get('code_hashes'))==code_hash;continue
            inner=StratifiedKFold(3,shuffle=True,random_state=seed+fold*10000)
            cfg=grid12(seed)
            names=FAMILIES
            rows=[];tuning=[];best_scores={};preds={};fit_statistics=[]
            def add(name,v):
                assert np.isfinite(v).all()
                rows.extend({'subject_id':raw.subject_id.iloc[i],'seed':seed,'fold':fold,'model':name,'y_poor':int(y[i]),'value':float(q),
                    'value_type':'raw_score' if name.endswith('_raw') else 'probability'} for i,q in zip(va,v))
            def search(name,pipe,grid,features=FEATURES):
                Z=raw[features].to_numpy()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');gs=GridSearchCV(pipe,grid,scoring='roc_auc',cv=inner,n_jobs=a.jobs,error_score='raise').fit(Z[tr],y[tr])
                tuning.append({'model':name,'n_candidates':len(gs.cv_results_['params']),'best_inner_auc':gs.best_score_,'params':gs.best_params_})
                return gs.best_estimator_,gs.best_score_
            for name in names:
                pipe,grid=cfg[name];best,score=search(name,pipe,grid);best_scores[name]=score
                pred=best.predict_proba(X[va])[:,1];preds[name]=pred;add(name,pred)
                if name in FAMILIES:
                    prep=best.named_steps['imputer']
                    fit_statistics.append({'model':name,'training_subjects':raw.subject_id.iloc[tr].tolist(),
                        'fill_values':{c:prep.named_transformers_[f'v{j}'].statistics_.tolist() for j,c in enumerate(FEATURES)},
                        'scale_mean':best.named_steps['scaler'].mean_.tolist()})
            winner=max(FAMILIES,key=lambda n:best_scores[n]);add('Selected_pipeline',preds[winner])
            # Score components use only outer-training fill values. No upstream fills.
            prep=preprocessing(COMPONENTS);Z=prep.fit_transform(raw[COMPONENTS].iloc[tr]);V=prep.transform(raw[COMPONENTS].iloc[va])
            st,sv=scores(Z),scores(V)
            for name in st:
                add(name,sv[name]);lr=LogisticRegression(C=1,max_iter=2000).fit(st[name][:,None],y[tr])
                add(name.replace('_raw','_OOF_LR'),lr.predict_proba(sv[name][:,None])[:,1])
            # Frozen sensitivities, no selection based on their outer performance.
            for label,features in [('LR22_OOF',FEATURES),('LR26_procedural_OOF',FEATURES+LEAKAGE),
                                   ('LR21_no_alert_OOF',[f for f in FEATURES if f!='Alert to door_min'])]:
                m=Pipeline([('imputer',preprocessing(features)),('scaler',StandardScaler()),('clf',LogisticRegression(C=1,max_iter=3000))])
                m.fit(raw[features].iloc[tr],y[tr]);add(label,m.predict_proba(raw[features].iloc[va])[:,1])
            keep=[f for f in FEATURES if f!='Alert to door_min'];pipe,grid=grid12(seed)['GBM'];pipe.set_params(imputer=preprocessing(keep))
            b,_=search('GBM21_no_alert',pipe,grid,keep);add('GBM21_no_alert',b.predict_proba(raw[keep].iloc[va].to_numpy())[:,1])
            prep=SimpleImputer(strategy='constant',fill_value=-1,keep_empty_features=True)
            # Missingness-only baseline has no measured values and uses training-only fit.
            missing=raw[FEATURES].isna().astype(float).to_numpy();lr=LogisticRegression(C=1,max_iter=2000).fit(missing[tr],y[tr]);add('Missingness_only_LR',lr.predict_proba(missing[va])[:,1])
            pd.DataFrame(rows).to_csv(p,index=False)
            dump(meta,{'cohort_hash':cohort_hash,'code_hashes':code_hash,'seed':seed,'fold':fold,'tuning':tuning,'fit_statistics':fit_statistics,
                       'selected_family':winner})
            print(f'{run}: seed={seed} fold={fold+1}/10 elapsed={(time.time()-t0)/60:.1f} min',flush=True)
    parts=[pd.read_csv(cp/f'{s}_{k}.csv') for s in seeds for k in range(10)]
    df=pd.concat(parts,ignore_index=True);df.to_csv(out/(run+'_oof_predictions.csv'),index=False)
    dump(out/(run+'_complete.json'),{'rows':len(df),'seeds':seeds,'elapsed_minutes':(time.time()-t0)/60,'source_cohort_sha256':cohort_hash})
    print(run+' COMPLETE',flush=True)

if __name__=='__main__':main()
