"""Reproducible auxiliary checks; never overwrite primary model outputs."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from round2_common import (ROOT, FEATURES, SEEDS, FAMILIES, _pipe, grid12, preprocessing,
                           load_cohort, bootstrap_counts, auc_boot, ci, p_two, dump, digest)

OUT = ROOT / 'outputs/stage_ac_revision_checks'
STRUCTURES = [(50,2,.05,1.), (80,3,.1,.7), (100,2,.1,.8),
              (150,2,.05,.8), (100,3,.05,1.), (150,3,.1,.7)]

def main():
    OUT.mkdir(exist_ok=True); cp=OUT/'checkpoints'; cp.mkdir(exist_ok=True)
    raw,y,_,_=load_cohort(); ids=raw.subject_id.tolist()
    checksum=digest(ROOT/'outputs/stage_a_data/cohort_source_missingness.csv')
    protocol={'cohort_sha256':checksum,'seeds':SEEDS,'structures':STRUCTURES,
              'weighting':['none','balanced'],'inner_folds':3,'outer_folds':10,
              'Cand4_in_grid':True,'selection_history':'auxiliary post-review diagnostic; not prespecified',
              'statistics':'2000 patient bootstraps, conditional on saved predictions; auxiliary p unadjusted',
              'source_sha256':digest(__file__)}
    dump(OUT/'protocol.json',protocol)
    fillrows=[]
    for seed in SEEDS:
        for fold,(tr,va) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(raw,y)):
            target=cp/f'{seed}_{fold}.csv';meta=cp/f'{seed}_{fold}.json'
            if target.exists() and meta.exists():
                saved=json.loads(meta.read_text()); assert saved['protocol']==protocol
                fillrows.append(saved['CRP_fill']); continue
            records=[]
            def add(name,p):
                records.extend({'subject_id':ids[i],'seed':seed,'fold':fold,'model':name,
                                'y_poor':int(y[i]),'value':float(v)} for i,v in zip(va,p))
            pipe=_pipe(GradientBoostingClassifier(random_state=seed))
            grid=[{'model__estimator__n_estimators':[n],'model__estimator__max_depth':[d],
                   'model__estimator__learning_rate':[lr],'model__estimator__subsample':[sub],
                   'model__weight_mode':[w]} for n,d,lr,sub in STRUCTURES for w in ['none','balanced']]
            search=GridSearchCV(pipe,grid,scoring='roc_auc',
                  cv=StratifiedKFold(3,shuffle=True,random_state=seed+fold*10000),
                  n_jobs=6,error_score='raise').fit(raw[FEATURES].to_numpy()[tr],y[tr])
            add('GBM_inner12_aux',search.predict_proba(raw[FEATURES].to_numpy()[va])[:,1])
            for name,features in [('GBM_no_CRP_aux',[f for f in FEATURES if f!='CRP']),
                                  ('GBM_CRP_missing_aux',FEATURES+['CRP_missing']),
                                  ('GBM_with_centre_aux',FEATURES+['Center'])]:
                z=raw.assign(CRP_missing=raw.CRP.isna().astype(float))[features].to_numpy()
                model=Pipeline([('imputer',preprocessing(features)),('scaler',StandardScaler()),
                    ('model',GradientBoostingClassifier(n_estimators=150,max_depth=2,
                          learning_rate=.05,subsample=.8,random_state=seed))])
                model.fit(z[tr],y[tr]);add(name,model.predict_proba(z[va])[:,1])
            for label,scaler in [('LR_C1_Standard_aux',StandardScaler()),('LR_C1_MinMax_aux',MinMaxScaler())]:
                model=Pipeline([('imputer',preprocessing()),('scaler',scaler),
                      ('model',LogisticRegression(C=1,max_iter=3000))])
                z=raw[FEATURES].to_numpy(); model.fit(z[tr],y[tr]); add(label,model.predict_proba(z[va])[:,1])
            fill={'seed':seed,'fold':fold,'training_median_CRP':float(raw.CRP.iloc[tr].median())}
            fillrows.append(fill)
            pd.DataFrame(records).to_csv(target,index=False)
            dump(meta,{'protocol':protocol,'best_params':search.best_params_,
                       'best_inner_auc':search.best_score_,'CRP_fill':fill})
            print(f'seed {seed}: fold {fold+1}/10 complete',flush=True)
    aux=pd.concat([pd.read_csv(cp/f'{s}_{f}.csv') for s in SEEDS for f in range(10)],ignore_index=True)
    aux.to_csv(OUT/'oof_predictions.csv',index=False)
    primary=pd.read_csv(ROOT/'outputs/stage_b_training/primary_oof_predictions.csv')
    df=pd.concat([aux,primary[primary.model.isin(['GBM','L2Logistic','Selected_pipeline'])]],ignore_index=True)
    counts=bootstrap_counts(y,2000,42); metrics=[]; boots={};foldrows=[]
    for name,g in df.groupby('model'):
        m=g.pivot(index='subject_id',columns='seed',values='value').loc[ids,SEEDS].to_numpy()
        a=[roc_auc_score(y,m[:,j]) for j in range(5)]
        boot=np.mean([auc_boot(y,m[:,j],counts) for j in range(5)],axis=0);boots[name]=boot
        c=ci(boot);ce=ci(auc_boot(y,m.mean(1),counts))
        row={'model':name,'mean_seed_auc':float(np.mean(a)),'sd_seed_auc':float(np.std(a)),
             'CI95_low':c[0],'CI95_high':c[1],'ensemble_auc':roc_auc_score(y,m.mean(1)),
             'ensemble_CI95_low':ce[0],'ensemble_CI95_high':ce[1],
             'brier':float(np.mean([brier_score_loss(y,m[:,j]) for j in range(5)])),
             'accuracy':float(np.mean([(m[:,j]>=.5).__eq__(y).mean() for j in range(5)])),
             'f1':float(np.mean([f1_score(y,m[:,j]>=.5) for j in range(5)]))}
        for (seed,fold),h in g.groupby(['seed','fold']):
            foldrows.append({'model':name,'seed':int(seed),'fold':int(fold),
                             'auc':roc_auc_score(h.y_poor,h.value)})
        metrics.append(row)
    perf=pd.DataFrame(metrics);perf.to_csv(OUT/'performance.csv',index=False)
    pd.DataFrame(foldrows).to_csv(OUT/'within_fold_auc.csv',index=False)
    indexed=perf.set_index('model');pairs=[]
    for first,second in [('GBM',n) for n in indexed.index if n!='GBM']+[
               ('LR_C1_Standard_aux','L2Logistic'),('LR_C1_Standard_aux','LR_C1_MinMax_aux')]:
        delta=boots[first]-boots[second];c=ci(delta)
        pairs.append({'first':first,'second':second,
               'delta_mean_seed_auc':indexed.loc[first,'mean_seed_auc']-indexed.loc[second,'mean_seed_auc'],
               'CI95_low':c[0],'CI95_high':c[1],'unadjusted_p':p_two(delta)})
    pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
    crp=[]
    for center,g in raw.groupby('Center'):
        crp.append({'centre':int(center),'n':len(g),'poor':int(g.y_poor.sum()),
                    'CRP_missing':int(g.CRP.isna().sum()),'CRP_observed':int(g.CRP.notna().sum())})
    pd.DataFrame(crp).to_csv(OUT/'CRP_by_centre.csv',index=False)
    pd.DataFrame(fillrows).to_csv(OUT/'training_CRP_medians.csv',index=False)
    meta=[json.loads(p.read_text()) for p in (ROOT/'outputs/stage_b_training/checkpoints/primary').glob('*.json')]
    dump(OUT/'family_selection_audit.json',{'eligible_families':FAMILIES,
             'GBM_configuration':'fixed Cand4; one candidate, not the auxiliary inner12 grid',
             'selected_counts':pd.Series([m['selected_family'] for m in meta]).value_counts().to_dict(),
             'total_outer_folds':len(meta)})
    print(perf.to_string(index=False),flush=True)

if __name__=='__main__':main()
