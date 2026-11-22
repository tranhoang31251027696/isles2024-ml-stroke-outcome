"""Nested missing-data sensitivities using source-defined missingness."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,warnings,time
import numpy as np,pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from round2_common import ROOT,FEATURES,BINARY,SEEDS,WeightedClassifier,preprocessing,load_cohort,grid12,dump,digest,auc_boot,bootstrap_counts,ci,p_two

DROP=['mRS at admission','mRS premorbid','Leucocytes','CRP','INR','Onset to door_min','Alert to door_min']

def train_fold(seed,fold,tr,va,raw,y,cp):
    target=cp/f'{seed}_{fold}.csv'
    if target.exists():return str(target)
    inner=StratifiedKFold(3,shuffle=True,random_state=seed+fold*10000);rows=[]
    for tag in ['drop_heavy','iterative']:
        feats=[f for f in FEATURES if f not in DROP] if tag=='drop_heavy' else FEATURES
        X=raw[feats].to_numpy()
        for name in ['GBM','SVM_RBF','RandomForest']:
            pipe,grid=grid12(seed)[name]
            if tag=='drop_heavy':prep=preprocessing(feats)
            else:
                binary=[j for j,f in enumerate(feats) if f in BINARY];cont=[j for j,f in enumerate(feats) if f not in BINARY]
                prep=ColumnTransformer([('binary',SimpleImputer(strategy='most_frequent',keep_empty_features=True),binary),
                    ('continuous',IterativeImputer(max_iter=10,random_state=seed,keep_empty_features=True),cont)],sparse_threshold=0)
            pipe.set_params(imputer=prep)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');b=GridSearchCV(pipe,grid,cv=inner,scoring='roc_auc',n_jobs=1,error_score='raise').fit(X[tr],y[tr]).best_estimator_
            pred=b.predict_proba(X[va])[:,1]
            rows.extend({'subject_id':raw.subject_id.iloc[i],'seed':seed,'fold':fold,'model':name,'analysis':tag,'y_poor':int(y[i]),'p_poor':q} for i,q in zip(va,pred))
    pd.DataFrame(rows).to_csv(target,index=False)
    return str(target)

def main():
    out=ROOT/'outputs/stage_g_missing_sensitivity';cp=out/'checkpoints';cp.mkdir(parents=True,exist_ok=True)
    raw,y,_,_=load_cohort();ids=raw.subject_id.tolist();cohort_hash=digest(ROOT/'outputs/stage_a_data/cohort_source_missingness.csv');t0=time.time()
    protocol={'drop_heavy':DROP,'retained_features':15,'continuous_iterative_imputer':'10 iterations, fit within inner/outer training, continuous/ordinal predictors only; binary predictors retain training-mode imputation',
        'source_cohort_sha256':cohort_hash,'script_sha256':digest(__file__),'same_primary_seeds_and_folds':True,'candidates_by_family':{'GBM':1,'RandomForest':12,'SVM_RBF':12},'primary_GBM':'fixed Cand4'}
    if (out/'protocol.json').exists():
        previous=json.loads((out/'protocol.json').read_text())
        assert previous['source_cohort_sha256']==cohort_hash and previous['script_sha256']==digest(__file__), 'Input/code changed: use a new run directory'
    dump(out/'protocol.json',protocol)
    from joblib import Parallel,delayed
    tasks=[]
    for seed in SEEDS:
        for fold,(tr,va) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(raw,y)):
            tasks.append((seed,fold,tr,va))
    for i,result in enumerate(Parallel(n_jobs=8,return_as='generator_unordered')(delayed(train_fold)(*args,raw,y,cp) for args in tasks),1):
        print(f'missing sensitivity {i}/50 outer folds complete',flush=True)
    df=pd.concat([pd.read_csv(cp/f'{s}_{k}.csv') for s in SEEDS for k in range(10)],ignore_index=True);df.to_csv(out/'oof_predictions.csv',index=False)
    dump(out/'training_complete.json',{'rows':len(df),'source_cohort_sha256':cohort_hash})

def evaluate():
    out=ROOT/'outputs/stage_g_missing_sensitivity';raw,y,_,_=load_cohort();ids=raw.subject_id.tolist();counts=bootstrap_counts(y)
    df=pd.read_csv(out/'oof_predictions.csv');primary=pd.read_csv(ROOT/'outputs/stage_b_training/primary_oof_predictions.csv');results=[]
    for (model,tag),g in df.groupby(['model','analysis']):
        M=g.pivot(index='subject_id',columns='seed',values='p_poor').loc[ids,SEEDS].to_numpy()
        F=primary[primary.model==model].pivot(index='subject_id',columns='seed',values='value').loc[ids,SEEDS].to_numpy()
        aucs=[roc_auc_score(y,M[:,j]) for j in range(5)];b=np.mean([auc_boot(y,M[:,j],counts) for j in range(5)],axis=0)
        bf=np.mean([auc_boot(y,F[:,j],counts) for j in range(5)],axis=0);c=ci(b);delta=b-bf;cd=ci(delta)
        results.append({'model':model,'analysis':tag,'mean_seed_auc':np.mean(aucs),'sd_seed_auc':np.std(aucs),'CI95_low':c[0],'CI95_high':c[1],
            'delta_vs_primary':np.mean(aucs)-np.mean([roc_auc_score(y,F[:,j]) for j in range(5)]),'delta_CI95_low':cd[0],'delta_CI95_high':cd[1],'p_two_sided_exploratory':p_two(delta)})
    pd.DataFrame(results).to_csv(out/'sensitivity_performance.csv',index=False)
    strata=[]
    groups={'all_three_labs_observed':raw[['Leucocytes','CRP','INR']].notna().all(axis=1).to_numpy(),
        'all_three_labs_missing':raw[['Leucocytes','CRP','INR']].isna().all(axis=1).to_numpy()}
    for label,mask in groups.items():
        for model in ['GBM','RandomForest','SVM_RBF']:
            M=primary[primary.model==model].pivot(index='subject_id',columns='seed',values='value').loc[ids,SEEDS].to_numpy()
            value=np.mean([roc_auc_score(y[mask],M[mask,j]) for j in range(5)]) if len(np.unique(y[mask]))==2 else np.nan
            strata.append({'stratum':label,'model':model,'n':int(mask.sum()),'n_poor':int(y[mask].sum()),'mean_seed_auc':value})
    pd.DataFrame(strata).to_csv(out/'auc_by_lab_availability.csv',index=False)

if __name__=='__main__':
    import sys
    if '--evaluate' in sys.argv:evaluate()
    else:main()
