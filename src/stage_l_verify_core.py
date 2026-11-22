"""Check retained primary predictions against the simplified training code on one outer fold."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from round2_common import ROOT,FEATURES,FAMILIES,grid12,load_cohort,dump

def main():
    raw,y,_,_=load_cohort();X=raw[FEATURES].to_numpy();seed=42
    tr,va=next(StratifiedKFold(10,shuffle=True,random_state=seed).split(X,y))
    saved=pd.read_csv(ROOT/'outputs/stage_b_training/primary_oof_predictions.csv')
    results=[]
    for name in FAMILIES:
        pipe,grid=grid12(seed)[name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            best=GridSearchCV(pipe,grid,scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=4,error_score='raise').fit(X[tr],y[tr]).best_estimator_
        predicted=best.predict_proba(X[va])[:,1]
        g=saved[(saved.model==name)&(saved.seed==seed)&(saved.fold==0)].set_index('subject_id').loc[raw.subject_id.iloc[va]]
        error=float(np.max(np.abs(predicted-g.value.to_numpy())))
        assert error<1e-12,(name,error)
        results.append({'model':name,'seed':seed,'fold':0,'patients':len(va),'max_abs_probability_difference':error,'passed':True})
    out=ROOT/'outputs/stage_b_training'
    pd.DataFrame(results).to_csv(out/'core_refit_replay_checks.csv',index=False)
    dump(out/'core_cleanup_verification.json',{'original_held_out_probabilities_retained':True,'all_eight_families_reproduced_on_one_outer_fold':True,
        'coverage':'one prespecified fold, seed=42 fold=0, all eight families; not a rerun of all 50 outer folds',
        'primary_model':'GBM','max_abs_error':max(r['max_abs_probability_difference'] for r in results)})
    print('Eight-family core replay passed; maximum difference',max(r['max_abs_probability_difference'] for r in results),flush=True)

if __name__=='__main__':main()
