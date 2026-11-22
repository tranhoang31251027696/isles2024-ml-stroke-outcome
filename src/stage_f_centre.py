"""Exploratory actual-centre transport check: train one centre, test the other."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_auc_score,brier_score_loss
from round2_common import ROOT,FEATURES,load_cohort,grid12,bootstrap_counts,auc_boot,ci,dump

def main():
    out=ROOT/'outputs/stage_f_centre';out.mkdir(exist_ok=True)
    raw,y,_,_=load_cohort();X=raw[FEATURES].to_numpy();results=[];rows=[]
    for centre in sorted(raw.Center.unique()):
        va=np.flatnonzero(raw.Center.to_numpy()==centre);tr=np.flatnonzero(raw.Center.to_numpy()!=centre)
        inner=StratifiedKFold(3,shuffle=True,random_state=42)
        cfg=grid12(42)
        for name in ['GBM','RandomForest','SVM_RBF']:
            pipe,grid=cfg[name];gs=GridSearchCV(pipe,grid,scoring='roc_auc',cv=inner,n_jobs=2,error_score='raise').fit(X[tr],y[tr]);b=gs.best_estimator_
            pred={name:b.predict_proba(X[va])[:,1]}
            for model,p in pred.items():
                counts=bootstrap_counts(y[va],2000,42);c=ci(auc_boot(y[va],p,counts))
                results.append({'model':model,'held_out_centre':centre,'train_n':len(tr),'validation_n':len(va),
                    'validation_poor':int(y[va].sum()),'auc':roc_auc_score(y[va],p),'auc_CI95_low':c[0],'auc_CI95_high':c[1],
                    'brier':brier_score_loss(y[va],p)})
                rows.extend({'subject_id':raw.subject_id.iloc[i],'held_out_centre':centre,'model':model,'y_poor':int(y[i]),'p_poor':q} for i,q in zip(va,p))
        print('Completed held-out centre',centre,flush=True)
    pd.DataFrame(results).to_csv(out/'leave_one_centre_out_performance.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'leave_one_centre_out_predictions.csv',index=False)
    dump(out/'protocol.json',{'purpose':'Exploratory internal-external validation between the two centres in this same public cohort',
        'centre_labels':'Actual source Center field, excluded from predictors','seed':42,'inner_folds':3,'candidates_by_family':{'GBM':1,'RandomForest':12,'SVM_RBF':12},'primary_GBM':'fixed Cand4 after cohort exploration',
        'caution':'Only 5 poor outcomes in centre 2; high uncertainty. Not independent external validation.'})

if __name__=='__main__':main()
