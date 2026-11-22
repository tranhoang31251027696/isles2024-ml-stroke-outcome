"""Threshold metrics from saved OOF predictions, not a new fit or threshold search."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
from round2_common import ROOT,dump

def main():
    out=ROOT/'outputs/stage_i_literature_benchmark'
    lit=pd.read_csv(out/'oof_predictions_long.csv')
    gbm=pd.read_csv(ROOT/'outputs/stage_b_training/primary_oof_predictions.csv')
    gbm=gbm[gbm.model=='GBM'].rename(columns={'value':'p_poor'}).copy()
    gbm['model']='Fixed Cand4 GBM (this study)'
    df=pd.concat([lit,gbm[lit.columns]],ignore_index=True)
    rows=[]
    for (model,seed),g in df.groupby(['model','seed']):
        y=g.y_poor.to_numpy();pred=(g.p_poor.to_numpy()>=.5).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn)
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        assert np.isclose(precision,precision_score(y,pred,zero_division=0))
        assert np.isclose(recall,recall_score(y,pred)) and np.isclose(f1,f1_score(y,pred,zero_division=0))
        rows.append({'model':model,'seed':seed,'TN':tn,'FP':fp,'FN':fn,'TP':tp,
            'precision':precision,'recall':recall,'F1':f1,'specificity':tn/(tn+fp),
            'NPV':tn/(tn+fn) if tn+fn else np.nan,'accuracy':accuracy_score(y,pred)})
    per_seed=pd.DataFrame(rows);per_seed.to_csv(out/'classification_metrics_by_seed.csv',index=False)
    summary=[]
    for model,g in per_seed.groupby('model'):
        row={'model':model,'threshold':.5,'positive_class':'poor outcome (90-day mRS 3-6)'}
        for metric in ['precision','recall','F1','specificity','NPV','accuracy']:
            row[metric+'_mean']=g[metric].mean();row[metric+'_sd_seed']=g[metric].std(ddof=0)
        summary.append(row)
    result=pd.DataFrame(summary)
    auc=pd.read_csv(out/'literature_benchmark.csv')[['model','mean_seed_auc']]
    result=auc.merge(result,on='model',validate='one_to_one').sort_values('mean_seed_auc',ascending=False)
    result.to_csv(out/'classification_metrics_summary.csv',index=False)
    dump(out/'classification_metrics_protocol.json',{'threshold':.5,'threshold_optimized_on_validation':False,
        'positive_class':'poor=1, 90-day mRS 3-6','aggregation':'compute on all 129 OOF predictions per seed, then average 5 seeds',
        'SD':'population SD over seeds, not a confidence interval','confusion_counts_checked_against_sklearn':True,
        'precision_zero_denominator_rule':'0, matching sklearn zero_division=0','training_rerun_needed':False})
    lines=['# Classification metrics — literature benchmark V3','',
        'Positive class: poor outcome (90-day mRS 3–6). Threshold fixed at 0.5. Metrics are calculated on 129 OOF predictions per seed and averaged over five seeds. No threshold tuning.',
        '', '| Model | Precision (%) | Recall (%) | F1 (%) | Specificity (%) | Accuracy (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for _,r in result.iterrows():
        lines.append('| '+r.model+' | '+' | '.join(f'{100*r[m+"_mean"]:.1f}' for m in ['precision','recall','F1','specificity','accuracy'])+' |')
    lines+=['','These metrics describe a fixed threshold; their ordering may differ from AUC. They are not metrics for favorable outcome as the positive class.','']
    (out/'CLASSIFICATION_METRICS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(result[['model','precision_mean','recall_mean','F1_mean','specificity_mean','accuracy_mean']].to_string(index=False))

if __name__=='__main__':main()
