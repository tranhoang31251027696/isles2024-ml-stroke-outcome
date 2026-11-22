"""Recompute inference, tables and plot data exclusively from V3 held-out predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
import numpy as np,pandas as pd
from scipy.special import logit
from scipy.stats import fisher_exact,mannwhitneyu
from sklearn.metrics import roc_auc_score,brier_score_loss,confusion_matrix,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from round2_common import ROOT,FEATURES,BINARY,FAMILIES,SEEDS,load_cohort,bootstrap_counts,auc_boot,ci,p_two,holm,nri_idi,nb_contrib,dump

def main():
    out=ROOT/'outputs/stage_c_evaluation';out.mkdir(exist_ok=True)
    raw,y,mask,_=load_cohort();ids=raw.subject_id.tolist();n=len(y)
    df=pd.read_csv(ROOT/'outputs/stage_b_training/primary_oof_predictions.csv')
    counts=bootstrap_counts(y,2000,42);M={};boots={};stats=[];seedsrows=[];calrows=[];rocrows=[]
    for name,g in df.groupby('model'):
        m=g.pivot(index='subject_id',columns='seed',values='value').loc[ids,SEEDS].to_numpy();assert np.isfinite(m).all();M[name]=m
        aucs=[roc_auc_score(y,m[:,j]) for j in range(len(SEEDS))]
        boots[name]=np.mean([auc_boot(y,m[:,j],counts) for j in range(len(SEEDS))],axis=0)
        p=m.mean(1);b=auc_boot(y,p,counts);c=ci(boots[name]);ce=ci(b)
        r={'model':name,'mean_seed_auc':np.mean(aucs),'sd_seed_auc':np.std(aucs),'mean_seed_auc_CI95_low':c[0],
           'mean_seed_auc_CI95_high':c[1],'ensemble_auc':roc_auc_score(y,p),'ensemble_auc_CI95_low':ce[0],'ensemble_auc_CI95_high':ce[1],
           'value_type':'raw_score' if name.endswith('_raw') else 'probability'}
        for j,seed in enumerate(SEEDS):seedsrows.append({'model':name,'seed':seed,'auc':aucs[j]})
        if not name.endswith('_raw'):
            assert ((m>=0)&(m<=1)).all()
            metrics=[]
            for j in range(len(SEEDS)):
                tn,fp,fn,tp=confusion_matrix(y,m[:,j]>=.5,labels=[0,1]).ravel()
                metrics.append({'brier':brier_score_loss(y,m[:,j]),'accuracy':(tp+tn)/n,'sensitivity_poor':tp/(tp+fn),
                    'specificity_good':tn/(tn+fp),'PPV':tp/(tp+fp) if tp+fp else np.nan,'NPV':tn/(tn+fn) if tn+fn else np.nan})
            for k in metrics[0]:r['mean_seed_'+k]=float(np.nanmean([v[k] for v in metrics]))
            z=logit(np.clip(p,1e-6,1-1e-6))[:,None];cal=LogisticRegression(C=1e9,max_iter=3000).fit(z,y)
            r.update(ensemble_brier=brier_score_loss(y,p),ensemble_calibration_intercept=cal.intercept_[0],ensemble_calibration_slope=cal.coef_[0,0])
            obs,pr=calibration_curve(y,p,n_bins=8,strategy='quantile')
            calrows.extend({'model':name,'mean_prediction':v,'observed_fraction':w} for v,w in zip(pr,obs))
        fpr,tpr,_=roc_curve(y,p);rocrows.extend({'model':name,'FPR':f,'TPR':t,'auc_of_plotted_vector':roc_auc_score(y,p)} for f,t in zip(fpr,tpr))
        stats.append(r)
    perf=pd.DataFrame(stats).sort_values('mean_seed_auc',ascending=False);perf.to_csv(out/'model_performance.csv',index=False)
    pd.DataFrame(seedsrows).to_csv(out/'seed_performance.csv',index=False)
    pd.DataFrame(calrows).to_csv(out/'calibration_bins.csv',index=False);pd.DataFrame(rocrows).to_csv(out/'roc_points.csv',index=False)
    pd.DataFrame({'subject_id':ids,'y_poor':y,**{k:v.mean(1) for k,v in M.items()}}).to_csv(out/'patient_mean_predictions.csv',index=False)
    ix=perf.set_index('model');pair=[]
    for flagship in ['GBM']:
        refs=[f for f in FAMILIES if f!=flagship]
        group=[]
        for ref in refs:
            delta=boots[flagship]-boots[ref];c=ci(delta)
            group.append({'model':flagship,'reference':ref,'delta_mean_seed_auc':ix.loc[flagship,'mean_seed_auc']-ix.loc[ref,'mean_seed_auc'],
                'CI95_low':c[0],'CI95_high':c[1],'p_two_sided_conditional':p_two(delta)})
        for r,p in zip(group,holm([r['p_two_sided_conditional'] for r in group])):r['p_Holm_within_comparison_set']=p
        pair+=group
    pd.DataFrame(pair).to_csv(out/'paired_ml_comparisons.csv',index=False)
    clinical=[];dca=[]
    for flagship in ['GBM']:
        group=[];p=M[flagship].mean(1)
        for score in ['THRIVE','SPAN100','AgeNIHSS']:
            for variant in ['raw','OOF_LR']:
                ref=score+'_'+variant;delta=boots[flagship]-boots[ref];c=ci(delta)
                r={'model':flagship,'reference':ref,'reference_mean_seed_auc':ix.loc[ref,'mean_seed_auc'],
                   'delta_mean_seed_auc':ix.loc[flagship,'mean_seed_auc']-ix.loc[ref,'mean_seed_auc'],
                   'CI95_low':c[0],'CI95_high':c[1],'p_two_sided_conditional':p_two(delta)}
                if variant=='OOF_LR':r.update(nri_idi(y,p,M[ref].mean(1),counts))
                group.append(r)
        # Raw-score and calibrated-probability contrasts are separate comparison sets.
        for variant in ['raw','OOF_LR']:
            selected=[r for r in group if r['reference'] in ['THRIVE_'+variant,'SPAN100_'+variant]]
            for r,q in zip(selected,holm([r['p_two_sided_conditional'] for r in selected])):r['p_Holm_two_prespecified_scores']=q
        clinical+=group
        for t in np.round(np.arange(.10,.751,.01),2):
            g=nb_contrib(y,p,t);gb=counts@g/n;c=ci(gb)
            r={'model':flagship,'threshold':t,'NB':g.mean(),'NB_CI95_low':c[0],'NB_CI95_high':c[1],
                'treat_none':0,'treat_all':y.mean()-(1-y.mean())*t/(1-t)}
            for score in ['THRIVE','SPAN100','AgeNIHSS']:
                ref=score+'_OOF_LR';v=nb_contrib(y,M[ref].mean(1),t);delta=counts@(g-v)/n;c=ci(delta)
                r.update({ref+'_NB':v.mean(),'delta_vs_'+ref:(g-v).mean(),'delta_vs_'+ref+'_CI95_low':c[0],
                    'delta_vs_'+ref+'_CI95_high':c[1],'delta_vs_'+ref+'_p_two_sided':p_two(delta)})
            dca.append(r)
    pd.DataFrame(clinical).to_csv(out/'clinical_comparisons.csv',index=False);pd.DataFrame(dca).to_csv(out/'dca_results.csv',index=False)
    # Describe source-observed values only, with original missingness in every variable.
    table=[]
    for f in FEATURES:
        r={'variable':f,'missing_n':int(raw[f].isna().sum()),'missing_pct':100*raw[f].isna().mean()}
        for label,g in [('All',np.ones(n,dtype=bool)),('Good',y==0),('Poor',y==1)]:
            x=raw.loc[g,f].dropna().to_numpy();r[label+'_observed_n']=len(x)
            r[label]=f'{int((x==1).sum())}/{len(x)} ({100*(x==1).mean():.1f}%)' if f in BINARY else f'{np.median(x):.2f} [{np.quantile(x,.25):.2f}, {np.quantile(x,.75):.2f}]'
        a=raw.loc[y==1,f].dropna().to_numpy();b=raw.loc[y==0,f].dropna().to_numpy()
        r['p_observed_only']=fisher_exact([[(a==1).sum(),(a==0).sum()],[(b==1).sum(),(b==0).sum()]])[1] if f in BINARY else mannwhitneyu(a,b,alternative='two-sided').pvalue
        table.append(r)
    pd.DataFrame(table).to_csv(out/'table1_observed_characteristics.csv',index=False)
    # Paired sensitivity estimands, not training-cohort fits.
    sensitivity=[]
    for a,b in [('LR26_procedural_OOF','LR22_OOF'),('LR21_no_alert_OOF','LR22_OOF'),('GBM21_no_alert','GBM')]:
        delta=boots[a]-boots[b];c=ci(delta);sensitivity.append({'model':a,'reference':b,'delta_mean_seed_auc':ix.loc[a,'mean_seed_auc']-ix.loc[b,'mean_seed_auc'],
            'CI95_low':c[0],'CI95_high':c[1],'p_two_sided_conditional':p_two(delta)})
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity_comparisons.csv',index=False)
    # Each patient's saved outer validation fold must match the manifest exactly.
    manifest=pd.read_csv(ROOT/'outputs/stage_a_data/fold_manifest.csv');manifest=manifest[(manifest.level=='outer_validation')&manifest.seed.isin(SEEDS)]
    check=df.merge(manifest[['subject_id','seed','outer_fold']],on=['subject_id','seed'],validate='many_to_one')
    assert (check.fold==check.outer_fold).all() and len(check)==len(df)
    assert not df.duplicated(['subject_id','seed','model']).any()
    tuning=[]
    for p in (ROOT/'outputs/stage_b_training/checkpoints/primary').glob('*.json'):
        m=json.loads(p.read_text());tuning.extend(m['tuning'])
        # Independently verify each saved outer-training imputer statistic.
        for fit in m['fit_statistics']:
            training=raw.set_index('subject_id').loc[fit['training_subjects']]
            for f,values in fit['fill_values'].items():
                v=training[f].dropna();expected=(v.mode().iloc[0] if f in BINARY else v.median()) if len(v) else 0
                assert np.isclose(values[0],expected), (p,f,values,expected)
    assert all(t['n_candidates']==(1 if t['model'] in ['GBM','GBM21_no_alert'] else 12) for t in tuning)
    dump(out/'integrity_checks.json',{'all_predictions_finite':True,'probabilities_in_unit_interval':True,'fold_manifest_matches_all_predictions':True,
        'unique_patient_seed_model_predictions':True,'candidate_counts_match_actual_design':True,'GBM_candidates':1,'other_family_candidates':12,'outer_imputation_statistics_independently_verified':True,
        'n_primary_outer_fits':50,'n_patient_bootstraps':len(counts),'bootstrap_scope':'conditional on saved fitted predictions; no training bootstrap',
        'raw_score_note':'Fold-imputed raw clinical scores vary across seeds when components are missing. No whole-cohort filling or calibration used.',
        'analysis_is_not_external_validation':True})
    print(perf[perf.model.isin(FAMILIES)][['model','mean_seed_auc','sd_seed_auc','ensemble_auc']].to_string(index=False),flush=True)

if __name__=='__main__':main()
