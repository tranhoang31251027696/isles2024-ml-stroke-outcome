"""Publication-style plots regenerated from V3 outputs; no historical figure copying."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from round2_common import ROOT,FAMILIES,FEATURES,load_cohort,dump,grid12

plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
LABEL={'GBM':'Fixed GBM',
       'THRIVE_OOF_LR':'THRIVE (OOF calibration)','SPAN100_OOF_LR':'SPAN-100 (OOF calibration)'}
COLORS=['#0072B2','#009E73','#CC79A7']
OUT=ROOT/'outputs/stage_d_figures'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    with Image.open(OUT/(name+'.png')) as im:im.convert('RGB').save(OUT/(name+'.tiff'),compression='tiff_lzw',dpi=(300,300))
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);tab=ROOT/'outputs/stage_c_evaluation'
    perf=pd.read_csv(tab/'model_performance.csv').set_index('model');roc=pd.read_csv(tab/'roc_points.csv');cal=pd.read_csv(tab/'calibration_bins.csv');dca=pd.read_csv(tab/'dca_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,raw_scores in zip(axes,[False,True]):
        for (name,label),color in zip(LABEL.items(),COLORS):
            if raw_scores and name.endswith('_OOF_LR'):
                name=name.replace('_OOF_LR','_raw');label=name.replace('_raw','')+' (fold-imputed raw score)'
            g=roc[roc.model==name];ax.plot(g.FPR,g.TPR,label=f'{label}: AUC {perf.loc[name,"ensemble_auc"]:.3f}',color=color,lw=1.8)
        ax.plot([0,1],[0,1],'--',color='gray',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='True positive rate')
        ax.set_title('B. Mean raw clinical scores' if raw_scores else 'A. Mean OOF probabilities',fontsize=10)
        ax.legend(loc='lower right',fontsize=7)
    fig.tight_layout();save(fig,'fig_roc_oof_probabilities')
    fig,ax=plt.subplots(figsize=(7,5.5))
    for (name,label),color in zip(LABEL.items(),COLORS):
        g=cal[cal.model==name];ax.plot(g.mean_prediction,g.observed_fraction,'o-',label=label,color=color)
    ax.plot([0,1],[0,1],'--',color='gray',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability of poor outcome',ylabel='Observed poor-outcome fraction')
    ax.legend(fontsize=8);fig.tight_layout();save(fig,'fig_calibration_oof')
    fig,ax=plt.subplots(figsize=(7,5.5))
    for flag in ['GBM']:
        g=dca[dca.model==flag];ax.plot(g.threshold,g.NB,label=LABEL[flag],color=COLORS[0]);ax.fill_between(g.threshold,g.NB_CI95_low,g.NB_CI95_high,color=COLORS[0],alpha=.12)
        for score,color in zip(['THRIVE','SPAN100'],COLORS[1:]):ax.plot(g.threshold,g[score+'_OOF_LR_NB'],label=score+' OOF',color=color)
        ax.plot(g.threshold,g.treat_all,'--',color='gray',label='Treat all');ax.axhline(0,color='black',ls=':',label='Treat none')
        ax.set(xlabel='Threshold probability',xlim=(.1,.75),ylim=(-.15,.35));ax.legend(fontsize=7)
    ax.set_ylabel('Net benefit');fig.tight_layout();save(fig,'fig_dca_oof')
    fig,ax=plt.subplots(figsize=(7,5))
    g=perf.loc[FAMILIES].sort_values('mean_seed_auc');x=g.mean_seed_auc.to_numpy()
    ax.errorbar(x,np.arange(len(g)),xerr=np.vstack([x-g.mean_seed_auc_CI95_low,g.mean_seed_auc_CI95_high-x]),fmt='o',color=COLORS[0],capsize=3)
    ax.set_yticks(np.arange(len(g)),g.index);ax.set(xlim=(.4,1),xlabel='Mean seed OOF AUC (conditional patient-bootstrap 95% CI)')
    fig.tight_layout();save(fig,'fig_matched_algorithm_comparison')
    fig,ax=plt.subplots(figsize=(7,5.5));m=pd.read_csv(ROOT/'outputs/stage_a_data/missingness.csv');m=m[m.variable.isin(FEATURES)].sort_values('pct_cohort129')
    ax.barh(m.variable,m.pct_cohort129,color=COLORS[0]);ax.set(xlim=(0,100),xlabel='Source missing or invalid values in analysis cohort (%)')
    fig.tight_layout();save(fig,'fig_source_missingness')
    # Descriptive full-cohort refit is kept separate from every validation prediction.
    from sklearn.model_selection import GridSearchCV,StratifiedKFold
    import shap,joblib
    raw,y,_,_=load_cohort();X=raw[FEATURES].to_numpy();pipe,grid=grid12(42)['GBM']
    fixed_params={k:v[0] for k,v in grid[0].items()}
    best=pipe.set_params(**fixed_params).fit(X,y);Z=best[:-1].transform(X);sv=np.asarray(shap.TreeExplainer(best.named_steps['model'].estimator_).shap_values(Z))
    pd.DataFrame(sv,columns=FEATURES).assign(subject_id=raw.subject_id).to_csv(tab/'descriptive_GBM_shap_values.csv',index=False)
    imp=pd.DataFrame({'feature':FEATURES,'mean_abs_shap':np.abs(sv).mean(0)}).sort_values('mean_abs_shap')
    imp.to_csv(tab/'descriptive_GBM_shap_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5.5));ax.barh(imp.feature,imp.mean_abs_shap,color=COLORS[0]);ax.set_xlabel('Mean absolute SHAP value (log odds; descriptive refit)');fig.tight_layout();save(fig,'fig_shap_GBM_descriptive')
    folder=ROOT/'outputs/stage_b_training/descriptive_refit';folder.mkdir(exist_ok=True)
    joblib.dump(best,folder/'GBM_full_cohort_descriptive.joblib',compress=3)
    dump(folder/'metadata.json',{'purpose':'Descriptive full-cohort refit for SHAP only; not a validation or deployment model',
        'fixed_params':fixed_params,'features':FEATURES,'not_used_for_any_oof_metric':True})
    print('Regenerated six figure sets and descriptive SHAP from V3.',flush=True)

if __name__=='__main__':main()
