"""Independently check and summarize the fixed literature configurations in V3."""
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from round2_common import ROOT,SEEDS,load_cohort,dump

def main():
    out=ROOT/'outputs/stage_i_literature_benchmark'
    t=pd.read_csv(out/'literature_benchmark.csv').sort_values('mean_seed_auc',ascending=False)
    p=pd.read_csv(out/'oof_predictions_long.csv')
    primary=pd.read_csv(ROOT/'outputs/stage_b_training/primary_oof_predictions.csv')
    checks=[]
    for _,row in t.iterrows():
        g=primary[primary.model=='GBM'].rename(columns={'value':'p_poor'}) if row.model=='Fixed Cand4 GBM (this study)' else p[p.model==row.model]
        aucs=[roc_auc_score(g[g.seed==seed].y_poor,g[g.seed==seed].p_poor) for seed in SEEDS]
        assert np.isclose(np.mean(aucs),row.mean_seed_auc,atol=1e-12,rtol=0)
        assert np.isclose(np.std(aucs),row.sd_seed_auc,atol=1e-12,rtol=0)
        checks.append({'model':row.model,'recomputed_auc':np.mean(aucs),'passed':True})
    pd.DataFrame(checks).to_csv(out/'independent_point_estimate_checks.csv',index=False)
    old=pd.read_csv(ROOT/'data/reference/historical_literature_benchmark.csv')
    compare=old[['model','mean_seed_auc']].merge(t[['model','mean_seed_auc']],on='model',suffixes=('_old','_V3'),validate='one_to_one')
    compare['delta_V3_minus_old']=compare.mean_seed_auc_V3-compare.mean_seed_auc_old
    compare.to_csv(out/'old_vs_V3_literature.csv',index=False)
    lines=['# Literature benchmark — V3','',
        'Eight fixed literature configurations were refitted on 129 patients with the same 22 predictors and 5 seeds × 10 outer folds used for the V3 GBM (400 fits).',
        'Official v7 source missingness was used; binary variables were mode-imputed, other variables median-imputed, and scaling was fitted within training folds. Both XGBoost models used XGBClassifier.',
        '', '| Rank by AUC point estimate | Model | Mean seed OOF AUC ± SD | 95% CI for mean seed AUC |',
        '|---:|---|---:|---|']
    for rank,(_,r) in enumerate(t.iterrows(),1):lines.append(f'| {rank} | {r.model} | {r.mean_seed_auc:.4f} ± {r.sd_seed_auc:.4f} | {r.mean_seed_auc_CI95_low:.4f}–{r.mean_seed_auc_CI95_high:.4f} |')
    lines+=['','Literature configurations were held fixed. The primary GBM was retained after exploratory work on this cohort, whereas the other seven families used inner CV. This auxiliary benchmark does not establish algorithm superiority or equal tuning.',
        'Ranking is based on point estimates and does not itself establish a statistical difference. CIs and p values use a conditional paired patient bootstrap with 2,000 replicates; p values are Holm-adjusted across eight comparisons with GBM.',
        'These are reimplementations on the present cohort, not the published performance of the source studies and not external validation.','']
    report='\n'.join(lines);(out/'LITERATURE_V3_REPORT.md').write_text(report,encoding='utf-8')
    main=ROOT/'V3_RUN_REPORT.md';text=main.read_text(encoding='utf-8') if main.exists() else '# V3 analysis\n';marker='\n## Auxiliary literature benchmark\n'
    text=text.split(marker)[0]+marker+'\n'+report;main.write_text(text,encoding='utf-8')
    dump(out/'report_verification.json',{'all_nine_auc_and_sd_recomputed':True,'historical_comparison_rows':len(compare),
        'auxiliary_fixed_configuration_benchmark':True,'not_an_equal_tuning_comparison':True})
    print(t[['model','mean_seed_auc','sd_seed_auc']].to_string(index=False),flush=True)

if __name__=='__main__':main()
