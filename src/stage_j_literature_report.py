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
        'Tái huấn luyện 8 cấu hình literature: 129 ca, cùng 22 biến, 5 seed × 10 outer folds như GBM V3. 400 lần fit.',
        'Dùng missing gốc v7; binary điền mode, biến khác median, scaler fit trên training fold. Hai XGBoost dùng XGBClassifier thật.',
        '', '| Hạng theo điểm AUC | Mô hình | Mean seed OOF AUC ± SD | 95% CI của mean seed AUC |',
        '|---:|---|---:|---|']
    for rank,(_,r) in enumerate(t.iterrows(),1):lines.append(f'| {rank} | {r.model} | {r.mean_seed_auc:.4f} ± {r.sd_seed_auc:.4f} | {r.mean_seed_auc_CI95_low:.4f}–{r.mean_seed_auc_CI95_high:.4f} |')
    lines+=['','Cấu hình literature được giữ cố định, GBM chính cố định Cand4 sau khảo sát trên cohort; bảy họ khác dùng inner CV. Đây là benchmark bổ sung, không chứng minh superiority hay equal tuning.',
        'Xếp hạng là thứ tự điểm ước lượng, không tự chứng minh khác biệt thống kê. CI/p dùng paired patient bootstrap có điều kiện trên dự đoán đã fit, 2.000 lần; p được điều chỉnh Holm cho 8 so sánh với GBM.',
        'Đây là các cấu hình tái triển khai trên cohort hiện tại, không phải hiệu năng đã công bố của các bài nguồn và không phải external validation.',
        'Manuscript/response chưa được sửa trong lần chạy này.','']
    report='\n'.join(lines);(out/'LITERATURE_V3_REPORT.md').write_text(report,encoding='utf-8')
    main=ROOT/'V3_RUN_REPORT.md';text=main.read_text(encoding='utf-8') if main.exists() else '# V3 analysis\n';marker='\n## Literature benchmark bổ sung sau lần chạy chính\n'
    text=text.split(marker)[0]+marker+'\n'+report;main.write_text(text,encoding='utf-8')
    dump(out/'report_verification.json',{'all_nine_auc_and_sd_recomputed':True,'historical_comparison_rows':len(compare),
        'auxiliary_fixed_configuration_benchmark':True,'not_an_equal_tuning_comparison':True})
    print(t[['model','mean_seed_auc','sd_seed_auc']].to_string(index=False),flush=True)

if __name__=='__main__':main()
