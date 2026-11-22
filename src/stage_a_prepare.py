"""Build the analysis cohort directly from official v7 clinical records."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from round2_common import ROOT, FEATURES, LEAKAGE, BINARY, SEEDS, dump, environment

def main():
    src=ROOT/'data/source_v7'; out=ROOT/'outputs/stage_a_data'; out.mkdir(parents=True, exist_ok=True)
    d=pd.read_csv(src/'clinical_v7_original_missingness.csv').sort_values('subject_id').reset_index(drop=True)
    assert len(d)==149 and d.subject_id.is_unique
    # 99 is not a valid binary value. Its undocumented meaning is not guessed.
    invalid=[]
    for c in BINARY:
        m=d[c].notna() & ~d[c].isin([0,1])
        invalid.extend({'subject_id':r.subject_id,'variable':c,'source_value':r[c],'action':'invalid binary -> NaN'} for _,r in d[m].iterrows())
        d.loc[m,c]=np.nan
    pd.DataFrame(invalid).to_csv(out/'invalid_codes.csv',index=False)
    assert d.loc[d['mRS 3 months'].notna(),'mRS 3 months'].isin(range(7)).all()
    excluded=d[d['mRS 3 months'].isna()]
    excluded[['subject_id']].assign(reason='90-day outcome missing').to_csv(out/'excluded_missing_outcome.csv',index=False)
    cohort=d[d['mRS 3 months'].notna()].copy().reset_index(drop=True)
    cohort['y_poor']=(cohort['mRS 3 months']>=3).astype(int)
    assert len(cohort)==129 and cohort.y_poor.sum()==44
    cohort.to_csv(out/'cohort_source_missingness.csv',index=False)
    d.to_csv(out/'all149_source_missingness.csv',index=False)
    mask=cohort[FEATURES+LEAKAGE].isna();mask.insert(0,'subject_id',cohort.subject_id)
    mask.to_csv(out/'missing_cell_mask.csv',index=False)
    pd.DataFrame([{'variable':c,'missing_all149':int(d[c].isna().sum()),'missing_cohort129':int(cohort[c].isna().sum()),
                   'pct_cohort129':100*cohort[c].isna().mean()} for c in FEATURES+LEAKAGE]).to_csv(out/'missingness.csv',index=False)
    folds=[]; y=cohort.y_poor.to_numpy()
    for seed in SEEDS:
        for fold,(tr,va) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(cohort,y)):
            inner=StratifiedKFold(3,shuffle=True,random_state=seed+fold*10000)
            for i in va:folds.append({'seed':seed,'outer_fold':fold,'level':'outer_validation','inner_fold':-1,'subject_id':cohort.subject_id.iloc[i]})
            for k,(_,iv) in enumerate(inner.split(cohort.iloc[tr],y[tr])):
                for i in tr[iv]:folds.append({'seed':seed,'outer_fold':fold,'level':'inner_validation','inner_fold':k,'subject_id':cohort.subject_id.iloc[i]})
    pd.DataFrame(folds).to_csv(out/'fold_manifest.csv',index=False)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(src.rglob('*')) if p.is_file()}
    dump(out/'source_hashes.json',hashes)
    dump(out/'data_protocol.json',{'source_record':'17652035','release':'v7','source_xlsx_crc_verified':299,
        'full_archive_md5_checked':False,'n_source':149,'n_analysis':129,'n_poor':44,'n_good':85,
        'missing_outcomes_excluded':20,'invalid_binary_codes':invalid,
        'outcome':'poor=1 for 90-day mRS 3-6','binary_imputation':'training-fold most frequent',
        'other_imputation':'training-fold median; entirely empty training column retained and filled 0',
        'scaling':'training-fold StandardScaler inside model pipelines',
        'structural_missingness':'Alert to door defined only for wake-up strokes; empty cells preserved. Sensitivity omits this variable.',
        'dictionary_units':{'Glucose':'mg/dl','Leucocytes':'G/l','CRP':'mg/dl','INR':'unitless'},
        'laboratory_anonymization':'provider describes random +/-5% alteration',
        'postprocedural_variables_excluded_from_primary':LEAKAGE,'environment':environment()})
    print('Prepared 129 cases directly from v7, preserving source missingness.',flush=True)

if __name__=='__main__':main()
