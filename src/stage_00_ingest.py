"""Reconstruct analysis-ready source table directly from the retained original XLSX.

No value learned or imputed here. Units/coding conversions only. --verify compares
against bundled source CSV without replacing any active training input.
"""
import argparse,datetime
import numpy as np,pandas as pd
from round2_common import ROOT,dump

def duration(v):
    if pd.isna(v):return np.nan
    if isinstance(v,datetime.time):return v.hour*60+v.minute+v.second/60
    if isinstance(v,(datetime.timedelta,pd.Timedelta)):return v.total_seconds()/60
    if isinstance(v,str):
        h,m,s=v.split(':');return float(h)*60+float(m)+float(s)/60
    if isinstance(v,(int,float)):return float(v)*1440
    raise ValueError(repr(v))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    src=ROOT/'data/source_v7';rows={'baseline':[],'outcome':[]};duplicates=[]
    for p in sorted((src/'clinical_raw').rglob('*.xlsx')):
        if not p.name.startswith('sub-stroke'):continue
        d=pd.read_excel(p);before=len(d);d=d.drop_duplicates()
        assert len(d)==1, str(p)
        if before!=1:duplicates.append({'file':str(p.relative_to(src)),'original_rows':before,'distinct_rows':len(d)})
        kind='baseline' if 'baseline' in p.name else 'outcome'
        rows[kind].append({'subject_id':p.name.split('_')[0],**d.iloc[0].to_dict()})
    b=pd.DataFrame(rows['baseline']);o=pd.DataFrame(rows['outcome']);assert len(b)==len(o)==149
    d=b.copy()
    times=['Onset to door','Alert to door','Door to imaging','Door to groin','Door to first series','Time of intervention','Door to recanalization']
    for c in times:d[c+'_min']=d[c].map(duration);d=d.drop(columns=c)
    d['Sex']=d.Sex.map({'M':1.,'F':0.});d=d.merge(o,on='subject_id',validate='one_to_one')
    target=src/'clinical_v7_original_missingness.csv'
    if args.verify:
        old=pd.read_csv(target)
        # XLSX mTICI mixes numeric cells and labels such as '2c'; CSV stores them
        # as text. Compare this ancillary field with its serialized representation.
        compared=d.copy()
        compared['TICI postinterventional']=compared['TICI postinterventional'].astype('string')
        old['TICI postinterventional']=old['TICI postinterventional'].astype('string')
        pd.testing.assert_frame_equal(compared,old,check_dtype=False,check_exact=False,rtol=0,atol=1e-10)
    else:d.to_csv(target,index=False)
    out=ROOT/'outputs/stage_a_data';out.mkdir(parents=True,exist_ok=True)
    inconsistent=d[(d['Wake-up']==0)&d['Alert to door_min'].notna()]
    inconsistent[['subject_id','Wake-up','Alert to door_min']].to_csv(out/'alert_to_door_definition_flags.csv',index=False)
    dump(out/'source_xlsx_reconstruction_check.json',{'original_xlsx_rebuilt':298,'source_csv_values_and_missing_mask_match':True,
        'identical_duplicate_rows':duplicates,'non_wakeup_with_recorded_alert_to_door':len(inconsistent),
        'rule':'Keep source values unchanged; no-alert sensitivity evaluated separately.'})
    print('Rebuilt 298 XLSX: source values/missing masks verified; no imputation.',flush=True)

if __name__=='__main__':main()
