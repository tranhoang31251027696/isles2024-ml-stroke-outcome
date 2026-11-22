"""Run the public numerical V3 workflow; source data stay outside version control."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
steps=[['stage_00_ingest.py','--verify'],['stage_a_prepare.py'],['stage_b_train.py','--jobs','8'],
 ['stage_c_evaluate.py'],['stage_d_figures.py'],['stage_f_centre.py'],['stage_g_missing_sensitivity.py'],
 ['stage_g_missing_sensitivity.py','--evaluate'],['stage_i_literature.py'],
 ['stage_j_literature_report.py'],['stage_k_classification_metrics.py'],['stage_l_verify_core.py']]
for step in steps:
 print('Running',step,flush=True)
 subprocess.run([sys.executable,str(ROOT/'src'/step[0]),*step[1:]],check=True,cwd=ROOT)
