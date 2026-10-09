import json,os,subprocess,tempfile,sys,argparse
from pathlib import Path
r=Path(__file__).resolve().parents[1];results=[]
p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--mbridge-root',type=Path,required=True);a=p.parse_args()
mbridge_root=a.mbridge_root.resolve()
assert (mbridge_root/'mbridge').is_dir(), 'Missing mbridge checkout'
with tempfile.TemporaryDirectory(prefix='aq-launch-contract-') as td:
 t=Path(td);capture=t/'argv.json';fake=t/'capture-python'
 fake.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["AQ_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n');fake.chmod(0o755)
 for model,script in [('v4',r/'recipes/deepseek-v4/dsv4_flash_verl.sh'),('k26',r/'recipes/kimi-k2.6/kimi_k26_verl.sh')]:
  env=os.environ.copy();env.update(PYTHON_BIN=str(fake),AQ_CAPTURE=str(capture),TMPDIR=str(t),LOG_DIR=str(t/'logs'),OUTPUT_DIR=str(t/'output'),MODEL_PATH='/models/test',TRAIN_FILES='/data/train.parquet',VAL_FILES='/data/val.parquet',REWARD_FN_PATH='/data/reward.py',WITH_TRACKING='false',LOGGER='["console"]',NNODES='40' if model=='k26' else '8',PYTHONWARNINGS='ignore',MBRIDGE_ROOT=str(mbridge_root))
  p=subprocess.run(['bash',str(script),'trainer.total_training_steps=2'],env=env,capture_output=True,text=True,errors="replace")
  assert p.returncode==0,(model,p.stderr[-1800:]);args=json.loads(capture.read_text());assert args[:2]==['-m','verl.trainer.main_ppo'];assert args[-1]=='trainer.total_training_steps=2'
  config=dict(s.split('=',1) for s in args[2:] if '=' in s)
  assert config['algorithm.adv_estimator']=='grpo';assert config['data.train_files']=='/data/train.parquet';assert config['reward.custom_reward_function.path']=='/data/reward.py';assert config.get('distillation.enabled','False').lower()!='true'
  results.append({'profile':model,'status':'passed','entrypoint':str(script),'arguments':len(args),'checks':['main_ppo','grpo','data path override','reward path override','no distillation enabled','Hydra tail override'],'scope':'command capture only; Python training replaced with recorder; no model or NPU execution'})
a.report.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
