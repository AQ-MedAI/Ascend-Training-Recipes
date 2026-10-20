"""Run K26 shell profiles with a fake trainer and inspect effective configuration."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'recipes/kimi-k2.6'


class LaunchProfilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recipe = self.root / 'recipe'
        shutil.copytree(RECIPE, self.recipe)
        (self.recipe / 'runtime_env.sh').write_text(
            'export MINDSPEED_ROOT=/stub/mindspeed\n'
            'export MEGATRON_ROOT=/stub/core\n'
            'export MEGATRON_BRIDGE_ROOT=/stub/bridge\n')
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text(json.dumps({'text_config': {'num_hidden_layers': 61}}))
        runner = self.root / 'python-runner'
        runner.write_text(
            f'#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n'
            "if sys.argv[1:3] == ['-m','verl.trainer.main_ppo']:\n"
            "    Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[3:]))\n"
            "elif sys.argv[1].endswith('prepare_kimi_k26_cut_model.py'):\n"
            "    p=Path(sys.argv[sys.argv.index('--output-path')+1]); p.mkdir(exist_ok=True)\n"
            "    (p/'config.json').write_text(json.dumps({'text_config':{'num_hidden_layers':2}}))\n"
            'else:\n'
            f'    os.execv({sys.executable!r},[{sys.executable!r}]+sys.argv[1:])\n')
        runner.chmod(0o755)
        self.env = dict(PATH=os.environ['PATH'], PYTHONDONTWRITEBYTECODE='1',
                        PYTHON_BIN=str(runner), MODEL_PATH=str(model),
                        DATA_ROOT=str(self.root / 'data with spaces'),
                        VERL_ROOT=str(self.root / 'verl'), LOG_DIR=str(self.root / 'logs'),
                        OUTPUT_DIR=str(self.root / 'output'), CAPTURE=str(self.root / 'capture.json'))

    def launch(self, name, overrides=None, args=()):
        result = subprocess.run(['bash', str(self.recipe / name), *args],
                                env={**self.env, **(overrides or {})}, cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        raw = json.loads(Path(self.env['CAPTURE']).read_text())
        ray_keys = [arg.lstrip('+').split('=', 1)[0] for arg in raw
                    if arg.lstrip('+').startswith('ray_kwargs.')]
        self.assertEqual(len(ray_keys), len(set(ray_keys)), 'duplicate Ray environment overrides')
        return dict(arg.lstrip('+').split('=', 1) for arg in raw)

    def test_common_overrides_reach_every_profile_and_remote_workers(self):
        values = dict(HCCL_HOST_SOCKET_PORT_RANGE='60100-60200',
                      HCCL_NPU_SOCKET_PORT_RANGE='61100-61200', HCCL_BUFFSIZE='64',
                      GROUP_HCCL_BUFFER_SIZE='96', HF_HOME='/shared/hf cache',
                      PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:128',
                      OMP_NUM_THREADS='3', OPTIMIZER_CPU_OFFLOAD='False',
                      OPTIMIZER_OFFLOAD_FRACTION='0.5', CHECKPOINT_SAVE_CONTENTS='[model,hf_model]',
                      LOGGER='["console"]')
        for profile in ('kimi_k26_mopd.sh', 'kimi_k26_opd.sh', 'kimi_k26_mopd_reverse_kl.sh',
                        'kimi_k26_mopd_shared_teacher.sh', 'kimi_k26_rl.sh',
                        'kimi_k26_rl_8npu_cutlayer.sh', 'kimi_k26_mopd_16npu_cutlayer.sh'):
            # The fake crop converter changes the checkpoint depth; restore it for the next full profile.
            (Path(self.env['MODEL_PATH']) / 'config.json').write_text(
                json.dumps({'text_config': {'num_hidden_layers': 61}}))
            with self.subTest(profile=profile):
                config = self.launch(profile, values)
                for key in ('HCCL_HOST_SOCKET_PORT_RANGE','HCCL_NPU_SOCKET_PORT_RANGE',
                            'HCCL_BUFFSIZE','GROUP_HCCL_BUFFER_SIZE','HF_HOME',
                            'PYTORCH_NPU_ALLOC_CONF','OMP_NUM_THREADS'):
                    self.assertEqual(config['ray_kwargs.ray_init.runtime_env.env_vars.' + key],
                                     "'" + values[key] + "'")
                self.assertEqual(config['actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload'], 'False')
                self.assertEqual(config['actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction'], '0.5')
                self.assertEqual(config['actor_rollout_ref.actor.checkpoint.save_contents'], '[model,hf_model]')
                self.assertEqual(config['actor_rollout_ref.actor.optim.lr_warmup_steps_ratio'],
                                 '0' if 'cutlayer' in profile else '0.03')

    def test_distillation_profiles_keep_their_teacher_and_loss_policy(self):
        base = self.launch('kimi_k26_mopd.sh')
        reverse = self.launch('kimi_k26_mopd_reverse_kl.sh')
        shared = self.launch('kimi_k26_mopd_shared_teacher.sh')
        single = self.launch('kimi_k26_opd.sh')
        loss = 'distillation.distillation_loss.'
        self.assertEqual(base[loss+'use_task_rewards'], 'True')
        self.assertEqual(base[loss+'use_policy_gradient'], 'True')
        for config in (reverse, shared, single):
            self.assertEqual(config[loss+'use_task_rewards'], 'False')
            self.assertEqual(config[loss+'use_policy_gradient'], 'False')
            self.assertEqual(config[loss+'loss_max_clamp'], '10.0')
        self.assertEqual(single[loss+'loss_mode'], 'forward_kl_topk')
        self.assertEqual(single[loss+'distillation_loss_coef'], '1.0')
        single_names = {key.split('.')[2] for key in single if key.startswith('distillation.teacher_models.')}
        self.assertEqual(single_names, {'teacher_model'})
        self.assertEqual(single['distillation.teacher_models.teacher_model.num_replicas'], '0')
        self.assertEqual(single['distillation.teacher_models.teacher_model.inference.enable_chunked_prefill'], 'True')
        for config in (reverse, shared):
            self.assertEqual(config[loss+'loss_mode'], 'reverse_kl_full_vocab')
            self.assertEqual(config['distillation.teacher_models.math.key'], 'k26_rl')
            self.assertEqual(config['distillation.teacher_models.gsm8k.key'], 'openai/gsm8k')
        self.assertEqual(reverse['distillation.share_gpu_group'], 'False')
        self.assertEqual(shared['distillation.share_gpu_group'], 'True')
        self.assertEqual(shared['distillation.max_awake_teachers'], '1')
        self.assertEqual(shared['distillation.teacher_models.math.num_replicas'], '2')
        self.assertEqual(shared['distillation.teacher_models.gsm8k.num_replicas'], '2')
        self.assertEqual(shared['trainer.total_training_steps'], '30')
        self.assertEqual(reverse['trainer.total_training_steps'], '1145')

    def test_explicit_paths_teacher_geometry_and_final_cli_overrides(self):
        config = self.launch('kimi_k26_mopd_reverse_kl.sh',
                             dict(TRAIN_FILES='["/shared/train one.parquet"]',
                                  VAL_FILES='["/shared/valid one.parquet"]',
                                  REWARD_FN_PATH='/shared/reward.py',
                                  GSM8K_TEACHER_MODEL='/shared/teacher one',
                                  GSM8K_TEACHER_DP='2', GSM8K_TEACHER_EP='',
                                  ROLLOUT_EP='', USE_POLICY_GRADIENT='True'),
                             args=('trainer.save_freq=7',))
        self.assertEqual(config['data.train_files'], '["/shared/train one.parquet"]')
        self.assertEqual(config['data.val_files'], '["/shared/valid one.parquet"]')
        self.assertEqual(config['reward.custom_reward_function.path'], '/shared/reward.py')
        self.assertEqual(config['distillation.teacher_models.gsm8k.model_path'], '/shared/teacher one')
        self.assertEqual(config['distillation.teacher_models.gsm8k.inference.data_parallel_size'], '2')
        self.assertNotIn('distillation.teacher_models.gsm8k.inference.expert_parallel_size', config)
        self.assertNotIn('actor_rollout_ref.rollout.expert_parallel_size', config)
        self.assertEqual(config['distillation.distillation_loss.use_policy_gradient'], 'True')
        self.assertEqual(config['trainer.save_freq'], '7')


if __name__ == '__main__':
    unittest.main()
