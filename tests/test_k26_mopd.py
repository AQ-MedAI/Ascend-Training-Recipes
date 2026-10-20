"""Exercise MOPD reward routing and launch arguments without NPU dependencies."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/kimi-k2.6"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MopdRewardTest(unittest.TestCase):
    def test_real_scorers_route_dataset_answers_and_reject_unknown_sources(self):
        with patch.dict(sys.modules):
            for name in ("gsm8k", "math_dapo"):
                load_module(
                    f"verl.utils.reward_score.{name}",
                    ROOT / f"third_party/verl/verl/utils/reward_score/{name}.py",
                )
            reward = load_module("k26_multi_reward_test", RECIPE / "multi_reward.py")
            cases = [
                ("DigitalLearningGmbH/MATH-lighteval", r"Final: \boxed{42}", 1.0),
                ("DigitalLearningGmbH/MATH-lighteval", r"Final: \boxed{41}", -1.0),
                ("openai/gsm8k", "Calculation\n#### 42", 1.0),
                ("openai/gsm8k", "Calculation\n#### 41", 0.0),
                ("openai/gsm8k", "No final answer", 0.0),
            ]
            for source, solution, expected in cases:
                with self.subTest(source=source, solution=solution):
                    result = reward.compute_score(source, solution, "42", extra_info={}, router_option=None)
                    self.assertEqual(result["score"] if isinstance(result, dict) else result, expected)
            result = reward.compute_score(
                "DigitalLearningGmbH/MATH-lighteval", "Answer: 42", "42", math_strict_box_verify=False
            )
            self.assertEqual(result["score"], 1.0)
            with self.assertRaisesRegex(ValueError, "Unsupported MOPD reward"):
                reward.compute_score("unexpected", "42", "42")


class MopdLauncherTest(unittest.TestCase):
    def test_mopd_reuses_validated_rl_settings_and_routes_two_teachers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipe"
            recipe.mkdir()
            for name in (
                "kimi_k26_rl.sh",
                "kimi_k26_rl_8npu_cutlayer.sh",
                "kimi_k26_mopd_16npu_cutlayer.sh",
                "kimi_k26_mopd.sh",
                "pipeline_layout.py",
            ):
                shutil.copy2(RECIPE / name, recipe / name)
            (recipe / "runtime_env.sh").write_text(
                "export MEGATRON_ROOT=/stub/core\nexport MEGATRON_BRIDGE_ROOT=/stub/bridge\n"
            )
            runner = root / "python-runner"
            runner.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\nfrom pathlib import Path\n"
                "if sys.argv[1].endswith('/prepare_kimi_k26_cut_model.py'):\n"
                "    path = Path(sys.argv[sys.argv.index('--output-path') + 1])\n"
                "    path.mkdir(parents=True, exist_ok=True)\n"
                "    (path / 'config.json').write_text(json.dumps({'text_config': {'num_hidden_layers': 2}}))\n"
                "elif sys.argv[1:3] == ['-m', 'verl.trainer.main_ppo']:\n"
                "    Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[3:]))\n"
                "else:\n"
                f"    os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n"
            )
            runner.chmod(0o755)
            env = {
                "PATH": os.environ["PATH"],
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHON_BIN": str(runner),
                "MODEL_PATH": str(root / "model"),
                "DATA_ROOT": str(root / "data with spaces"),
                "VERL_ROOT": str(root / "verl"),
                "LOG_DIR": str(root / "logs"),
                "OUTPUT_DIR": str(root / "output"),
                "CAPTURE": str(root / "capture.json"),
            }

            def launch(name, overrides=None):
                result = subprocess.run(
                    ["bash", str(recipe / name)],
                    cwd=root,
                    env={**env, **(overrides or {})},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return dict(arg.lstrip("+").split("=", 1) for arg in json.loads(Path(env["CAPTURE"]).read_text()))

            rl = launch("kimi_k26_rl_8npu_cutlayer.sh")
            mopd = launch("kimi_k26_mopd_16npu_cutlayer.sh")
            for key, value in rl.items():
                if key.startswith(("actor_rollout_ref.", "algorithm.")):
                    self.assertEqual(mopd[key], value, key)
            for key in (
                "data.max_prompt_length",
                "data.max_response_length",
                "data.truncation",
                "trainer.save_freq",
                "trainer.total_training_steps",
                "trainer.n_gpus_per_node",
            ):
                self.assertEqual(mopd[key], rl[key], key)
            self.assertEqual(mopd["distillation.enabled"], "True")
            self.assertEqual(mopd["distillation.n_gpus_per_node"], "8")
            self.assertEqual(mopd["distillation.teacher_models.math.key"], "DigitalLearningGmbH/MATH-lighteval")
            self.assertEqual(mopd["distillation.teacher_models.gsm8k.key"], "openai/gsm8k")
            self.assertEqual(mopd["reward.custom_reward_function.path"], str(recipe / "multi_reward.py"))
            self.assertEqual(
                json.loads(mopd["data.train_files"]),
                [
                    str(root / "data with spaces/math/train.parquet"),
                    str(root / "data with spaces/gsm8k/train.parquet"),
                ],
            )
            changed = launch(
                "kimi_k26_mopd_16npu_cutlayer.sh",
                {
                    "MATH_TRAIN_FILE": str(root / "custom math.parquet"),
                    "GSM8K_TEACHER_MODEL": "/shared/gsm8k-teacher",
                    "REWARD_FN_PATH": "/shared/custom_reward.py",
                },
            )
            self.assertEqual(json.loads(changed["data.train_files"])[0], str(root / "custom math.parquet"))
            self.assertEqual(changed["distillation.teacher_models.gsm8k.model_path"], "/shared/gsm8k-teacher")
            self.assertEqual(changed["reward.custom_reward_function.path"], "/shared/custom_reward.py")

            full = launch("kimi_k26_mopd.sh")
            for key, value in rl.items():
                if key.startswith("algorithm."):
                    self.assertEqual(full[key], value, key)
            self.assertNotIn("algorithm.kl_coef", full)
            self.assertNotIn("algorithm.kl_penalty", full)
            for key in (
                "data.train_files",
                "data.val_files",
                "data.truncation",
                "data.filter_overlong_prompts",
                "reward.custom_reward_function.path",
                "reward.custom_reward_function.name",
                "distillation.teacher_key",
                "distillation.teacher_models.math.key",
                "distillation.teacher_models.gsm8k.key",
                "distillation.distillation_loss.use_task_rewards",
            ):
                self.assertEqual(full[key], mopd[key], key)
            self.assertEqual(full["actor_rollout_ref.model.path"], env["MODEL_PATH"])
            self.assertEqual(full["trainer.nnodes"], "40")
            self.assertEqual(full["trainer.n_gpus_per_node"], "16")
            self.assertEqual(full["actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"], "10")
            self.assertEqual(full["data.max_prompt_length"], "65536")
            self.assertEqual(full["data.max_response_length"], "8192")
            self.assertEqual(full["distillation.distillation_loss.loss_mode"], "reverse_kl_full_vocab")
            self.assertEqual(full["actor_rollout_ref.actor.checkpoint.async_save"], "False")
            self.assertEqual(full["ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_CHECKPOINT_WRITER"], "'inprocess'")
            full_custom = launch(
                "kimi_k26_mopd.sh",
                {
                    "MATH_VAL_FILE": "/shared/custom math.parquet",
                    "MATH_TEACHER_MODEL": "/shared/math-teacher",
                    "GSM8K_TEACHER_MODEL": "/shared/gsm8k-teacher",
                    "REWARD_FN_PATH": "/shared/custom_reward.py",
                },
            )
            self.assertEqual(json.loads(full_custom["data.val_files"])[0], "/shared/custom math.parquet")
            self.assertEqual(full_custom["distillation.teacher_models.math.model_path"], "/shared/math-teacher")
            self.assertEqual(full_custom["distillation.teacher_models.gsm8k.model_path"], "/shared/gsm8k-teacher")
            self.assertEqual(full_custom["reward.custom_reward_function.path"], "/shared/custom_reward.py")


if __name__ == "__main__":
    unittest.main()
