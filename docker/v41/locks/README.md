# Python dependency locks

目标平台：Linux aarch64、Ubuntu 22.04（glibc 2.35）、CPython 3.12；解析器：uv 0.11.6。

训练和 rollout 共用一个 Python 环境，Transformers 统一为 5.14.1、NumPy 为 1.26.4。镜像不再创建 rollout 虚拟环境。算子的动态库搜索路径仍可按进程角色隔离，这与 Python 依赖隔离无关。

| 文件 | 状态 | 用途 |
| --- | --- | --- |
| `build.in` / `build.lock` | 已解析，15 个包，固定版本和 SHA256 | 构建工具及其传递依赖 |
| `runtime.in` / `runtime-constraints.txt` | 候选，存在源码声明冲突 | 训练、rollout、四个 fork 及源码构建所需依赖的并集 |
| `runtime.lock` | 尚未生成 | 完整运行依赖锁；当前不能完成框架镜像构建 |

`training.in`、`rollout.in` 及对应 constraints 文件仅作为统一输入的兼容别名，不再分别生成锁。`build.lock` 不是完整训练锁；哈希来自公开包索引，解析通过不代表完成镜像构建或 NPU 验收。

## 生成

```bash
bash docker/v41/locks/compile.sh build
# 修正并验证下述源码依赖声明后：
bash docker/v41/locks/compile.sh runtime
```

脚本不安装包，仅解析目标平台的依赖，要求存在 wheel；失败不覆盖已有锁。已有锁作为约束，升级需有意调整版本。`UV_BIN` 可指定 uv 路径。完整解析会联网，直接 URL wheel 没有独立元数据时可能下载完整 wheel。

`sources/` 保留固定源码 commit 的真实依赖声明；`sources/manifest.json` 记录原文件 SHA256。不能只修改快照而不修改构建时使用的源码。更新 fork 后，需要将提交推送、更新 registry 和子模块引用，再同步快照。verl 从 `setup.py` 的 `install_requires` 提取；mbridge 没有必选依赖。vLLM 使用 empty 后端的 `requirements/common.txt`，不引入 CUDA 后端构建要求。

## 完整锁的阻塞项

- 固定 MindSpeed 源码声明 `numpy<=1.26.0`，需更新为目标 `numpy==1.26.4`。
- 固定 MindSpeed-LLM 源码声明 `transformers==5.2.0`，需更新为统一目标 `5.14.1`，完成训练验证后更新引用。
- vLLM 要求 `fastapi>=0.133,<0.137`、`starlette>=1.0.1`；vLLM-Ascend 要求 `fastapi<0.124`。当前候选 FastAPI 0.123.10 / Starlette 0.50.0 不能满足 vLLM 声明。需验证兼容源码或调整框架版本，不能靠 constraints 覆盖。
- 候选 OpenCV 5.0.0.93 在 Python >=3.9 下声明 `numpy>=2`，与 NumPy 1.26.4 冲突。vLLM 的 OpenCV 下限是 4.13.0，已核对的 4.13.0.92 也要求 NumPy 2。需验证替代组合或受控构建方案。
- Megatron 默认 `PY_ENV=pytorch_25.03` 声明 `nvidia-modelopt[torch]>=0.23.2`，需确认 Ascend 训练是否需要该依赖，并在安装配置中处理。
- Triton Ascend cp312 wheel 哈希尚需核验；缺少目标平台 wheel 的包需补充受控构建方案。

以上是已发现的直接约束问题，尚未完成所有传递依赖解析。单环境不消除框架自身的冲突；不得通过禁用依赖检查将候选集合标为已验证锁。

## 验收

镜像构建使用 `runtime.lock`，训练和 rollout 框架都安装到构建时的 `python3`。框架收据登记 `python_environment=single`，包版本记录在 `runtime-environment.json`。启动时 `ROLLOUT_PYTHON` 默认跟随训练解释器；初始化拒绝不同的 Python 环境。

```bash
python3 docker/v41/collect_environment.py > runtime-environment.json
python3 -m pip check
```

完整锁通过后，还必须在目标 NPU 上检查：配置和 tokenizer 加载、训练 forward/backward、optimizer step、权重同步、同步后 rollout、保存恢复和数值变化。Engram 直接使用 AutoTokenizer 且没有固定版本判断，只能作为源码核对结果，不能代替上述运行验证。
