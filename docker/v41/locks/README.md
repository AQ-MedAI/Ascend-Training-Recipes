# Python dependencies

训练和 rollout 共用 Python 环境。目标为 Linux aarch64、Ubuntu 22.04、Python 3.12。

## 固定部署版本

| 包 | 版本 |
| --- | --- |
| transformers | 5.14.1 |
| numpy | 1.26.4 |
| fastapi | 0.123.10 |
| starlette | 0.50.0 |
| opencv-python-headless | 5.0.0.93 |
| prometheus-fastapi-instrumentator | 8.1.0 |
| mcp | 2.2.0 |
| model-hosting-container-standards | 0.1.16 |

`runtime-constraints.txt` 固定这些版本。`runtime-overrides.txt` 在 uv 解析时显式覆盖 NumPy、FastAPI、Starlette 的声明要求，不修改第三方包 METADATA。vLLM 的 FastAPI/Starlette 下限、OpenCV 的 NumPy 下限、instrumentator 的 Starlette 下限与这组版本不一致；普通 `pip check` 会报告这些差异。构建及初始化保留报告，但不因报告退出；关键包的实际版本仍须匹配。

## 安装与构建

`build.lock` 已固定 14 个构建工具依赖。完整 `runtime.lock` 尚未生成，但不再阻止开发镜像构建：

- 存在 `runtime.lock`：以 `--require-hashes --no-deps` 安装完整锁定集合，避免再次解析改变已选择的版本。
- 不存在 `runtime.lock`：使用 `runtime.in` 与显式 overrides 正常解析、安装依赖。
- 若普通解析仍不能安装，可在 Docker 构建时传 `--build-arg PYTHON_DEPENDENCY_MODE=no-deps`。此模式只安装列出的直接依赖，不保证传递依赖齐全；安装失败本身仍会终止构建。

```bash
docker build --platform linux/arm64 --target v41 \
  -f docker/v41/Dockerfile -t ascend-training-v41:dev .

# 需要跳过运行依赖解析时：
docker build --platform linux/arm64 --target v41 \
  --build-arg PYTHON_DEPENDENCY_MODE=no-deps \
  -f docker/v41/Dockerfile -t ascend-training-v41:dev .
```

源码框架和 recipe 源码依赖 始终以 `--no-deps --no-build-isolation` 安装。硬件基础层和构建工具仍按各自约束安装，不受上面运行依赖安装模式影响。

## 锁文件生成

```bash
bash docker/v41/locks/compile.sh build
bash docker/v41/locks/compile.sh runtime
```

使用 uv 0.11.6；`UV_BIN` 可指定路径。生成过程完整解析传递依赖、要求目标平台 wheel 并收集哈希，失败不覆盖已有锁。`sources/manifest.json` 记录固定源码 commit 和依赖文件 SHA256。`training.in` / `rollout.in` 是统一输入的兼容别名。

Web/OpenCV 固定组合已完成 Python 3.12 / aarch64 的独立解析及哈希安装 dry-run；不是完整训练锁或实际镜像安装验证。完整锁、Triton Ascend cp312 wheel 哈希和公开镜像构建/NPU 验收仍待完成，不阻止先尝试开发镜像构建。

构建工具与运行依赖共用环境，因此 `packaging==25.0` 与 `setuptools-scm==8.3.1` 一起固定：TransferQueue 0.1.7 和 verl 的约束将 tensordict 限定到 0.10.0，其 pyvers 依赖要求 packaging >=25,<26。新版本 setuptools-scm 的 vcs-versioning 依赖要求 packaging >=26.2，不能混用。
