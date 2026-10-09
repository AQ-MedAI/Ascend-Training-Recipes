# 验证范围

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| V4.1 CPU 回归 | 69 项通过 | checkpoint、Engram、导出与同步契约，CPU Gloo/ZMQ/共享内存 |
| V4 CPU 检查 | 17 项通过 | 教师状态管理、单 rank KL 前向/梯度 |
| K2.6 CPU 检查 | 17 项通过 | 教师状态管理、单 rank KL 前向/梯度 |
| V4/K2.6 RL 入口 | 参数检查通过 | 以记录器替代训练进程，不启动模型 |

KL 检查提取实际实现，用 torch.log_softmax 替代 TP normalizer，不覆盖完整框架、多 rank KL 或教师 RPC。V4.1 回归还依赖尚未独立分发的测试和运行时组件，当前仓库不提供该套件的完整复跑入口。

## 可复跑的检查

使用 Python 3.10+，CPU 检查需要 torch 和 pytest，Gloo 需要本机通信权限。假设本仓库与 verl、mbridge clone 平级，也可传入实际目录。

```bash
python tools/check_distillation_cpu.py ../verl /tmp/v4-cpu.json --ref e001090e88532bda2d4264bb806bc914fe7c6822
python tools/check_distillation_cpu.py ../verl /tmp/k26-cpu.json --ref be9a6dc3a01cadea0035fcb3e105eb665209d05e
python tools/check_launchers.py /tmp/launchers.json --mbridge-root ../mbridge
```

提交读取方式不改变当前分支。启动参数检查使用临时目录和 Python 记录器，mbridge 目录应检出 [登记版本](DEPENDENCIES.md)。

## NPU 验收待完成

原生算子构建、模型前后向、多节点权重同步、真实 rollout、完整 checkpoint 保存恢复、资源峰值和训练数值行为尚待验证。全层、裁层、OPD、双教师与共享教师配置分别验收。
