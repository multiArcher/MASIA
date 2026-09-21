# GLIDE 实验配置

稳定基线：`42fb8eaf`。以下覆盖项追加到已有训练入口末尾；实际 map、seed、预算、层数和宽度应与所比较的 GLIDE run 一致。

```bash
bash scripts/train_scripts/GLIDE-N.sh 1 GLIDE_N1_K2_seed1 \
  env_args.map_name=8m_vs_9m bcrbc_generation_horizon=1 bcrbc_flow_steps=2 \
  env_args.delay_during_training=False
```

| 实验 | 追加/替换的参数 |
|---|---|
| GLIDE-0（纯 masked representation） | `bcrbc_generation_horizon=0 bcrbc_flow_steps=0` |
| GLIDE-1 | `bcrbc_generation_horizon=1 bcrbc_flow_steps=2` |
| GLIDE-2 | `bcrbc_generation_horizon=2 bcrbc_flow_steps=2` |
| GLIDE-4 | `bcrbc_generation_horizon=4 bcrbc_flow_steps=2` |
| GLIDE-Full（保留窗口内） | `bcrbc_generation_horizon=-1 bcrbc_flow_steps=2` |
| K 扫描 | 固定 `bcrbc_generation_horizon=1`，使用 `bcrbc_flow_steps=1` 或 `4` |
| Per-step encoder | `bcrbc_encoder_context_window=1`；其他配置保持参考 run 不变 |
| 默认因果 encoder | `bcrbc_encoder_context_window=null`（继承 `bcrbc_context_window`） |

`N=0` 自动禁用 generation auxiliary losses；不要把 `K=0,N=1` 当成完全相同的消融。Full 受实际 context window 和序列起始长度限制。

## 编码器窗口

只将 `bcrbc_encoder_context_window` 设为 1，不将全局 `bcrbc_context_window` 设为 1。新选项不改变 decoder、completion transformer 的窗口或 N 的上限。默认 null 与稳定版本行为一致；旧配置缺少此字段时也继承全局窗口。

该实验必须重新训练。保持实际参考 run 的 N/K、depth/time-block interval、batch、seed 和训练预算相同。

## 训练期延迟

`delayed_sc2` 已支持 `env_args.delay_during_training=True`，并使用 `env_args.delay_type`、`delay_mean`、`delay_std`、`max_delay`。这只配置当前仓库的完整 observation 延迟 wrapper，不会自动复现 RDC 原始特征级延迟接口，也不等于配置了 RDC 的训练流程。

GLIDE 的主实验保留 `env_args.delay_during_training=False`。评估分布由独立评估入口指定。

## 结构化掩码

burst 和每个决策时刻的 recent-suffix 实验放在 `experiment/glide-structured-masks` 分支，见该分支的本文件。不要通过当前 main 的 episode 末尾遮挡代替 decision-relative suffix。

## 实验控制

- 先运行 `N=0` 与 `N=1,K=2`；N 扫描固定 K，无需完整 N×K 矩阵。
- mask 与 encoder 消融一次改变一个因素，使用相同 seed 对照。
- 训练入口保存 commit 和 diff；评估保留 checkpoint、训练 seed、实际 episode 数及延迟采样/截断参数。
- 本文档仅提供配置；未启动训练或提交集群作业。
