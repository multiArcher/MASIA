# CACOM 接入说明

来源：[原作者 LXXXXR/CACOM](https://github.com/LXXXXR/CACOM)，固定版本
`97493a0b2c402e88a06d4e0d21327c41bbd21709`，Apache-2.0。
论文：Xinran Li、Jun Zhang，*Context-aware Communication for Multi-agent
Reinforcement Learning*，AAMAS 2024。

原仓库基于 **PyMARL**。本次迁移的是作者公开的 **CACOM + QMIX** 实现；
不把论文提及的 actor-critic 版本当作已接入功能。

## 文件与接口

- `src/modules/agents/cacom_agent.py`：实体特征编码、请求广播、按接收者生成响应、
  注意力融合、GRU、辅助 Q 值预测、ExpGate 和 LSQ。辅助类都在同一文件内。
- `src/controllers/cacom_controller.py`：继承现有 BasicMAC 的输入构建与隐藏状态初始化，
  适配 rollout、gate 训练、门控保存和目标网络同步。
- `src/learners/cacom_learner.py`：一步 Double-Q/QMIX、辅助损失、单独 RMSprop 门控优化器。
- `src/config/algs/cacom.yaml`：配置入口；三个现有 Maker 分别注册 agent、MAC、learner。

没有修改 `src/main.py`、`src/run.py`、runner、EpisodeBatch、基础 controller/learner、
环境实现，也没有新增 CACOM 专用功能目录。环境数据继续沿用 runner → EpisodeBatch → learner。

## 保留的算法机制

1. 按观测实体分段线性编码，经自注意力和残差 MLP 得到局部特征。
2. 根据上一步 GRU 隐状态和当前特征生成请求，量化后广播。
3. 发送者用各接收者的请求查询自身特征，分别生成响应。
4. ExpGate 决定是否发送，去掉自发自收消息，再量化响应。
5. 本地特征与收到的响应经过注意力融合、GRU，输出每个动作的 Q 值。
6. 通过消息预测其他智能体的 Q 向量，使用 detached Q 目标计算辅助 MSE。
7. 门控训练随机选择一个发送者，比较其全部发送和全部不发送时的最大 Q 值差，
   用交叉熵训练 gate。沿用作者对未做动作屏蔽的 max-Q 的比较。

保留 `pred_weight=0.1`、`discrete_bits=2`、`nn_hidden_multi=4`、
`start_train_gate=200000`、`train_gate_intervel=10000`、`gate_lr=0.0001`。
`train_gate_intervel` 保留作者拼写。普通网络和门控均用 RMSprop，
`target_update_interval=200` 仍以 episode 计数，不采用其他 learner 的更新步计数。
本仓库 `hidden_dim` 对应原作者 `rnn_hidden_dim`。

`encode_dim=8, request_dim=6, response_dim=13` 取自作者 MMM3 示例；
迁移到其他地图时它们是初始设置，并非该地图已调优参数。
`discrete_bits: null` 可关闭量化；启用时要求整数且至少为 2。
沿用作者对称 LSQ，2 bit 对应整数值 -1、0、1，步长初始化为 1。

作者的 warm-up 逻辑也保留：learner 在线 Q 路径在阈值前全通信，门控标签全为发送；
rollout 和目标网络仍使用当前 gate。`communication/reply_freq` 沿用作者口径，
在删除自环之前统计，包括对角线，不等同于实际网络流量。

## 明确的迁移修正

- 从当前 SMAC/SMACv2 `env_info.obs_components` 自动获取分段，避免硬编码地图尺寸；
  明确传入 `obs_segs` 时保留原作者“整个 agent 输入”的语义。
- 适配 `nn.Module`、`.train()`、`.eval()`、`.to(device)`；主优化器仅包含 agent 和 mixer，
  gate 单独优化，避免注册为子模块后被重复优化。
- 目标同步同时覆盖 agent、gate、量化步长及 mixer；作者的 `load_state` 只同步 agent。
- 保存并恢复门控优化器，加载后同步目标 mixer/gate。旧检查点无 `gate_opt.th` 时允许加载，
  但 gate 优化器从空状态开始。继承仓库启动流程，恢复模型并不恢复 runner 全局计数，
  因而不是严格续跑原实验时间轴。
- 辅助预测和门控损失使用 `filled`/`terminated` mask，并排除序列末尾 bootstrap-only
  状态；原代码在全序列上直接平均。这会改变变长回合上的训练权重，属于明确的修正。
- 目标计算关闭梯度，补齐门控 loss 日志，训练结束释放 controller 的计算图引用。

因此，核心网络计算与作者对照一致，但修正后的完整训练轨迹不保证与原仓库逐步一致。
当前仅接受公共团队奖励，默认不标准化奖励或回报；支持 QMIX、VDN 和无 mixer。

## 运行

在仓库根目录、已配置 PyTorch 与 SMAC 的环境中：

```powershell
$env:SC2PATH = 'E:\StarCraft2\StarCraft II'
python src/main.py --config=cacom --env-config=sc2 with env_args.map_name=3m seed=0
```

Linux/WSL 可复用启动脚本（SC2PATH 按本机设置）：

```bash
bash scripts/train_scripts/cacom.sh 3m seed=0 t_max=2050000
```

SMAC/SMACv2 默认自动分段。其他环境须显式指定 `obs_segs`：每项为 `[实体数量, 每实体维度]`，
所有分段的尺寸之和必须等于 obs + last action（启用时）+ agent ID（启用时）。
顺序必须与实际输入一致，不能只填原始 obs。

本仓库 MPE2 `simple_spread_v3` 默认 3 个智能体，obs=18、actions=5。
其观测顺序为自身速度、自身位置、3 个地标相对位置、2 个同伴相对位置、2 个同伴通信向量。
下面的分段是本次环境适配，不是作者提供的 MPE 实验配置：

```powershell
python src/main.py --config=cacom --env-config=mpe with seed=0 'obs_segs=[[1,2],[1,2],[3,2],[2,2],[2,2],[1,5],[1,3]]'
```

SMAC 可追加 `runner=parallel batch_size_run=8` 使用现有并行 runner（本次未实测并行 SMAC）。
本仓库现有 `run.py` 会把 MPE 强制设置为 episode runner 和 batch_size_run=1；
本次保留该行为。
观测时延沿用 BasicMAC/ObservationDelayModel：
`obs_delay_enabled=True obs_delay_apply_train=True obs_delay_apply_test=True obs_gaussian_delay_mean=1`。
也可沿用仓库现有 delayed 环境/runner，但不要无意中同时打开两层观测时延。
CACOM 本身仍为作者的同步两阶段消息交换；本次未增加消息链路时延模型。

## 验证

```powershell
python -m unittest discover -s tests -p test_cacom.py -v
# 可选：使用独立下载的原作者源码作数值对照。
$env:CACOM_UPSTREAM_PATH = 'C:\path\to\CACOM'
python -m unittest discover -s tests -p test_cacom.py -v
```

测试覆盖分段、动作可用性、观测时延、两阶段门控训练、padding 屏蔽、优化器分离、
目标同步、checkpoint 恢复、关闭量化/辅助损失、VDN/无 mixer、CPU/CUDA 切换。
设置原作者路径时，直接加载未经修改的原文件，以相同权重对照前向值、隐藏状态、
辅助损失、参数梯度和 gate 的反事实计算。

本次在 Windows 上 7 项测试全部通过（包括 CUDA 和原作者数值对照）。
真实 MPE episode runner 完成 35 个环境步，并验证评估和模型保存；
打开固定 1 步观测时延后完成 20 个环境步。此次尝试的 MPE parallel 配置被现有
run.py 自动回退到 episode，不能作为并行环境验证。
真实 SMAC `3m` 自动观测分段、训练、gate 更新、评估、保存链路通过，
累计 113 个环境步、3 个训练回合（t_max=65，按完整回合结束）。
SMACv2 只验证了元数据格式解析，未启动真实 SMACv2 地图。
这些是接口和短训练验证，不代表论文收敛结果。

本次详细运行输出保存在本地 `results/cacom_validation/`；Sacred 原始配置和指标、
checkpoint 仍在仓库默认 `results/sacred/` 和 `results/models/` 下。
