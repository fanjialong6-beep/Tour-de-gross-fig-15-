# QEC Logical Measurement Simulation

## 模拟如何实现

1. **构造编码与初态。** 建立 Gross 码的稳定子和逻辑算符，按照论文逻辑编号选择测量目标；保留逻辑自由度与理想参考比特纠缠，以检查这些自由度是否受到破坏。
2. **构造测量电路。** 建立 LPU 图及相应检查，执行 10 轮带噪声的变形测量，随后读出 LPU 边比特，执行 Return 原码检查并恢复到原码。单 X 使用每轮 12 步的着色排程和理想 Return；其余模式使用带噪声 Return。末端理想稳定子检查用于封闭综合征时间边界。
3. **加入物理噪声。** 使用独立 Pauli 故障事件，每个基本事件概率为 $q=p/15$。双比特门后加入 15 种非平凡 Pauli 事件；空闲步加入 X、Y、Z 各 5 个事件；准备与测量采用 15 个相应的翻转事件。这是独立事件模型，允许多个事件同时发生；它与“概率 p 时仅选择一种 Pauli 错误”的互斥信道只在一阶近似上一致。
4. **生成故障矩阵。** Stim 将电路转换为探测器错误模型。$H$ 将故障映射为综合征，$A$ 将故障映射为报告及逻辑评分量。只有同时具有相同 H 列和 A 列的事件才合并，并保存其重数 $m$。
5. **抽样和译码。** 对有效独立故障数 $N_{\mathrm{eff}}$ 抽取 $w\sim\mathrm{Binomial}(N_{\mathrm{eff}},p/15)$，均匀选择 w 个不同故障位置，再对合并组取奇偶。Relay 接收 H、综合征和目标 p 下的组先验 $[1-(1-2p/15)^m]/2$，返回候选纠正。
6. **统计失败。** 独立计算纠正后的综合征与 A 残差，记录完整指令是否失败。

对于真实故障向量 e、候选纠正 c，完整指令失败定义为：

$$
Hc\ne He\quad\text{或}\quad A(e\oplus c)\ne 0.
$$

Relay-BP-5 参数固定在 `qecsim/decoder.py`：

```text
alpha = 1.0
gamma0 = 0.125
pre_iter = 80
num_sets = 600
set_max_iter = 60
gamma_dist_interval = (-0.24, 0.66)
stop_nconv = 5
```

复现实验使用 Python 3.11，在仓库根目录依次执行：

```bash
python -m pip install -r requirements.txt
python build_models.py
python run.py --config configs/single_x.json --output runs/single_x
python run.py --config configs/batch_a.json --output runs/batch_a
python run.py --config configs/batch_b.json --output runs/batch_b
```

`build_models.py` 从源码及四份排程 JSON 生成 `models/`，无需下载预生成模型。三个配置指定实验的模式、噪声点、样本量和种子；`single_x.json` 为 p = 0.003 下的 40,000 次单 X，随机种子为 202610251203，`batch_a.json` 和 `batch_b.json` 分别为 21,525 和 64,575 次多模式实验。

运行结果写入各输出目录的 `summary.json` 和 `summary.csv`，主指标为 `complete_instruction`。同一模式、同一 p 的两批结果按 `(failures_a + failures_b) / (shots_a + shots_b)` 合并。重复相同运行命令会继续缺失样本；可选 `--workers 4` 使用四个普通工作进程。

## 已完成的模拟结果

下表为完整指令错误率，单位为百分比；`±` 表示一个二项标准误（1σ）。XX、concurrent、XZ 合并两批相同模拟设置的独立样本，共 86,100 次；单 X 使用 12 步排程，共 40,000 次。合并采用总失败数除以总样本数。

| p | X single | XX joint | concurrent | XZ joint |
|---:|---:|---:|---:|---:|
| 0.0030 | 0.1350% ± 0.0184% | 0.944% ± 0.065% | 0.628% ± 0.053% | 0.601% ± 0.052% |
| 0.0035 | — | 5.638% ± 0.311% | 3.934% ± 0.262% | 3.716% ± 0.255% |
| 0.0040 | — | 21.875% ± 2.110% | 19.531% ± 2.023% | 22.135% ± 2.119% |
| 0.0045 | — | 49.510% ± 3.501% | 45.098% ± 3.484% | 47.059% ± 3.495% |
| 0.0050 | — | 85.000% ± 3.018% | 75.000% ± 3.660% | 76.429% ± 3.587% |

| p | XX / concurrent / XZ 各自样本数 | 单 X 样本数 |
|---:|---:|---:|
| 0.0030 | 22,456 | 40,000 |
| 0.0035 | 5,516 | — |
| 0.0040 | 384 | — |
| 0.0045 | 204 | — |
| 0.0050 | 140 | — |

单 X 在 p = 0.003 时为 **54 / 40,000 = 0.1350%**。
