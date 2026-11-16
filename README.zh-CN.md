# lerobot-surgery

[English](README.md) | **中文**

对 [LeRobot](https://github.com/huggingface/lerobot) 数据集做"手术"的命令行工具，补上 lerobot 自己没有提供的操作：
检查数据集、混合前对比多个数据集、删集、按任务拆分、合并、裁掉静止帧、修正标错的帧率。
支持 **LeRobot v2.1**，只依赖 numpy 和 pyarrow，不需要安装 lerobot。

所有操作都不改源数据集，不重新编码视频，并且记录每一集的来历。

## 安装

```bash
git clone https://github.com/heliang-pu/lerobot-surgery && cd lerobot-surgery
pip install -e .              # numpy、pyarrow
pip install -e ".[video]"     # 加上 PyAV：check --videos 和 retime 要用
```

`retime` 还需要系统里有 ffmpeg（4.4 或更新，要有 `setts` 码流过滤器）。
不想安装也可以直接跑：`PYTHONPATH=. python -m lerobot_surgery <命令> ...`

## 命令

| 命令 | 做什么 |
|---|---|
| `inspect <数据集>...` | 每个数据集有什么：帧率、各任务集数、相机、每个向量特征逐维的取值范围；多个数据集之间帧率或特征不一致会报警告 |
| `check <数据集>... [--videos] [--lerobot]` | 文件、行和元数据是否互相对得上 |
| `drop <源> <输出> --drop 3,10-12 --reason "..."` | 去掉一些集，其余从 0 重新编号；`--keep` 则只保留列出的集 |
| `split <源> <输出目录>` | 每个任务拆成一个独立数据集 |
| `merge <数据集或目录>... -o <输出>` | 多个数据集合成一个；提示词相同算同一个任务 |
| `trim <源> <输出>` | 裁掉没有运动的行：末尾的静止帧、中间的停顿、开头的一段 |
| `retime <源> <输出> --fps 10` | 修正帧率标签（比如 10 Hz 采的数据被标成了 25 fps） |

```bash
lerobot-surgery inspect data/sim_pick data/real_pick
lerobot-surgery check data/real_pick --videos --lerobot
lerobot-surgery drop data/real data/real_clean --drop 12,40-43 --reason "画面里有手"
lerobot-surgery split data/ten_tasks data/by_task --name "Pick up the cup=pick_cup"
lerobot-surgery merge data/by_task data/new_batch -o data/merged --repeat-task "Pick up the cup=3"
lerobot-surgery trim data/real data/real_trimmed --cut-pauses 5 --min-length 40 --next-state-action
lerobot-surgery retime data/real25 data/real10 --fps 10 --link-map data/.retime_links.json
```

每个命令的全部参数见 `lerobot-surgery <命令> --help`。

## 所有命令共同的保证

- **不改源数据集。** 输出先建在目标旁边的 `.<名字>.building` 目录里，通过检查后才改名成目标；目标已存在就拒绝。
- **视频不重新编码。** 视频用硬链接（跨文件系统时复制）。`retime` 只改时间戳，画面数据原样复制。
- **每个输出都先过一遍 `check`。** 不通过就报错，并把半成品留在 `.building` 目录里供查看。
- **来历可查。** 每个输出带两个文件：
  - `meta/surgery.json`：做了什么操作、用了什么参数、来源是哪些数据集；来源自己的 `surgery.json` 会放进 `history`。
  - `meta/surgery_episodes.jsonl`：每一集来自哪个数据集的第几集；多次操作后在 `origin` 里一层层保留。
- **其他元数据不丢。** 单个来源时，`meta/` 下带 `episode_index` 的 `.jsonl` 按新编号重写（原编号记在
  `source_episode_index`），其余文件原样复制。多个来源时（`merge`），各自的其他元数据放到 `meta/by_source/<来源目录名>/`。
- **统计量跟着更新。** 逐集统计里的 `episode_index`、`index`、`task_index` 重新计算；行被裁过的集，数值特征的统计重算，
  图像统计沿用来源的。源里有 `meta/stats.json` 时，按逐集统计重新汇总一份。

## 各命令的说明

**check** 核对 `info.json` 里的总数、每集的行数、`episode_index` / `index` / `task_index`、时间戳是否等于
`frame_index / fps`、每个视频是否存在、有没有不属于任何一集的 parquet 或 mp4（lerobot 会把数据集目录下的这类文件都算进去）。
`--videos` 再打开每个视频，和行对比。`--lerobot` 再用 lerobot 实际加载一次并解码最后一帧。

**inspect** 在数据集之间帧率或特征不一致时退出码为 1。它还会给出"提示"：同一维在一个数据集里的取值比另一个大 15 倍以上。
这通常是单位不同（比如夹爪一个用米、一个用 0–1 的比例），但也可能只是某个任务里这个关节几乎不动，所以只是提示，不算错误。

**merge** 的集编号顺序：先按任务，任务内按来源的顺序。`--repeat-task "<提示词或它的前缀>=n"` 和
`--repeat-source <来源目录名>=n` 把每一集连续写 n 次，是原样的副本，用来给没有采样权重的训练代码加权。
`--tasks-from <数据集>` 采用那个数据集的任务顺序。

**trim** 只删行。视频不动，留下的行保留原来的时间戳，所以解码出来的画面仍然对得上。
- 默认裁掉最后一次运动之后的静止帧（保留最后一个运动的位姿）；`--keep-tail` 关掉。
- `--cut-pauses N`：集中间连续 N 步以上的静止只保留第一帧。
  **这样裁过之后行与行不再相隔 1/fps，lerobot 自带的加载器会拒绝这个数据集**（它的时间戳检查）。
  直接按每行时间戳取帧的读取器不受影响。`check` 会对此给出警告。
- `--start-at '{"<集号>": 起始行}'` 从那一行开始保留；值为 `null` 则整集不要。
- `--next-state-action`：被裁位置前一行的动作改成下一个保留行的状态。只用于"动作 = 下一帧状态"的数据。
- "静止"的判断：相邻两行状态各维变化的最大值不超过 `--static-threshold`（默认 0.002）。

**retime** 用于帧率标错的数据：行、帧、画面都不变，时间戳从 `k / 原帧率` 改成 `k / 真实帧率`。
`--link-map` 指定一个跨次运行保留的 JSON 文件，让通过硬链接共用视频的多个数据集在改完之后仍然共用。

## 限制

- 只支持 LeRobot **v2.1**（每集一个 parquet，每个相机每集一个 mp4）。v2.0 请先用 lerobot 的转换脚本升级；v3.0 不支持。
- 每一集只能属于一个任务。

## 测试

```bash
python -m pytest -q
```

测试用临时生成的小数据集（真的 parquet 和 32×24 的真视频，包含有帧重排和没有帧重排两种编码）。
装了 lerobot 时，还会用 lerobot 加载各个操作的结果。
