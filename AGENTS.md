# AGENTS.md — 工作区原则与规范

## 项目背景

本工作区主要混合了以下三个项目，聚焦于**昇腾硬件（Ascend）下TTS、ASR等语音类多模态模型的加速和优化研究**：

| 项目 | 说明 |
|------|------|
| `vllm/` | 上游 vLLM 推理加速框架 |
| `vllm-ascend/` | vLLM 的 Ascend 适配插件 |
| `vllm-omni/` | vLLM的Omni多模态推理框架 |

同时还有一些辅助仓库，大多是数据集和模型仓库。

另外，工作区的根目录还存在一些独立脚本，它们实际参与部署和测试。如：

- `run_abse.sh`是跑CPA的benchmark入口
- `run_prse.sh`是跑PG的benchmark入口
- `serve.sh`是服务化入口
- `simple_warmup.sh`运行一个基本的预热
- `test_no_pf.sh`运行一个简单的时间性能测试，它只有一条定长的英文prompt，且不打profilling
- `test.sh`运行一个简单的时间性能测试，它只有一条定长的英文prompt，且打profilling
- `tts.py`是简单时间性能测试的实际py文件
- `vllm-omni/abse.py`是benchmark的实际py文件

目前，我们关注于部署在`vllm-omni`上的Qwen3-TTS模型，并考虑针对它的下述优化:

1. NPU prefix graph。它主要在Qwen3-TTS的MTP模块，code predictor中实现了NPU路径上的prefix graph。我们已经知道code predictor实际上是因为RVQ上的码本补全需要Q次的AR，而re-prefill结构的实现导致每步推理都需要输入seq=Q长度的token（剩余部分用pad填充）；但是实际上每次只需要replay 1, 2, ... Q即可。本特性就是希望在NPU分支上实现这一点。这个特性简写做`PG`。
2. VRVQ Code Predictor Accelerator。它也是在Qwen3-TTS的MTP模块，code predictor由于RVQ特性，后面预测出来的codec实际上是前面codec的残差，因此对code2wav作用比较小，可以使用一些手段在第k步时不再做AR，后续的codec使用一个较轻量的模型来拟合。这个特性简写做`CPA`。
3. Async Schedule。主要是计划强化现有vllm-omni的异步调度特性，让其掩盖程度更高。这个特性简写做`AS`。

我们后续一切研究和讨论都需落实在这三个特性之一上。

约定服务化配置如下：

```bash
vllm serve ./Qwen3-TTS-12Hz-1.7B-Base \
  --omni \
  --port 18091 \
  --allowed-local-media-path / \
  --deploy-config ./tts.yaml \
```

其中的tts.yaml在当前工作区的根目录下。



## 环境约束

- **不得执行任何直接测试命令**（如 `pytest`、`python -m ...`、benchmark 脚本等）（用户明确使用【绕过限制】作为指令前缀除外）。
- **语法检查是允许的**，例如 `python -m py_compile`、`python -c "import ast; ..."` 等纯静态分析命令。
- 代码修改应在理解上下文的基础上进行，避免在不清楚上下游影响时做跨越式改动。

## 代码修改规范

### 1. 最小修改原则（Minimal Change）

- 修改范围应精确锚定目标，**避免引入重构性质的改动**。
- 不随意重组文件结构、不批量重命名、不进行与当前任务无关的格式调整。
- 若必须修改变量名或函数签名，只修改与问题直接相关的部分。

### 2. Python 注释规范

- 在函数/方法定义下方使用 `"""..."""` 形式的 docstring，说明：
  - 方法用途
  - 关键参数（输入）
  - 返回值（输出）
  - 抛出的异常（如适用）
- 遵循 Google 或 NumPy 风格的 docstring 均可，但同一文件内保持一致。
- 不要在代码中引入 `TODO:` 而没有对应的 issue 编号或说明。

示例：
```python
def foo(bar: int, baz: str) -> bool:
    """判断 bar 与 baz 是否满足某种条件。

    Args:
        bar: 整数输入。
        baz: 字符串输入。

    Returns:
        True 当条件满足时，否则 False。
    """
    ...
```

### 3. 代码审查要点

- 修改前先阅读目标文件及其 import 依赖，确保理解现有逻辑。
- 改动后检查是否影响其他调用方。
- 跨项目（如 vllm ↔ vllm-ascend）的修改需特别标注影响面。

## 禁止事项

- ❌ 不得运行任何实际测试、benchmark 或需要昇腾硬件的命令（用户明确使用【绕过限制】作为指令前缀除外）。
- ❌ 不得对工作区进行大规模重构或格式化。
- ❌ 不得在未确认的情况下删除或覆盖已有文件。
- ❌ 不得将当前环境视为可执行真实推理或训练的环境（用户明确使用【绕过限制】作为指令前缀除外）。
