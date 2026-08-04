现在在当前工作区代码仓库中基于vllm-omni运行qwen3-tts。warmup之后采集一次TTS请求（其中ref\_audio为一段约10个token的音频，请求的生成音频约30token）的profilling。观察这段profilling的泳道，得到以下事实：

1. vllm整体不断运行一个busy\_loop，当其中处理请求的基本单元是`_process_engine_step`。观察每个step，除了第一个step以外，后续steps的内部执行的步骤逻辑都遵循相同的模式：首先执行`ar_scheduler`，然后执行模型（主要部分），最后`sample_token`。
   - 其中执行模型阶段又可分为profile\_memory和npu的worker，其中npu的worker按顺序执行了`_prepare_inputs`、`_preprocess`、`_model_forward`和`compute_logits`。
   - `_preprocess`的本质是执行一个`flush_decode_batch`和Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM，后者交由CodePredictorWrapper实际forward。
   - `_model_forward`则执行了一段acl流同步，然后执行replay，然后`update_full_graph_params`，最后`make_omni_output`。
   - 从npu泳道看，整个step有两段非eager：CodePredictor的大部分时间段、\_model\_forward的replay触发之后的一段时间。
   - CodePredictor的持续时间很长，以至于产生了同步。
2. 第一个step表现与其他step有一些不同，不同主要是在`_model_forward`和`_preprocess`中：
   - `_preprocess`主要包括`maybe_run_batch_preprocess`和`build_prompt_embeds`
   - `_model_forward`则调用了Qwen3TTSTalkerForConditionalGeneration的forward。forward运行了若干个fx\_run\_eagerly（对应npu\_fx\_compiler\_inference）。
   - 从NPU泳道看，整个step都是eager的算子下发。

根据观察profilling得到的事实、给出的启动命令、配置yaml、输入样本规格以及工作区代码，综合分析这些材料，考虑以下问题：

1. 什么决定了处理这个请求的实际step的数量？
2. 典型的step（也就是除了第一个step之外的step）其中的若干个阶段分别对应到Qwen3-TTS模型在vllm中推理的哪些阶段？
3. 典型的step中非eager部分来自ACLGraph（或者称之为NPUGraph）。哪些配置和机制使能了这部分非eager？
4. 典型的step中哪些部分仍是eager？为什么？
5. 为什么典型的step中CodePredictor需要花费的时间如此长？
6. 为什么第一个step表现出来与其他step的不一致？尤其注意到它似乎缺失了CodePredictor，为什么？为什么在该step中Talker是eager的？这个step具体在做什么？

