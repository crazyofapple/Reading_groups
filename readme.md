# **大规模预训练语言模型相关热点方向资源整理**

**计算的力量**： 很多证据表明，机器学习的进步很大程度上是由计算驱动的，而不是研究，请参考："[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)"，而且往往会出现[Emergence和Homogenization](https://arxiv.org/abs/2108.07258)现象。
有研究表明，[人工智能计算使用量大约每3.4个月翻一番，而效率提升每16个月才翻一番](https://openai.com/blog/)。其中计算使用量主要由计算力驱动，而效率则由研究驱动。 
这意味着计算增长在历史上主导了机器学习和其子领域的进步。尽管如此，未来是否有更颠覆Transformer的架构仍需要我们重视。
目前的NLP研究热点大部分基于更先进的LLM （~100B, $10^{23}$ FLOPs）。尤其是ChatGPT通过[Alignment](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/lectures/lec22.pdf)技术利用少于预训练几十倍的计算（4.9+60 petaflops/s-days vs 3640 petaflops/s-days）和人类反馈（500k美元, 20k小时，13+33+31k数据，相比于GPT-3的12000k美元
）释放了GPT大模型对话能力并火出圈。所以本库对大规模预训练语言模型LLM相关文章进行追踪和归类，更能让我们把握前沿，看清方向。

关于LLM更多topics的论文请参考[这里](https://self-supervised.cs.jhu.edu/fa2022/)和[这里](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)。

--- 
**论文** (*粗糙类别*)
- [模型训练和优化](#大模型训练和优化)
- [应用与LLM+](#应用与llm)
- [原理分析](#原理分析)
- [技术改进](#技术改进-如生成技术prompt工程指标可信等)
- [Survey和数据集](#survey和数据集)

**资源**
- [LLM课程](course.md)
- [重要的图](figures.md)
- [LLM Demo](demo.md)
- [重要的博客与自选文章](custom.md)
- 训练，推理，应用工具 (未整理)
---
## **大模型训练和优化**

【InstructGPT论文，包括sft,ppo等，最重要的文章之一】Training language models to follow instructions with human feedback

【scalable oversight: 人类在模型超过自己的任务后怎么持续的提升模型？】Measuring Progress on Scalable Oversight for Large Language Models

- Self-critiquing models for assisting human evaluators
- 定义：以标签、奖励信号或批评的形式向模型提供可靠监督的能力，这种监督在模型开始达到广泛的人类水平表现之后仍将保持有效。
- Scalable oversight技术可以提升模型的容量和对齐（即以人类期待的方式进行应用和实现目标）。
- 如果我们能找到在现有模型（水平在非专家之上，专家之下）的基础上，找到一个监督学习的范式，能够提升模型答案的正确性，那我们不再依赖专家就能获得一个超越专家的系统。
- 另一个角度想法是通过使用多种提示和策略来提示模型，并仅接受模型在一致且合理的证据的基础上一致给出的答案。但这个角度的技术可能扩展性不足。 当然，任何能够以高可靠性解决此类挑战的技术都可能代表可扩展监督方面的重要进展。
- 现有解决方案：让现有模型辅助人类获取知识来让人类产出高质量的监督。

A General Language Assistant as a Laboratory for Alignment

【必读】Improving language models by retrieving from trillions of tokens


【中英文的大模型，超过GPT-3】GLM-130B: An Open Bilingual Pre-trained Model

【预训练目标优化】UL2: Unifying Language Learning Paradigms

【Alignment新的基准，模型库和新方法】Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization

【通过技术不使用[MASK]标记进行MLM】Representation Deficiency in Masked Language Modeling

【文字转为图像训练，缓解了Vocabulary的需要并抗某些攻击】Language Modelling with Pixels

LexMAE: Lexicon-Bottlenecked Pretraining for Large-Scale Retrieval

InCoder: A Generative Model for Code Infilling and Synthesis

【检索Text相关图像进行语言模型预训练】Visually-Augmented Language Modeling

A Non-monotonic Self-terminating Language Model

【通过prompt设计进行负面反馈比较微调】Languages are Rewards: Hindsight Finetuning using Human Feedback

【Sparrow模型】Improving alignment of dialogue agents via targeted human judgements

【用小模型参数加速大模型训练过程（不从头）】Learning to Grow Pretrained Models for Efficient Transformer Training

【多种知识源MoE半参数知识融合模型】Knowledge-in-Context: Towards Knowledgeable Semi-Parametric Language Models

【不同数据集上的多个已训练模型合并方法】Dataless Knowledge Fusion by Merging Weights of Language Models

【很有启发，检索机制代替 Transformer 中的 FFN 的通用架构(×2.54 time)，以便解耦存储在模型参数中的知识】Language model with Plug-in Knowldge Memory

【自动生成Instruction tuning的数据用于GPT-3的训练】Self-Instruct: Aligning Language Model with Self Generated Instructions

- 【和yizhong wang那篇类似自动生成Instruction的数据，面向T0】Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor
- Language model acceptability judgements are not always robust to context
- SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks
- (FLAN-T5-CoT) 【CoT微调】Scaling Instruction-Finetuned Language Models  

-![image](https://user-images.githubusercontent.com/3351073/214972577-e9fb27a6-55a8-40b6-b38f-cbb68a115739.png)

Towards Conditionally Dependent Masked Language Models

【迭代地校准不完美生成的独立校正器，Sean Welleck的后续文章】Generating Sequences by Learning to Self-Correct


- 预测：AI反馈很快会取代人工用户反馈用于模型更新 
- Towards Boosting the Open-Domain Chatbot with Human Feedback
- 类似想法 1. Constitutional AI: Harmlessness from AI Feedback 
- 类似想法 2. Discovering Language Model Behaviors with Model-Written Evaluations
- 应用：【OpenAI】Recursively Summarizing Books with Human Feedback

【持续学习：新任务增加一个prompt，且上一个任务的prompt和大模型不变】Progressive Prompts: Continual Learning for Language Models without Forgetting

【EMNLP 2022，模型的持续更新】MemPrompt: Memory-assisted Prompt Editing with User Feedback


【新的神经架构 (FOLNet)，其中包含一阶逻辑归纳偏差】Learning Language Representations with Logical Inductive Bias

GanLM: Encoder-Decoder Pre-training with an Auxiliary Discriminator

【基于state-space models的预训练语言模型，超过BERT】Pretraining Without Attention

【预训练的时候就考虑人类反馈】Pretraining Language Models with Human Preferences

【Meta的开源LLaMA模型，7B-65B，训练比通常使用的更多的标记的小模型，在各种推理预算下实现最佳性能】LLaMA: Open and Efficient Foundation Language Models

## **应用与LLM+**

【应用ICL的多步推理方法，很有启发】ReAct: Synergizing Reasoning and Acting in Language Models

 - 【单独使用LLM不足以创建真正强大的APP，将LLM与其他计算或知识来源相结合时，真正的力量才会出现】
- 【工具】 langchain - Building applications with LLMs through composability
- 【survey】 Augmented Language Models: a Survey
- 类似想法 0. TALM: Tool Augmented Language Models
- 类似想法 1. DEMONSTRATE–SEARCH–PREDICT:Composing retrieval and language models for knowledge-intensive NLP
- 类似想法 2. LAMBADA: Backward Chaining for Automated Reasoning in Natural Language
- 类似想法 3.【选择和推理】Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning
- 类似想法 4. Language Models as Agent Models
- 类似想法 5. Prompting Is Programming: A Query Language For Large Language Models
- 类似想法 6.【Neurips 22'】Language Model Cascades 

【CoT直接生成program code，然后让python interpreter执行】Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks

- 相关文章：【EMNLP 22'】Language Models of Code are Few-Shot Commonsense Learners
- 【Heng Ji组】Code4Struct: Code Generation for Few-Shot Structured Prediction from Natural Language
PAL: Program-aided Language Models
- 【Qing Lyu, Chris Callison-Burch组】Faithful Chain-of-Thought Reasoning

【大模型直接产生证据上下文】Generate rather than Retrieve: Large Language Models are Strong Context Generators

【具有4个特定操作的写作模型】PEER: A Collaborative Language Model

【将Python、SQL执行器和大模型结合】Binding Language Models in Symbolic Languages

【检索文档生成代码】DocPrompting: Generating Code by Retrieving the Docs

【Grounding+LLM的系列文章接下来会有很多】LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models

- Do As I Can, Not As I Say: Grounding Language in Robotic Affordances  
https://say-can.github.io/

【自我迭代生成（利用python验证过）训练数据】Language Models Can Teach Themselves to Program Better


- 相关文章：Specializing Smaller Language Models towards Multi-Step Reasoning
- STaR: Bootstrapping Reasoning With Reasoning, 来自Neurips 22 (生成CoT数据用于模型微调）, 引起后续一系列教小模型的CoT的文章
- 类似想法 【知识蒸馏】 Teaching Small Language Models to Reason 与 Learning by Distilling Context 

- 类似想法 KAIST和Xiang Ren组（【CoT的rationale微调（教授）时进行扰动】PINTO: Faithful Language Reasoning Using Prompt-Generated Rationales等） 与 Large Language Models Are Reasoning Teachers 
- ETH的【CoT的数据分别训练问题分解和问题解答模型】Distilling Multi-Step Reasoning Capabilites of Large Language Models into Smaller Models via Semantic Decompositions

【让小模型学会CoT能力】In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models

【大模型教小模型CoT】Large Language Models Are Reasoning Teachers

【大模型生成证据（背诵）然后进行小样本闭卷问答】Recitation-Augmented Language Models

【归纳推理的自然语言方式】Language Models as Inductive Reasoners

【GPT-3用于数据标注（如情感分类）】Is GPT-3 a Good Data Annotator?

【基于多任务训练用于少样本数据增强的模型】KnowDA: All-in-One Knowledge Mixture Model for Data Augmentation in Low-Resource NLP

【procedural planning的工作，暂时不感兴趣】Neuro-Symbolic Procedural Planning with Commonsense Prompting

【目标：为维基百科中某些参考文献支持的Query生成一篇事实正确的文章】WebBrain: Learning to Generate Factually Correct Articles for Queries by Grounding on Large Web Corpus

【将外部物理模拟器的结果结合在context中】Mind's Eye: Grounded Language Model Reasoning through Simulation

【检索增强的CoT做知识Intensive的任务】Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions

【对比一致搜索 (CCS)无监督识别语言模型中的潜在（二元）知识】Discovering Latent Knowledge in Language Models Without Supervision

## **原理分析**

【在我看来是最重要的文章之一，语言模型在交叉熵损失下的比例定律，损失与模型大小，数据集大小，用于训练的计算量成幂律关系，而宽度深度等架构细节影响较小】Scaling Laws for Neural Language Models

【另一篇最重要的文章之一，Chinchilla，限定计算下，最优的模型并不是最大的模型，而是更多数据训练的较小模型（60-70B）】Training Compute-Optimal Large Language Models

【哪种架构和优化目标有助于零样本泛化】What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?

【Grokking “顿悟”学习过程 Memorization->Circuit formation->Cleanup】Progress measures for grokking via mechanistic interpretability
 
【调查检索式模型的特点，发现两者均对reasoning有限】Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model

 
- 检索式+LLM的想法是下一步的方向，但是不是唯一的答案还需要看看
- Rethink Search: Making Domain Experts out of Dilettantes
- Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models

【人类-AI语言交互评价框架】Evaluating Human-Language Model Interaction

- 类似文章 Measuring the Human Utility of Free-Text Rationales in Human-AI Collaboration


What learning algorithm is in-context learning? Investigations with linear models

【模型编辑，这块是Hot topic】Mass-Editing Memory in a Transformer

【模型对无关上下文的敏感性，向提示中示例添加不相关的信息和添加忽略不相关上下文的指令部分解决】Large Language Models Can Be Easily Distracted by Irrelevant Context

【zero-shot CoT在敏感问题下会表现出bias和toxicity】 On Second Thought, Let’s Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning

【大模型的CoT具有跨语言能力】Language models are multilingual chain-of-thought reasoners

【不同Prompt序列困惑度越低性能越好】 Demystifying Prompts in Language Models via Perplexity Estimation

【大模型的binary implicature resolution任务，这种暗示难并没有缩放现象】Large language models are not zero-shot communicators （https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/implicatures）

【复杂的提示提升了CoT】Complexity-Based Prompting for Multi-step Reasoning


- 目标：提升CoT自身的效用，与CoT效用的分析息息相关
- 【生成后先单个样例选择后组合选择】Explanation Selection Using Unlabeled Data for In-Context Learning
- 【自动构建CoT中的样例的解释并用于CoT】Automatic Chain of Thought Prompting in Large Language Models
- 【对CoT生成的解释进行二次调整，用带参数的refiner模块+信息熵优化】Explanation Regeneration via Information Bottleneck

What Matters In The Structured Pruning of Generative Language Models?

【AmbiBench数据集，任务歧义：缩放 RLHF 模型在消除歧义任务方面表现最佳。微调比few-shot prompting更有帮助】Task Ambiguity in Humans and Language Models

【GPT-3的测试，包括记忆，校准，偏见等】Prompting GPT-3 To Be Reliable

【OSU研究CoT哪个部分对性能有效】Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters

- 类似想法1 Complementary Explanations for Effective In-Context Learning (UT Austin, Xi Ye, Greg Durrett) 
- 类似想法2 Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango

【离散提示的跨语言模型研究】Can discrete information extraction prompts generalize across language models?

【记忆率与训练中的模型大小、前缀长度和重复率呈对数线性关系】Quantifying Memorization Across Neural Language Models

【很有启发，将问题通过GPT迭代分解为子问题并回答】Measuring and Narrowing the Compositionality Gap in Language Models


- 【研究是否或什么时候分步作答对阅读有效，零样本和低资源有效】When Do Decompositions Help for Machine Reading?
- 类似想法 Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
- 类似想法 Successive Prompting for Decomposing Complex Questions

【对GPT-3类似公务员那种智力题类比测试】Emergent Analogical Reasoning in Large Language Models

【短文本训练，长文本测试，评估模型的变长适应能力】A Length-Extrapolatable Transformer


【什么时候检索，什么时候用大模型足够】When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories

【ICL是另一种形式的gradient更新】Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta Optimizers

- 相关文章：Transformers learn in-context by gradient descent

Is GPT-3 a Psychopath? Evaluating Large Language Models from a Psychological Perspective


【对OPT模型进行不同大小训练的过程研究，发现困惑度是ICL的指标】Training Trajectories of Language Models Across Scales

【EMNLP 2022, 预训练纯英语语料包含着其他语言，模型跨语言能力可能来自于数据泄露】Language Contamination Helps Explains the Cross-lingual Capabilities of English Pretrained Models

## **技术改进** （如生成技术，Prompt工程，指标，可信等）

【个性化风格的prompt学习，OPT】Extensible Prompts for Language Models

【加速大模型解码，利用小模型和大模型直接的共识一次调用多次可用，毕竟输入长了会很慢】 Accelerating Large Language Model Decoding with Speculative Sampling


【语义解析任务，ICL的样例选择方法，CODEX和T5-large】Diverse Demonstrations Improve In-context Compositional Generalization

【一种文本生成的新的优化方式】Tailoring Language Generation Models under Total Variation Distance

【条件生成的不确定性估计，采用多个采样输出的语义聚类合并后簇的熵来估计】Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation 

- 相关文章：1. Language Models (Mostly) Know What They Know
- 相关文章：2. Teaching Models to Express Their Uncertainty in Words
- 相关文章：3. Navigating the Grey Area: Expressions of Overconfidence and Uncertainty in Language Models
- 校准元分析：大模型的校准是否会因为模型的大小，模型的架构，Instruction的不同，Context的不同以及任务领域而发生变化？
- 最优的开放域对话生成的校准方法是什么？如何提升模型的校准性能，微调，RLHF，Instruction tuning？
- 大模型是否真正是因为理解问题而校准，而不是通过统计偏差得到良好的可信度评估？是不是和人类一样存在欺骗行为，知道自己不懂，但是假装自己知道？ 这样如何评价？
- 如果大模型拥有了良好的校准，下一步我们可以做些什么，如何应用于对话生成等应用？

Go-tuning: Improving Zero-shot Learning Abilities of Smaller Language Models

【很有启发，自由文本约束下的文本生成方法】Controllable Text Generation with Language Constraints


【生成预测时采用相似度选phrase而不是softmax预测token】Nonparametric Masked Language Modeling

【长文本的ICL方法】Parallel Context Windows Improve In-Context Learning of Large Language Models

【InstructGPT模型自己生成ICL的样例】Self-Prompting Large Language Models for Open-Domain QA


【通过分组和注意力机制使得ICL能够输入更多的标注样本】Structured Prompting: Scaling In-Context Learning to 1,000 Examples

Momentum Calibration for Text Generation


【两种ICL样例选择的方法，基于OPT和GPTJ的实验】Careful Data Curation Stabilizes In-context Learning

【对Mauve（pillutla 等人）生成评估指标的分析】On the Usefulness of Embeddings, Clusters and Strings for Text Generation Evaluation

Promptagator: Few-shot Dense Retrieval From 8 Examples

【三个臭皮匠，顶个诸葛亮】Self-Consistency Improves Chain of Thought Reasoning in Language Models

- 【用知识作为臭皮匠的参考】Rethinking with Retrieval: Faithful Large Language Model Inference

【反转，输入和标签为条件生成指令】Guess the Instruction! Making Language Models Stronger Zero-Shot Learners

【LLM 的反向推导自我验证】Large Language Models are reasoners with Self-Verification

【检索-生成证据流程下的安全场景的方法】Foveate, Attribute, and Rationalize: Towards Safe and Trustworthy AI

【基于beam search的文本生成式信息抽取片段的置信度估计】How Does Beam Search improve Span-Level Confidence Estimation in Generative Sequence Labeling?


SPT: Semi-Parametric Prompt Tuning for Multitask Prompted Learning

【对抽取式摘要黄金标签的探讨】Text Summarization with Oracle Expectation

【基于马氏距离的条件文本生成OOD检测方法】Out-of-Distribution Detection and Selective Generation for Conditional Language Models

【注意力模块集成Prompt进行样例级别的预测】Model ensemble instead of prompt fusion: a sample-specific knowledge transfer method for few-shot prompt tuning

【多个任务的Prompt通过分解和蒸馏到一个Prompt】Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning

【step-by-step推理生成文本的评估指标，可以作为下次分享选题】ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning

【校准序列似然改进条件语言生成】Calibrating Sequence likelihood Improves Conditional Language Generation

【基于梯度优化的文本攻击方法】TextGrad: Advancing Robustness Evaluation in NLP by Gradient-Driven Optimization

【GMM建模ICL决策分类边界从而校准】Prototypical Calibration for Few-shot Learning of Language Models

【改写问题，以及基于图的ICL聚合方法】Ask Me Anything: A simple strategy for prompting language models

【用于从未注释的示例池中选择好的候选作为ICL的数据库】Selective Annotation Makes Language Models Better Few-Shot Learners


PromptBoosting: Black-Box Text Classification with Ten Forward Passes

Attention-Guided Backdoor Attacks against Transformers


【Prompt Mask位置自动选标签词】Pre-trained Language Models can be Fully Zero-Shot Learners

【压缩FiD输入向量的长度，且输出时重新排序来输出文档排名】FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation

【大模型教小模型生成解释】PINTO: Faithful Language Reasoning Using Prompted-Generated Rationales

【寻找预训练影响子集】ORCA: Interpreting Prompted Language Models via Locating Supporting Evidence in the Ocean of Pretraining Data

【提示工程，针对的是Instruction，一阶段生成二阶段排序过滤】Large Language Models are Human-Level Prompt Engineers


Knowledge Unlearning for Mitigating Privacy Risks in Language Models

Editing models with task arithmetic

【不用每次都输入指令和样例，将其转换为参数高效模块，】HINT: Hypernetwork Instruction Tuning for Efficient Zero-Shot Generalisation

【不需要人工选样例的ICL展示生成方法】Z-ICL: Zero-Shot In-Context Learning with Pseudo-Demonstrations

【任务Instruction和文本一起生成Embedding】One Embedder, Any Task: Instruction-Finetuned Text Embeddings

【大模型教小模型CoT】KNIFE: Knowledge Distillation with Free-Text Rationales

【信息提取式生成模型的源和目标分词不一致问题】Tokenization Consistency Matters for Generative Models on Extractive NLP Tasks

Parsel: A Unified Natural Language Framework for Algorithmic Reasoning

【ICL样例选择，一阶段选择二阶段排序】Self-adaptive In-context Learning


【精读，可读的prompt无监督选择方法，GPT-2】Toward Human Readable Prompt Tuning: Kubrick's The Shining is a good movie, and a good prompt too

## **Survey和数据集** 

【PRONTOQA数据集测试CoT推理能力，发现Planning能力仍受限】Language Models Can (kind of) Reason: A Systematic Formal Analysis of Chain-of-Thought

【reasoning数据集】WikiWhy: Answering and Explaining Cause-and-Effect Questions

【reasoning数据集】STREET: A MULTI-TASK STRUCTURED REASONING AND EXPLANATION BENCHMARK

【reasoning数据集，比较OPT预训练和微调，包括CoT微调模型】 ALERT: Adapting Language Models to Reasoning Tasks

【浙大张宁豫团队对近期reasoning的总结】Reasoning with Language Model Prompting: A Survey

【复旦肖仰华团队对文本生成技术和方向的总结】Harnessing Knowledge and Reasoning for Human-Like Natural Language Generation: A Brief Review


【近期reasoning文章的总结，来自UIUC的Jie Huang】Towards Reasoning in Large Language Models: A Survey


【回顾数学推理和DL的任务、数据集和方法】A Survey of Deep Learning for Mathematical Reasoning

A Survey on Natural Language Processing for Programming


奖励建模数据集:

- 该数据集由Stiennon et al. (2020)提供并包含对模型生成的摘要的人工反馈。 这个数据集有两个部分：比较和axis。 在比较部分，人工注释者被要求从两个摘要中选择最好的。 在axis部分，人工注释者根据Likert scale为摘要质量打分。 比较部分只有训练和验证拆分，axis部分只有测试和验证拆分。 论文中用于训练奖励模型的摘要来自 TL;DR 数据集。 其他验证和测试数据来自 TL;DR 数据集、CNN 文章和每日邮报文章。https://huggingface.co/datasets/openai/summarize_from_feedback
- 该数据集来自 Ganguli et al. (2022); Bai et al. (2022a)  并包含人类评价的对话。3 一个例子包括人类和聊天机器人之间的一对对话。 人类更喜欢这两种对话中的一种。 https://huggingface.co/datasets/Anthropic/hh-rlhf
- 该数据集来自 Nakano et al. (2021). 共有 19,578 次比较。 数据集中的每个示例都包含一对问题的模型答案，以及相关的元数据。 每个答案都有一个来自人类的偏好分数，可用于确定两个答案中哪个更好。 https://huggingface.co/datasets/openai/webgpt_comparisons
- SHP是一个由385K个集体人类对18个不同主题领域的问题/指示的反应的偏好组成的数据集，从烹饪到法律咨询。这些偏好旨在反映一种回答对另一种回答的帮助程度，并打算用于训练RLHF奖励模型和NLG评估模型（例如SteamSHP）。 https://huggingface.co/datasets/stanfordnlp/SHP

Red-teaming数据集，harmless vs. helpful， *RLHF*+scale更难被攻击 （另一个有效的技术是CoT fine-tuning）:
- 对于什么是成功的攻击，人类之间总体上达成的共识很低。
- Meta’s Bot Adversarial Dialog dataset https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/bot_adversarial_dialogue
- Anthropic’s red-teaming attempts https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/red-team-attempts
- AI2’s RealToxicityPrompts https://huggingface.co/datasets/allenai/real-toxicity-prompts
---
## **其他**
【知识】+【推理】+【生成】

如果对您有帮助，请star支持一下，欢迎Pull Request~ 

主观整理，时间上主要从ICLR 2023 Rebuttal期间开始的，包括ICLR，ACL，ICML等预印版论文。

不妥之处或者建议请指正! Dongfang Li, crazyofapple@gmail.com 
