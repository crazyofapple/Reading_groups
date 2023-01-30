## 关于LLM和当前热点的文章，包括ICLR 2023 和 ACL 2023的预印版

**计算的力量**： 很多证据表明，机器学习的进步很大程度上是由计算驱动的，而不是研究，请参考："[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)"。
有研究表明，[人工智能计算使用量大约每3-4个月翻一番，而效率提升每16个月才翻一番](https://openai.com/blog/)。其中计算使用主要由计算驱动，而效率则由研究驱动。 
这意味着计算增长在历史上主导了机器学习和其子领域的进步。尽管如此，未来是否有更颠覆Transformer的架构仍需要我们重视。
目前的NLP研究热点大部分基于更先进的LLM （2022-）。尤其是ChatGPT通过Alignment技术利用少于预训练几千倍的计算和（$50w）释放了GPT大模型对话能力并火出圈。所以本库对这些文章进行追踪和归类，更能让我们把握前沿，看清方向。个人想法，不妥之处或者建议请指正。


【应用ICL的多步推理方法，很有启发】ReAct: Synergizing Reasoning and Acting in Language Models

```
 【单独使用LLM不足以创建真正强大的APP，将LLM与其他计算或知识来源相结合时，真正的力量才会出现】
 langchain - Building applications with LLMs through composability

类似想法 0. TALM: Tool Augmented Language Models
类似想法 1. DEMONSTRATE–SEARCH–PREDICT:Composing retrieval and language models for knowledge-intensive NLP
类似想法 2. LAMBADA: Backward Chaining for Automated Reasoning in Natural Language
类似想法 3.【选择和推理】Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning
类似想法 4. Language Models as Agent Models
类似想法 5. Prompting Is Programming: A Query Language For Large Language Models
类似想法 6.【Neurips 22'】Language Model Cascades 
```

What learning algorithm is in-context learning? Investigations with linear models

【大模型直接产生证据上下文】Generate rather than Retrieve: Large Language Models are Strong Context Generators

【中英文的大模型，超过GPT-3】GLM-130B: An Open Bilingual Pre-trained Model

【具有4个特定操作的写作模型】PEER: A Collaborative Language Model

【将Python、SQL执行器和大模型结合】Binding Language Models in Symbolic Languages

【一种文本生成的新的优化方式】Tailoring Language Generation Models under Total Variation Distance

【Alignment新的基准，模型库和新方法】Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization

LexMAE: Lexicon-Bottlenecked Pretraining for Large-Scale Retrieval

DocPrompting: Generating Code by Retrieving the Docs

【对Mauve（pillutla 等人）生成评估指标的分析】On the Usefulness of Embeddings, Clusters and Strings for Text Generation Evaluation

【文字转为图像训练，缓解了Vocabulary的需要并抗某些攻击】Language Modelling with Pixels

InCoder: A Generative Model for Code Infilling and Synthesis

Promptagator: Few-shot Dense Retrieval From 8 Examples

【检索Text相关图像进行语言模型预训练】Visually-Augmented Language Modeling

【三个臭皮匠，顶个诸葛亮】Self-Consistency Improves Chain of Thought Reasoning in Language Models

```【用知识作为臭皮匠的参考】Rethinking with Retrieval: Faithful Large Language Model Inference```

【反转，输入和标签为条件生成指令】Guess the Instruction! Making Language Models Stronger Zero-Shot Learners

【对抽取式摘要黄金标签的探讨】Text Summarization with Oracle Expectation

【基于马氏距离的条件文本生成OOD检测方法】Out-of-Distribution Detection and Selective Generation for Conditional Language Models

【基于多任务训练用于少样本数据增强的模型】KnowDA: All-in-One Knowledge Mixture Model for Data Augmentation in Low-Resource NLP

A Non-monotonic Self-terminating Language Model

【多个任务的Prompt通过分解和蒸馏到一个Prompt】Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning

【用小模型参数加速大模型训练过程（不从头）】Learning to Grow Pretrained Models for Efficient Transformer Training

【注意力模块集成Prompt进行样例级别的预测】Model ensemble instead of prompt fusion: a sample-specific knowledge transfer method for few-shot prompt tuning

Mass-Editing Memory in a Transformer

【step-by-step推理生成文本的评估指标，可以作为下次分享选题】ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning

【procedural planning的工作，暂时不感兴趣】Neuro-Symbolic Procedural Planning with Commonsense Prompting

【校准序列似然改进条件语言生成】Calibrating Sequence likelihood Improves Conditional Language Generation

【基于梯度优化的文本攻击方法】TextGrad: Advancing Robustness Evaluation in NLP by Gradient-Driven Optimization

【多种知识源MoE半参数知识融合模型】Knowledge-in-Context: Towards Knowledgeable Semi-Parametric Language Models

【GMM建模ICL决策分类边界从而校准】Prototypical Calibration for Few-shot Learning of Language Models

【改写问题，以及基于图的ICL聚合方法】Ask Me Anything: A simple strategy for prompting language models

【大模型的CoT具有跨语言能力】Language models are multilingual chain-of-thought reasoners

【大模型的binary implicature resolution任务，这种暗示难并没有缩放现象】Large language models are not zero-shot communicators （https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/implicatures）

【不同数据集上的多个已训练模型合并方法】Dataless Knowledge Fusion by Merging Weights of Language Models

【用于从未注释的示例池中选择好的候选作为ICL的数据库】Selective Annotation Makes Language Models Better Few-Shot Learners

【复杂的提示提升了CoT】Complexity-Based Prompting for Multi-step Reasoning

【PRONTOQA数据集测试CoT推理能力，发现Planning能力仍受限】Language Models Can (kind of) Reason: A Systematic Formal Analysis of Chain-of-Thought

PromptBoosting: Black-Box Text Classification with Ten Forward Passes

Attention-Guided Backdoor Attacks against Transformers

【很有启发，检索机制代替 Transformer 中的 FFN 的通用架构(×2.54 time)，以便解耦存储在模型参数中的知识】Language model with Plug-in Knowldge Memory

【Prompt Mask位置自动选标签词】Pre-trained Language Models can be Fully Zero-Shot Learners

【大模型生成证据（背诵）然后进行小样本闭卷问答】Recitation-Augmented Language Models

What Matters In The Structured Pruning of Generative Language Models?

Towards Conditionally Dependent Masked Language Models

【迭代地校准不完美生成的独立校正器，Sean Welleck的后续文章】Generating Sequences by Learning to Self-Correct

```
预测：AI反馈很快会取代人工用户反馈用于模型更新 
类似想法 1. Constitutional AI: Harmlessness from AI Feedback 
类似想法 2. Discovering Language Model Behaviors with Model-Written Evaluations
```

【压缩FiD输入向量的长度，且输出时重新排序来输出文档排名】FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation

【大模型教小模型生成解释】PINTO: Faithful Language Reasoning Using Prompted-Generated Rationales

【任务歧义：缩放 RLHF 模型在消除歧义任务方面表现最佳。微调比few-shot prompting更有帮助】Task Ambiguity in Humans and Language Models

【持续学习：新任务增加一个prompt，且上一个任务的prompt和大模型不变】Progressive Prompts: Continual Learning for Language Models without Forgetting

【GPT-3的测试，包括记忆，校准，偏见等】Prompting GPT-3 To Be Reliable

【目标：为维基百科中某些参考文献支持的Query生成一篇事实正确的文章】WebBrain: Learning to Generate Factually Correct Articles for Queries by Grounding on Large Web Corpus

【reasoning数据集】WikiWhy: Answering and Explaining Cause-and-Effect Questions

【reasoning数据集】STREET: A MULTI-TASK STRUCTURED REASONING AND EXPLANATION BENCHMARK

【reasoning数据集，比较OPT预训练和微调，包括CoT微调模型】 ALERT: Adapting Language Models to Reasoning Tasks

Towards Boosting the Open-Domain Chatbot with Human Feedback

【寻找预训练影响子集】ORCA: Interpreting Prompted Language Models via Locating Supporting Evidence in the Ocean of Pretraining Data

【离散提示的跨语言模型研究】Can discrete information extraction prompts generalize across language models?

【新的神经架构 (FOLNet)，其中包含一阶逻辑归纳偏差】Learning Language Representations with Logical Inductive Bias

【将外部物理模拟器的结果结合在context中】Mind's Eye: Grounded Language Model Reasoning through Simulation

【提示工程，针对的是Instruction，一阶段生成二阶段排序过滤】Large Language Models are Human-Level Prompt Engineers

【对比一致搜索 (CCS)无监督识别语言模型中的潜在（二元）知识】Discovering Latent Knowledge in Language Models Without Supervision

【记忆率与训练中的模型大小、前缀长度和重复率呈对数线性关系】Quantifying Memorization Across Neural Language Models

【很有启发，将问题通过GPT迭代分解为子问题并回答】Measuring and Narrowing the Compositionality Gap in Language Models

```
【研究是否或什么时候分步作答对阅读有效，零样本和低资源有效】When Do Decompositions Help for Machine Reading?
类似想法 Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
类似想法 Successive Prompting for Decomposing Complex Questions
```

Knowledge Unlearning for Mitigating Privacy Risks in Language Models

Editing models with task arithmetic

【短文本训练，长文本测试，评估模型的变长适应能力】A Length-Extrapolatable Transformer

【回顾数学推理和DL的任务、数据集和方法】A Survey of Deep Learning for Mathematical Reasoning

【两种ICL样例选择的方法，基于OPT和GPTJ的实验】Careful Data Curation Stabilizes In-context Learning

【语义解析任务，ICL的样例选择方法，CODEX和T5-large】Diverse Demonstrations Improve In-context Compositional Generalization

【很有启发，自由文本约束下的文本生成方法】Controllable Text Generation with Language Constraints

【人类-AI语言交互评价框架】Evaluating Human-Language Model Interaction

```类似文章 Measuring the Human Utility of Free-Text Rationales in Human-AI Collaboration```

GanLM: Encoder-Decoder Pre-training with an Auxiliary Discriminator

Go-tuning: Improving Zero-shot Learning Abilities of Smaller Language Models

【不用每次都输入指令和样例，将其转换为参数高效模块，】HINT: Hypernetwork Instruction Tuning for Efficient Zero-Shot Generalisation

Is GPT-3 a Psychopath? Evaluating Large Language Models from a Psychological Perspective

Parsel: A Unified Natural Language Framework for Algorithmic Reasoning

【基于state-space models的预训练语言模型，超过BERT】Pretraining Without Attention

【ICL样例选择，一阶段选择二阶段排序】Self-adaptive In-context Learning

【自动生成Instruction tuning的数据用于GPT-3的训练】Self-Instruct: Aligning Language Model with Self Generated Instructions

```
【和yizhong wang那篇类似自动生成Instruction的数据，面向T0】Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor
Language model acceptability judgements are not always robust to context
SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks
(FLAN-T5-CoT) 【CoT微调】Scaling Instruction-Finetuned Language Models  
```
![image](https://user-images.githubusercontent.com/3351073/214972577-e9fb27a6-55a8-40b6-b38f-cbb68a115739.png)

【信息提取式生成模型的源和目标分词不一致问题】Tokenization Consistency Matters for Generative Models on Extractive NLP Tasks

【精读，可读的prompt无监督选择方法，GPT-2】Toward Human Readable Prompt Tuning: Kubrick's The Shining is a good movie, and a good prompt too?

【近期reasoning文章的总结，来自UIUC的Jie Huang】Towards Reasoning in Large Language Models: A Survey

【浙大张宁豫团队对近期reasoning的总结】Reasoning with Language Model Prompting: A Survey

【复旦肖仰华团队对文本生成技术和方向的总结】Harnessing Knowledge and Reasoning for Human-Like Natural Language Generation: A Brief Review

【OSU研究CoT哪个部分对性能有效】Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters

```
类似想法 
Complementary Explanations for Effective In-Context Learning (UT Austin, Xi Ye, Greg Durrett) 
Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango
```

【对OPT模型进行不同大小训练的过程研究，发现困惑度是ICL的指标】Training Trajectories of Language Models Across Scales

【ICL是另一种形式的gradient更新】Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta Optimizers

```相关文章：Transformers learn in-context by gradient descent```

【不需要人工选样例的ICL展示生成方法】Z-ICL: Zero-Shot In-Context Learning with Pseudo-Demonstrations

【对CoT生成的解释进行二次调整，用带参数的refiner模块+信息熵优化】Explanation Regeneration via Information Bottleneck

【任务Instruction和文本一起生成Embedding】One Embedder, Any Task: Instruction-Finetuned Text Embeddings

【大模型教小模型CoT】KNIFE: Knowledge Distillation with Free-Text Rationales

【什么时候检索，什么时候用大模型足够】When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories

【检索增强的CoT做知识Intensive的任务】Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions

【调查检索式模型的特点，发现两者均对reasoning有限】Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model

``` 
检索式+LLM的想法是下一步的方向，但是不是唯一的答案还需要看看
Rethink Search: Making Domain Experts out of Dilettantes
Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models
```

【对GPT-3类似公务员那种智力题类比测试】Emergent Analogical Reasoning in Large Language Models

【LLM 的反向推导自我验证】Large Language Models are reasoners with Self-Verification

【检索-生成证据流程下的安全场景的方法】Foveate, Attribute, and Rationalize: Towards Safe and Trustworthy AI

【GPT-3用于数据标注（如情感分类）】Is GPT-3 a Good Data Annotator?

【基于beam search的文本生成式信息抽取片段的置信度估计】How Does Beam Search improve Span-Level Confidence Estimation in Generative Sequence Labeling?

【归纳推理的自然语言方式】Language Models as Inductive Reasoners

SPT: Semi-Parametric Prompt Tuning for Multitask Prompted Learning

【不同Prompt序列困惑度越低性能越好】 Demystifying Prompts in Language Models via Perplexity Estimation

【自动构建CoT中的样例的解释并用于CoT】Automatic Chain of Thought Prompting in Large Language Models

【让小模型学会CoT能力】In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models

【大模型教小模型CoT】Large Language Models Are Reasoning Teachers

【自我迭代生成（利用python验证过）训练数据】Language Models Can Teach Themselves to Program Better

```
相关文章： STaR: Bootstrapping Reasoning With Reasoning, 来自Neurips 22 (生成CoT数据用于模型微调）, 引起后续一系列教小模型的CoT的文章
类似想法 【知识蒸馏】 Teaching Small Language Models to Reason
类似想法很多，包括KAIST和Xiang Ren组（【CoT的rationale微调（教授）时进行扰动】PINTO: Faithful Language Reasoning Using Prompt-Generated Rationales等）和这一篇Large Language Models Are Reasoning Teachers和ETH的【CoT的数据分别训练问题分解和问题解答模型】Distilling Multi-Step Reasoning Capabilites of Large Language Models into Smaller Models via Semantic Decompositions
```

【生成预测时采用相似度选phrase而不是softmax预测token】Nonparametric Masked Language Modeling

【长文本的ICL方法】Parallel Context Windows Improve In-Context Learning of Large Language Models

【InstructGPT模型自己生成ICL的样例】Self-Prompting Large Language Models for Open-Domain QA

【zero-shot CoT在敏感问题下会表现出bias和toxicity】 On Second Thought, Let’s Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning

【通过分组和注意力机制使得ICL能够输入更多的标注样本】Structured Prompting: Scaling In-Context Learning to 1,000 Examples

A Survey on Natural Language Processing for Programming

【Grounding+LLM的系列文章接下来会有很多】LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models
```
比如说： Do As I Can, Not As I Say: Grounding Language in Robotic Affordances  
https://say-can.github.io/
```

Momentum Calibration for Text Generation

【个性化风格的prompt学习，OPT】Extensible Prompts for Language Models

【CoT直接生成program code，然后让python interpreter执行】Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks
```
相关文章：【EMNLP 22'】Language Models of Code are Few-Shot Commonsense Learners
【Heng Ji组】Code4Struct: Code Generation for Few-Shot Structured Prediction from Natural Language
```
