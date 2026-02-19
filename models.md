# Summary Table

|Model                  |Developer      |Release Date|Parameters          |Context Window|Popularity/Users                           |Market Share                    |Architecture Highlights               |
|-----------------------|---------------|------------|--------------------|--------------|-------------------------------------------|--------------------------------|--------------------------------------|
|**GPT-4**              |OpenAI         |Mar 2023    |~1.76T (est.)       |8K-128K       |800M+ weekly active users (Oct 2025)       |62.5% (AI tools), 81% (chatbots)|Multimodal, MoE (rumored), RLHF       |
|**GPT-3.5**            |OpenAI         |Nov 2022    |175B                |4K-16K        |Powers ChatGPT free tier                   |Included in ChatGPT share       |Transformer decoder, instruction-tuned|
|**Claude 3**           |Anthropic      |Mar 2024    |20-200B (est.)      |200K          |30M monthly (Q2 2025)                      |3.2% (US), 21% (global LLM)     |Constitutional AI, multimodal         |
|**Claude 4**           |Anthropic      |2025        |Undisclosed         |200K-1M (beta)|Growing enterprise adoption                |Part of Claude 3.2% share       |Extended thinking, Constitutional AI  |
|**Gemini**             |Google         |Dec 2023    |6B-1.56T            |32K-1M        |650M monthly (Nov 2025)                    |13.5%                           |Natively multimodal, MoE (1.5)        |
|**PaLM 2**             |Google         |May 2023    |~340B               |8K            |Powered Bard (deprecated)                  |Replaced by Gemini              |Compute-optimal, multilingual         |
|**LLaMA 1**            |Meta           |Feb 2023    |7B-65B              |2K            |Research community (leaked)                |N/A - Research only             |RoPE, SwiGLU, efficient inference     |
|**LLaMA 2**            |Meta           |Jul 2023    |7B-70B              |4K            |Widely used in open-source                 |Most popular open-source base   |GQA, commercially usable              |
|**LLaMA 3/3.1**        |Meta           |Apr/Jul 2024|8B-405B             |8K-128K       |Powers Meta AI, widespread OSS adoption    |20% (among LLM users surveyed)  |Largest open model (405B), tool use   |
|**Mistral 7B**         |Mistral AI     |Sep 2023    |7.3B                |8K (32K eff.) |Popular for edge deployment                |Growing in Europe               |Sliding Window Attention, GQA         |
|**Mixtral 8x7B**       |Mistral AI     |Dec 2023    |46.7B (12.9B active)|32K           |Enterprise & OSS adoption                  |Competitive with proprietary    |Sparse MoE, multilingual              |
|**Falcon 180B**        |TII            |Sep 2023    |180B                |2K            |Open-source community                      |Research/niche                  |RefinedWeb dataset, open-source       |
|**Vicuna**             |Berkeley et al.|Mar 2023    |7B-33B              |2K            |Research/academic use                      |Academic/research               |Fine-tuned LLaMA, low-cost training   |
|**Cohere Command R/R+**|Cohere         |2024        |35B-104B            |128K          |Enterprise RAG applications                |Strong in enterprise RAG        |RAG-optimized, multilingual           |
|**BLOOM**              |BigScience     |Jul 2022    |176B                |2K            |Open science community                     |Research/multilingual           |46 languages, fully open              |
|**MPT**                |MosaicML       |May 2023    |7B-30B              |Up to 65K     |Commercial OSS deployments                 |Niche/specialized               |ALiBi, FlashAttention, long context   |
|**Yi**                 |01.AI          |Nov 2023    |6B-34B              |4K-200K       |Strong in China/bilingual use              |Growing in Asia                 |Bilingual (EN/CN), extended context   |
|**Phi-2**              |Microsoft      |Dec 2023    |2.7B                |2K            |Research on data quality                   |Research/edge                   |“Textbook quality” data, efficient    |
|**StarCoder**          |BigCode        |May 2023    |15.5B               |8K            |Developer tools, GitHub Copilot alternative|Code generation niche           |80+ languages, fill-in-middle         |
|**Microsoft Copilot**  |Microsoft      |2023        |Powered by GPT-4    |Varies        |14% US market share                        |14% (integrated in Office)      |Integration with Microsoft 365        |

**Note on User Statistics:**

- ChatGPT (GPT-4/3.5): 800 million weekly active users, 5.8 billion monthly visits (as of October 2025)
- Gemini: 650 million monthly active users (November 2025), 1.2 billion monthly visits
- Claude: 30 million monthly active users, 25 billion API calls/month (Q2 2025)
- Market share data varies by region and measurement methodology
- Open-source models (LLaMA, Mistral, etc.) have distributed usage that’s harder to quantify but powers countless applications

-----

### 1. GPT-4 (March 2023)

**Developer:** OpenAI

**Architecture:** Transformer-based, decoder-only (rumored multimodal architecture)

**Parameters:** Estimated 1.76 trillion (unconfirmed, exact number proprietary)

**Context Window:** 8,192 tokens (GPT-4), 32,768 tokens (GPT-4-32K), 128,000 tokens (GPT-4 Turbo)

**Training Data Cutoff:** September 2021 (initial), April 2023 (Turbo versions)

**Key Technical Details:**

- Multimodal capabilities (text and image input)
- Mixture of Experts (MoE) architecture (rumored)
- Significantly improved reasoning and factual accuracy over GPT-3.5
- Enhanced instruction following and safety alignment
- RLHF (Reinforcement Learning from Human Feedback) training

**Notable Variants:**

- GPT-4 Turbo (November 2023)
- GPT-4V (Vision, September 2023)
- GPT-4-0125-preview, GPT-4-0613

**Applications:** ChatGPT Plus, API access, enterprise solutions, coding assistance, creative writing, analysis

-----

### 2. GPT-3.5 (November 2022)

**Developer:** OpenAI

**Architecture:** Transformer-based, decoder-only

**Parameters:** 175 billion

**Context Window:** 4,096 tokens (initially), 16,385 tokens (GPT-3.5-turbo-16k)

**Training Data Cutoff:** September 2021

**Key Technical Details:**

- Fine-tuned version of GPT-3 with instruction following
- RLHF for improved alignment
- Code-davinci-002 as base for ChatGPT
- Significantly more cost-effective than GPT-4
- Optimized for conversational interactions

**Notable Variants:**

- GPT-3.5-turbo (various versions)
- text-davinci-003
- gpt-3.5-turbo-instruct

**Applications:** ChatGPT free tier, API access, chatbots, content generation

-----

### 3. Claude 3 Family (March 2024)

**Developer:** Anthropic

**Architecture:** Transformer-based, custom architecture with Constitutional AI

**Models:**

- **Claude 3 Opus:** ~175-200 billion parameters (estimated)
- **Claude 3 Sonnet:** ~50-100 billion parameters (estimated)
- **Claude 3 Haiku:** ~20-40 billion parameters (estimated)

**Context Window:** 200,000 tokens (all variants)

**Training Data Cutoff:** August 2023

**Key Technical Details:**

- Constitutional AI (CAI) training methodology
- Extended context window capabilities
- Vision capabilities across all models
- Improved reasoning and coding abilities
- Harmlessness training with minimal loss in helpfulness
- Near-instant citations and analysis of long documents

**Applications:** Claude.ai chat interface, API access, research, coding, document analysis

-----

### 4. Claude 4 Family (2025)

**Developer:** Anthropic

**Architecture:** Advanced transformer-based with Constitutional AI

**Models:**

- **Claude Opus 4.1 and 4:** Latest flagship models
- **Claude Sonnet 4.5 and 4:** Balanced performance and efficiency

**Context Window:** Extended beyond previous generations

**Training Data Cutoff:** January 2025

**Key Technical Details:**

- Claude Sonnet 4.5 represents the smartest model in the family
- Improved efficiency and everyday usability
- Enhanced reasoning capabilities
- Maintained Constitutional AI principles
- Advanced multimodal processing

**Applications:** Claude.ai, API, Claude Code, Chrome extension, Excel plug-in

-----

### 5. PaLM 2 (May 2023)

**Developer:** Google

**Architecture:** Transformer-based, decoder-only with improved training techniques

**Parameters:** 340 billion (estimated, multiple model sizes)

**Context Window:** 8,192 tokens (varies by version)

**Key Technical Details:**

- Compute-optimal scaling approach
- Trained on multilingual and code-focused datasets
- Improved reasoning and mathematical capabilities
- Four model sizes: Gecko, Otter, Bison, Unicorn
- Enhanced multilingual capabilities (100+ languages)
- Pathways system for efficient training

**Notable Features:**

- Strong performance on coding benchmarks
- Advanced mathematical reasoning
- Multilingual translation capabilities

**Applications:** Google Bard (now Gemini), Google Workspace, API access

-----

### 6. Gemini Family (December 2023)

**Developer:** Google DeepMind

**Architecture:** Multimodal transformer, natively multimodal from ground up

**Models:**

- **Gemini Ultra:** ~1.56 trillion parameters (estimated)
- **Gemini Pro:** ~175 billion parameters (estimated)
- **Gemini Nano:** ~6-20 billion parameters (on-device versions)

**Context Window:** 32,768 tokens (Gemini Pro 1.0), up to 1 million tokens (Gemini 1.5 Pro)

**Key Technical Details:**

- Natively multimodal (text, image, audio, video, code)
- Mixture of Experts architecture (Gemini 1.5)
- Exceptional long-context understanding
- TPU v4 and v5 training infrastructure
- Advanced reasoning across modalities

**Notable Variants:**

- Gemini 1.5 Pro (February 2024) with 1M token context
- Gemini 1.5 Flash (May 2024) for speed

**Applications:** Google AI Studio, Gemini app, Google Workspace, Android integration

-----

### 7. LLaMA 1 (February 2023)

**Developer:** Meta AI

**Architecture:** Transformer-based, decoder-only

**Parameters:** 7B, 13B, 33B, 65B variants

**Context Window:** 2,048 tokens

**Training Data:** 1.4 trillion tokens

**Key Technical Details:**

- Optimized for research and inference efficiency
- Released as research-only (leaked publicly)
- Pre-normalization using RMSNorm
- SwiGLU activation function
- Rotary Positional Embeddings (RoPE)
- Trained on publicly available data only

**Training Dataset:**

- CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange

**Applications:** Research, foundation for many open-source models

-----

### 8. LLaMA 2 (July 2023)

**Developer:** Meta AI

**Architecture:** Transformer-based, decoder-only

**Parameters:** 7B, 13B, 70B variants

**Context Window:** 4,096 tokens

**Training Data:** 2 trillion tokens

**Key Technical Details:**

- Open-source and commercially usable
- Grouped Query Attention (GQA) for efficient inference
- RLHF fine-tuning for chat variants
- Improved safety and alignment
- 40% more training data than LLaMA 1
- RoPE positional embeddings

**Notable Variants:**

- LLaMA 2-Chat (instruction-tuned versions)
- Code LLaMA (specialized for coding)

**Applications:** Open-source applications, research, custom fine-tuning, chatbots

-----

### 9. LLaMA 3 (April 2024)

**Developer:** Meta AI

**Architecture:** Transformer-based, decoder-only with optimizations

**Parameters:** 8B, 70B, 405B variants

**Context Window:** 8,192 tokens (initial), 128K tokens (extended versions)

**Training Data:** 15+ trillion tokens

**Key Technical Details:**

- Significantly scaled training data
- Improved tokenizer (128K vocabulary)
- Grouped Query Attention
- Enhanced multilingual capabilities
- State-of-the-art performance on benchmarks
- Advanced post-training with RLHF

**Notable Features:**

- LLaMA 3.1 with extended context (128K)
- LLaMA 3.2 with multimodal variants
- Tool use capabilities

**Applications:** Open-source development, Meta AI assistant, research

-----

### 10. Mistral 7B (September 2023)

**Developer:** Mistral AI

**Architecture:** Transformer-based, decoder-only with Sliding Window Attention

**Parameters:** 7.3 billion

**Context Window:** 8,192 tokens (32K effective with sliding window)

**Key Technical Details:**

- Sliding Window Attention (SWA) mechanism
- Grouped Query Attention (GQA)
- Rolling buffer cache for efficient inference
- Apache 2.0 license
- Outperforms LLaMA 2 13B on many benchmarks
- Byte-fallback BPE tokenizer

**Training Approach:**

- Trained on undisclosed dataset
- Optimized for efficiency and performance balance

**Applications:** Open-source projects, edge deployment, custom applications

-----

### 11. Mixtral 8x7B (December 2023)

**Developer:** Mistral AI

**Architecture:** Sparse Mixture of Experts (SMoE), decoder-only

**Parameters:** 46.7 billion total, 12.9 billion active per token

**Context Window:** 32,768 tokens

**Key Technical Details:**

- 8 expert networks, 2 active per token
- Router network for expert selection
- Extremely efficient inference
- Multilingual capabilities (English, French, German, Spanish, Italian)
- Sliding Window Attention
- Outperforms GPT-3.5 on many benchmarks

**Training:**

- Trained on undisclosed multilingual dataset
- Apache 2.0 license

**Notable Variant:**

- Mixtral 8x22B (April 2024) - 141B total, 39B active parameters

**Applications:** Open-source applications, multilingual tasks, efficient inference

-----

### 12. Falcon 180B (September 2023)

**Developer:** Technology Innovation Institute (TII), Abu Dhabi

**Architecture:** Transformer-based, decoder-only

**Parameters:** 180 billion

**Context Window:** 2,048 tokens

**Training Data:** 3.5 trillion tokens (RefinedWeb dataset)

**Key Technical Details:**

- Multiquery attention
- FlashAttention for efficiency
- Trained primarily on web data (RefinedWeb)
- Open-source with permissive license
- Optimized decoder architecture
- RoPE positional embeddings

**Training Infrastructure:**

- 4,096 A100 GPUs
- 3-4 months training time

**Applications:** Research, open-source development, competitive with closed models

-----

### 13. Vicuna (March 2023)

**Developer:** UC Berkeley, CMU, Stanford, UC San Diego, MBZUAI

**Architecture:** Fine-tuned LLaMA (Transformer-based)

**Parameters:** 7B, 13B, 33B variants

**Context Window:** 2,048 tokens

**Key Technical Details:**

- Fine-tuned on 70K user-shared ChatGPT conversations
- Achieved ~90% of ChatGPT quality (early evaluations)
- Open-source training and evaluation framework
- Multi-turn conversation capabilities
- Cost-effective training approach (~$300)

**Training Methodology:**

- ShareGPT dataset
- Supervised fine-tuning on LLaMA base

**Applications:** Research, chatbot development, open-source alternatives

-----

### 14. Cohere Command Series (2023-2024)

**Developer:** Cohere

**Architecture:** Transformer-based with retrieval augmentation capabilities

**Models:**

- Command
- Command Light
- Command R (March 2024)
- Command R+ (April 2024)

**Parameters:**

- Command R: ~35 billion
- Command R+: ~104 billion

**Context Window:** 128,000 tokens (Command R/R+)

**Key Technical Details:**

- Optimized for enterprise RAG (Retrieval-Augmented Generation)
- Multi-step tool use
- Multilingual capabilities (10+ languages)
- Citation generation
- Structured output support
- Advanced grounding capabilities

**Applications:** Enterprise search, RAG applications, customer support, multi-lingual tasks

-----

### 15. BLOOM (July 2022)

**Developer:** BigScience (international collaboration)

**Architecture:** Transformer-based, decoder-only

**Parameters:** 176 billion (largest variant)

**Context Window:** 2,048 tokens

**Training Data:** 366 billion tokens (ROOTS corpus)

**Key Technical Details:**

- Trained on 46 natural languages and 13 programming languages
- ALiBi positional embeddings
- Fully open-source (including training data)
- Collaborative international research project
- Embedding LayerNorm
- 250,000 GPU-hours on Jean Zay supercomputer

**Model Variants:**

- 560M, 1.1B, 1.7B, 3B, 7.1B, 176B parameters

**Applications:** Multilingual research, open science, democratized AI access

-----

### 16. MPT (MosaicML Pretrained Transformer) (May 2023)

**Developer:** MosaicML (Databricks)

**Architecture:** Transformer-based, decoder-only with optimizations

**Parameters:** 7B, 30B variants

**Context Window:** Up to 65,536 tokens (MPT-7B-StoryWriter)

**Key Technical Details:**

- ALiBi positional embeddings for extended context
- FlashAttention for efficiency
- Commercially usable (Apache 2.0/CC-BY-SA-3.0)
- Trained on 1 trillion tokens (MPT-7B)
- Low-Precision LayerNorm
- No biases in attention and feedforward layers

**Notable Variants:**

- MPT-7B-StoryWriter (65K context)
- MPT-7B-Instruct
- MPT-7B-Chat
- MPT-30B

**Applications:** Long-form content generation, commercial applications, research

-----

### 17. Yi Series (November 2023)

**Developer:** 01.AI (Kai-Fu Lee)

**Architecture:** Transformer-based, decoder-only

**Parameters:** 6B, 34B variants

**Context Window:** 4K tokens (standard), 200K tokens (extended versions)

**Key Technical Details:**

- Bilingual (English and Chinese)
- Grouped Query Attention
- SwiGLU activation
- Strong performance on reasoning benchmarks
- Extended context versions (Yi-34B-200K)
- Open-source with Apache 2.0 license

**Training:**

- 3 trillion tokens
- High-quality filtered data

**Applications:** Bilingual applications, research, commercial use in China and globally

-----

### 18. Phi-2 (December 2023)

**Developer:** Microsoft Research

**Architecture:** Transformer-based, decoder-only

**Parameters:** 2.7 billion

**Context Window:** 2,048 tokens

**Key Technical Details:**

- Focused on “textbook quality” training data
- Exceptional performance for size
- Outperforms models 25x larger on some benchmarks
- Layer normalization
- Rotary positional embeddings
- Trained on synthetic and filtered data

**Training Philosophy:**

- Quality over quantity in training data
- Emphasis on reasoning and common sense

**Applications:** Research on data quality, edge deployment, efficient inference

-----

### 19. Llama 3.1 (July 2024)

**Developer:** Meta AI

**Architecture:** Enhanced transformer-based, decoder-only

**Parameters:** 8B, 70B, 405B variants

**Context Window:** 128,000 tokens

**Training Data:** 15+ trillion tokens

**Key Technical Details:**

- Largest open-source model (405B)
- Native tool use and function calling
- Multilingual capabilities (8 languages)
- Advanced reasoning and coding
- Grouped Query Attention with improved scaling
- Iterative post-training with RLHF and rejection sampling

**405B Model Specifications:**

- 126 layers
- 128 attention heads
- 16,384 hidden dimensions
- Trained on 15.6T tokens

**Applications:** Advanced reasoning, coding, multilingual tasks, tool use, open-source development

-----

### 20. StarCoder (May 2023)

**Developer:** BigCode (Hugging Face, ServiceNow)

**Architecture:** Transformer-based, decoder-only (GPT-2 style)

**Parameters:** 15.5 billion

**Context Window:** 8,192 tokens

**Training Data:** 1 trillion tokens (The Stack dataset)

**Key Technical Details:**

- Specialized for code generation
- Trained on 80+ programming languages
- Multi-Query Attention
- Fill-in-the-middle capability
- FlashAttention
- Open and responsible AI license (OpenRAIL)

**Training Dataset:**

- The Stack (GitHub permissive licenses)
- Opt-out mechanism for developers

**Notable Variants:**

- StarCoder2 (February 2024) - 3B, 7B, 15B variants

**Applications:** Code generation, code completion, programming assistance, developer tools

-----

### Architecture Patterns and Trends

### Common Architectural Components

1. **Transformer Foundation**: All models use transformer architecture as the base
1. **Decoder-Only Design**: Most modern LLMs use decoder-only architectures
1. **Attention Mechanisms**:
- Multi-Head Attention (standard)
- Multi-Query Attention (MQA) - reduced memory
- Grouped Query Attention (GQA) - balance between MHA and MQA
- Sliding Window Attention - efficient long context
- FlashAttention - memory-efficient computation
1. **Positional Encodings**:
- Rotary Positional Embeddings (RoPE) - most common
- ALiBi (Attention with Linear Biases) - enables context extension
- Learned absolute embeddings (older models)
1. **Activation Functions**:
- SwiGLU - common in modern models
- GELU - earlier models
- ReLU variants
1. **Normalization**:
- RMSNorm - computationally efficient
- LayerNorm - traditional approach
- Pre-normalization vs post-normalization

#### Training Innovations

1. **Mixture of Experts (MoE)**: Sparse activation for efficiency (Mixtral, GPT-4 rumored, Gemini 1.5)
1. **Instruction Tuning**: Fine-tuning on instruction-following datasets
1. **RLHF**: Reinforcement Learning from Human Feedback for alignment
1. **Constitutional AI**: Harmlessness training through AI feedback (Claude)
1. **Synthetic Data**: High-quality synthetic training data (Phi models)

#### Scaling Trends

- **Parameter Growth**: From 7B to 1.76T parameters
- **Context Length**: From 2K to 1M+ tokens
- **Training Data**: From hundreds of billions to 15+ trillion tokens
- **Multimodality**: Integration of vision, audio, and other modalities
- **Efficiency**: Focus on compute-optimal training and inference

-----

### Performance Considerations

#### Benchmark Categories

1. **General Capabilities**: MMLU, HellaSwag, ARC, TruthfulQA
1. **Reasoning**: GSM8K (math), BBH (BigBench Hard)
1. **Coding**: HumanEval, MBPP, CodeContests
1. **Multilingual**: XCOPA, multilingual MMLU
1. **Long Context**: RULER, NeedleInHaystack

#### Deployment Considerations

- **Cloud API**: GPT-4, Claude, Gemini, Cohere
- **Open-Source Self-Hosted**: LLaMA, Mistral, Falcon, Yi
- **Edge Deployment**: Phi-2, Gemini Nano, smaller variants
- **Specialized**: StarCoder (code), Command R+ (RAG)

-----

### Licensing Overview

- **Proprietary/API Only**: GPT-4, GPT-3.5, Claude, Gemini
- **Open Research**: LLaMA 1, some early models
- **Commercial Open-Source**: LLaMA 2/3, Mistral, Mixtral, Falcon, Yi, MPT
- **Collaborative Open**: BLOOM, StarCoder

-----

### Conclusion

The LLM landscape from 2022-2025 shows rapid evolution in capabilities, efficiency, and accessibility. Key trends include:

- Scaling to larger models with better compute efficiency
- Dramatic context window expansion (2K → 1M+ tokens)
- Multimodal integration becoming standard
- Open-source models approaching proprietary performance
- Specialized models for specific domains (code, RAG, multilingual)
- Increased focus on safety, alignment, and responsible AI
- Mixture of Experts for efficient scaling
- Tool use and agentic capabilities becoming mainstream

The field continues to advance rapidly, with new models and capabilities emerging regularly, democratizing access to powerful AI while pushing the boundaries of what’s possible with language models.
