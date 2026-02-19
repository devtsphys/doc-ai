# AI Algorithms, Models & Architectures — Reference Cheat Sheet

-----

## 1. Large Language Models (LLM) / Text Generation

|Model                        |Organization   |Architecture       |Parameters       |Key Strengths              |
|-----------------------------|---------------|-------------------|-----------------|---------------------------|
|★ GPT-4o / GPT-4 Turbo       |OpenAI         |Transformer decoder|~1.8T (MoE, est.)|General reasoning, tool use|
|↑ Claude 3.5 Sonnet / Opus   |Anthropic      |Transformer decoder|Undisclosed      |Safety, long context (200K)|
|★ Llama 3.1 / 3.3            |Meta           |Transformer decoder|8B–405B          |Open weights, fine-tuning  |
|Gemini 1.5 Pro / Ultra       |Google DeepMind|Transformer (MoE)  |Undisclosed      |1M context window          |
|Mistral Large / Mixtral 8x22B|Mistral AI     |MoE Transformer    |141B (MoE)       |Efficient, open weights    |
|Qwen 2.5                     |Alibaba        |Transformer decoder|0.5B–72B         |Multilingual, open weights |
|DeepSeek-V3 / R1             |DeepSeek       |MoE Transformer    |671B (MoE)       |Math, coding, open weights |
|Phi-3 / Phi-4                |Microsoft      |Transformer decoder|3.8B–14B         |Small but capable SLMs     |
|Falcon 180B                  |TII            |Transformer decoder|180B             |Open-access large model    |
|Command R+                   |Cohere         |Transformer decoder|104B             |RAG-optimized              |

### Key Accuracy Metrics (LLMs)

|Benchmark     |What it Measures                    |Top Scores (approx.)         |
|--------------|------------------------------------|-----------------------------|
|MMLU          |Multi-domain knowledge (57 subjects)|GPT-4o ~88%, Claude 3.5 ~89% |
|HumanEval     |Code generation (Python pass@1)     |GPT-4o ~90%, DeepSeek-R1 ~92%|
|GSM8K         |Grade-school math reasoning         |Top models >95%              |
|MATH          |Competition math                    |Top models ~70–80%           |
|HellaSwag     |Commonsense NLI                     |Top models >95%              |
|BIG-Bench Hard|Hard reasoning tasks                |Varies widely                |
|GPQA          |PhD-level Q&A                       |~60–70% top models           |
|MT-Bench      |Multi-turn conversation quality     |1–10 scale                   |

### Core Architectures

- **Transformer (Vaswani et al., 2017)** — Self-attention, positional encoding, feed-forward layers
- **Mixture of Experts (MoE)** — Sparse gating routes tokens to subsets of experts (GPT-4, Mixtral, Gemini)
- **Retrieval-Augmented Generation (RAG)** — Combines parametric LLM with external retrieval index
- **RLHF / DPO / PPO** — Alignment techniques used post-training

-----

## 2. Image Generation

|Model                     |Organization     |Architecture             |Key Strengths                   |
|--------------------------|-----------------|-------------------------|--------------------------------|
|★ DALL-E 3                |OpenAI           |Diffusion + CLIP         |Prompt adherence, text in images|
|★ Stable Diffusion 3 / XL |Stability AI     |Latent Diffusion (DiT)   |Open weights, fine-tunable      |
|↑ Flux.1 (Pro/Dev/Schnell)|Black Forest Labs|Flow Matching Transformer|High realism, text rendering    |
|Midjourney v6             |Midjourney       |Diffusion (proprietary)  |Aesthetic quality               |
|Imagen 3                  |Google           |Cascaded Diffusion       |Photorealism                    |
|Adobe Firefly 3           |Adobe            |Diffusion                |Commercial-safe training        |
|Kandinsky 3               |Sber             |Latent Diffusion         |Multilingual prompts            |

### Key Architectures

- **Latent Diffusion Model (LDM)** — Diffusion in compressed latent space (VAE encoder/decoder)
- **Denoising Diffusion Probabilistic Models (DDPM)** — Forward noise process + learned reverse
- **DDIM / PLMS / DPM-Solver** — Fast sampling schedulers (20–50 steps vs. 1000)
- **Flow Matching** — Continuous normalizing flows, deterministic trajectories (Flux)
- **DiT (Diffusion Transformer)** — Transformer backbone replacing U-Net (SD3, Flux)
- **U-Net** — Encoder-decoder with skip connections, original SD backbone
- **CLIP / T5** — Text encoders for conditioning

### Key Accuracy Metrics

|Metric                          |What it Measures                        |
|--------------------------------|----------------------------------------|
|FID (Fréchet Inception Distance)|Distribution similarity (lower = better)|
|IS (Inception Score)            |Quality + diversity (higher = better)   |
|CLIP Score                      |Text-image alignment                    |
|LPIPS                           |Perceptual similarity                   |
|Human Preference Score (HPS v2) |Human aesthetic ratings                 |
|GenEval                         |Compositional prompt following          |

-----

## 3. Image Classification & Vision

|Model                |Architecture      |Dataset     |Top-1 Accuracy (ImageNet)|
|---------------------|------------------|------------|-------------------------|
|ViT-H/14 (Google)    |Vision Transformer|JFT-3B      |~90.5%                   |
|↑ EVA-02 / InternViT |ViT variants      |Various     |~90%+                    |
|★ ConvNeXt V2-H      |ConvNet           |ImageNet-22K|~88.9%                   |
|EfficientNet V2-L    |NAS ConvNet       |ImageNet-21K|~87.3%                   |
|★ ResNet-50/101/152  |Residual CNN      |ImageNet    |76–80% (baseline)        |
|RegNet               |Regularized CNN   |ImageNet    |~83%                     |
|Swin Transformer V2-G|Shifted Window ViT|ImageNet-22K|~90.2%                   |
|DeiT III             |Data-efficient ViT|ImageNet    |~87.7%                   |

### Key Architectures

- **CNN (LeNet → AlexNet → VGG → ResNet → EfficientNet)** — Convolutional feature hierarchies
- **Vision Transformer (ViT)** — Image patches as tokens, global self-attention
- **Swin Transformer** — Hierarchical ViT with local shifted windows
- **ConvNeXt** — Modernized CNN matching ViT performance

### Key Metrics

|Metric                      |Description                        |
|----------------------------|-----------------------------------|
|Top-1 Accuracy              |Correct class is top prediction    |
|Top-5 Accuracy              |Correct class in top 5 predictions |
|mAP (mean Average Precision)|Detection/segmentation tasks       |
|F1 Score                    |Harmonic mean of precision & recall|
|AUC-ROC                     |Area under ROC curve               |

-----

## 4. Object Detection & Segmentation

|Model                      |Type         |Architecture          |COCO mAP      |Notes                     |
|---------------------------|-------------|----------------------|--------------|--------------------------|
|★ YOLOv8 / YOLO11          |Detection    |CSPNet + Neck         |~53–55%       |Real-time, widely deployed|
|↑ YOLOv9 / YOLOv10         |Detection    |GELAN                 |~55%+         |Improved efficiency       |
|★ RT-DETR                  |Detection    |Transformer           |~54%          |Real-time DETR            |
|DINO / Grounding DINO      |Detection    |DETR-based            |~63%+         |Zero-shot detection       |
|★ Detectron2 (Faster R-CNN)|Detection    |Region proposal       |~42–50%       |Meta’s research framework |
|↑ SAM 2 (Segment Anything) |Segmentation |MAE ViT + streaming   |—             |Zero-shot, video support  |
|Mask R-CNN                 |Instance Seg.|Region proposal       |~39% (mask AP)|Classic baseline          |
|★ DETR                     |Detection    |End-to-end Transformer|~42%          |No NMS needed             |

### Key Architectures

- **Two-stage (R-CNN family)** — Region proposals → RoI classification
- **One-stage (YOLO family)** — Direct grid-based prediction, real-time
- **DETR** — End-to-end detection with transformer encoder-decoder
- **Anchor-free detectors** — FCOS, CenterNet (predict object centers)

### Key Metrics

- **mAP@0.5 / mAP@0.5:0.95** — COCO standard detection metric
- **IoU (Intersection over Union)** — Overlap quality of bounding boxes
- **Panoptic Quality (PQ)** — For panoptic segmentation

-----

## 5. OCR (Optical Character Recognition)

|Model/System             |Type              |Architecture               |Notes                         |
|-------------------------|------------------|---------------------------|------------------------------|
|★ Tesseract 5.x          |Traditional + LSTM|LSTM-based                 |Open-source, 100+ languages   |
|★ PaddleOCR v3           |End-to-end        |PP-OCRv3 (DBNet + SVTR)    |Fast, multilingual, open      |
|↑ TrOCR (Microsoft)      |Transformer       |ViT encoder + GPT-2 decoder|Best on printed/handwritten   |
|EasyOCR                  |End-to-end        |CRAFT + CRNN               |Open-source, 80+ languages    |
|★ Google Cloud Vision OCR|Cloud API         |Proprietary                |High accuracy, handwriting    |
|Azure AI Vision OCR      |Cloud API         |Proprietary                |Form recognition strength     |
|★ Textract (AWS)         |Cloud API         |Proprietary                |Tables, forms extraction      |
|GOT-OCR 2.0              |End-to-end        |ViT + Transformer          |Formulas, charts, multilingual|
|Surya                    |End-to-end        |Transformer                |Open-source, line-level       |
|DocTR                    |End-to-end        |DBNet + CRNN/ViT           |HuggingFace integrated        |

### Key Architectures

- **CRNN** — CNN feature extractor + Bi-LSTM + CTC decoder (classic pipeline)
- **DBNet** — Differentiable Binarization for text detection
- **EAST** — Efficient and Accurate Scene Text detector
- **Attention-based Seq2Seq** — Encoder-decoder for irregular text
- **Transformer OCR (TrOCR, GOT)** — Image transformer + language decoder

### Key Metrics

|Metric                        |Description                                 |
|------------------------------|--------------------------------------------|
|CER (Character Error Rate)    |Edit distance / total chars (lower = better)|
|WER (Word Error Rate)         |Word-level edit distance                    |
|Accuracy (%)                  |Exact character/word matches                |
|F1 (Word-level)               |For word spotting tasks                     |
|NED (Normalized Edit Distance)|Benchmark standard for scene text           |

-----

## 6. Speech Recognition (ASR — Automatic Speech Recognition)

|Model                         |Organization|Architecture                       |WER (LibriSpeech clean)|
|------------------------------|------------|-----------------------------------|-----------------------|
|★ Whisper Large v3            |OpenAI      |Encoder-Decoder Transformer        |~2.7%                  |
|↑ Whisper Large v3 Turbo      |OpenAI      |Distilled Transformer              |~3.0% (4x faster)      |
|★ wav2vec 2.0 / XLS-R         |Meta        |CNN + Transformer (self-supervised)|~1.8% (fine-tuned)     |
|↑ MMS (Massively Multilingual)|Meta        |wav2vec 2.0 based                  |1000+ languages        |
|Conformer / Conformer-CTC     |Google      |Conv + Transformer hybrid          |~1.9%                  |
|HuBERT                        |Meta        |BERT-style self-supervised         |~2.0%                  |
|SpeechBrain                   |Open        |Modular (CRDNN, Conformer)         |~2.5%                  |
|★ AssemblyAI / Deepgram       |Cloud       |Proprietary                        |<3%                    |

### Key Architectures

- **CTC (Connectionist Temporal Classification)** — Alignment-free sequence prediction
- **Encoder-Decoder + Attention** — Seq2seq with cross-attention (Whisper)
- **Self-supervised pretraining** — Masked prediction on raw audio (wav2vec, HuBERT)
- **Conformer** — Convolution + Transformer hybrid for local+global features

### Key Metrics

|Metric                    |Description                        |
|--------------------------|-----------------------------------|
|WER (Word Error Rate)     |Primary ASR metric (lower = better)|
|CER (Character Error Rate)|For character-level languages      |
|RTF (Real-Time Factor)    |Inference speed vs. audio duration |
|MER (Match Error Rate)    |Alternative to WER                 |

-----

## 7. Text-to-Speech (TTS) / Speech Synthesis

|Model                        |Organization|Architecture                  |Notes                      |
|-----------------------------|------------|------------------------------|---------------------------|
|★ ElevenLabs                 |ElevenLabs  |Diffusion + Flow              |Best voice cloning quality |
|↑ Voicebox                   |Meta        |Flow Matching                 |Zero-shot, multilingual    |
|★ XTTS v2                    |Coqui       |Transformer + VQ-VAE          |Open-source, multilingual  |
|SoundStorm                   |Google      |Non-autoregressive Transformer|Fast, high quality         |
|★ OpenAI TTS (tts-1/tts-1-hd)|OpenAI      |Proprietary                   |Commercial quality         |
|VALL-E / VALL-E X            |Microsoft   |Neural codec LM (EnCodec)     |Few-shot voice cloning     |
|↑ CosyVoice 2                |Alibaba     |Flow Matching                 |Multilingual, open         |
|Bark                         |Suno AI     |GPT-based audio LM            |Emotions, non-speech sounds|
|StyleTTS 2                   |MIT         |Diffusion + Style Transfer    |Open-source SOTA           |
|VITS / VITS2                 |Kakao       |Flow-based VAE + GAN          |Fast, expressive           |

### Key Architectures

- **Neural Codec Language Models (VALL-E)** — Treat speech tokens as LM vocabulary
- **Flow Matching / Normalizing Flows** — Continuous bijective transformations
- **Diffusion-based** — Iterative denoising in mel-spectrogram or waveform space
- **VITS (Variational Inference + GAN)** — End-to-end, latent diffusion + adversarial
- **Vocoders** — WaveNet, WaveGlow, HiFi-GAN (mel-spec → waveform)

### Key Metrics

|Metric                  |Description                    |
|------------------------|-------------------------------|
|MOS (Mean Opinion Score)|Human naturalness rating (1–5) |
|UTMOS                   |Automatic MOS predictor        |
|WER on TTS output       |Intelligibility via ASR        |
|Speaker Similarity (SIM)|Cosine similarity of embeddings|
|DNSMOS                  |Noise/quality assessment       |

-----

## 8. Machine Translation (MT)

|Model                    |Organization      |Architecture             |Notes                         |
|-------------------------|------------------|-------------------------|------------------------------|
|★ DeepL Translator       |DeepL             |Transformer (proprietary)|Best European language quality|
|★ Google Translate / GNMT|Google            |Transformer + LLM        |200+ languages                |
|NLLB-200                 |Meta              |Transformer (200 langs)  |Open-source, low-resource     |
|M2M-100                  |Meta              |Transformer              |Many-to-many translation      |
|mBART / mT5              |Meta / Google     |Multilingual BART/T5     |Fine-tunable baselines        |
|↑ GPT-4 / Claude 3.5     |OpenAI / Anthropic|LLM                      |Often SOTA on high-resource   |
|Helsinki-NLP Opus-MT     |U Helsinki        |Lightweight Transformers |Open, 1000+ language pairs    |

### Key Metrics

|Metric                     |Description                             |
|---------------------------|----------------------------------------|
|BLEU                       |N-gram precision vs. reference (0–100)  |
|chrF / chrF++              |Character-level F-score                 |
|COMET                      |Neural MT evaluation (human correlation)|
|TER (Translation Edit Rate)|Edits to make hypothesis match reference|
|METEOR                     |Precision + recall with synonymy        |

-----

## 9. Time-Series Forecasting

|Model                            |Type       |Architecture                    |Best For                         |
|---------------------------------|-----------|--------------------------------|---------------------------------|
|★ Prophet                        |Statistical|Additive decomposition          |Business forecasting, seasonality|
|★ ARIMA / SARIMA                 |Statistical|Autoregressive + MA             |Univariate, short series         |
|N-BEATS                          |DL         |Backward/forward residual blocks|Univariate, M4 competition winner|
|N-HiTS                           |DL         |Hierarchical interpolation      |Long-horizon, multi-scale        |
|★ PatchTST                       |Transformer|Patched self-attention          |Multivariate, long-horizon       |
|↑ Chronos                        |Amazon     |LLM (T5-based)                  |Zero-shot forecasting            |
|↑ TimesFM                        |Google     |Decoder Transformer             |Foundation model for TS          |
|TFT (Temporal Fusion Transformer)|Google     |Transformer + LSTM + attention  |Interpretable, multi-horizon     |
|LSTM / GRU                       |DL         |Recurrent networks              |Sequential, variable-length      |
|LightGBM / XGBoost               |ML         |Gradient Boosted Trees          |Tabular features + lag features  |
|DeepAR                           |Amazon     |Autoregressive RNN              |Probabilistic forecasts          |
|TimeMixer                        |—          |MLP-Mixer based                 |Efficient multivariate           |

### Key Metrics

|Metric                        |Description                       |
|------------------------------|----------------------------------|
|MAE (Mean Absolute Error)     |Average absolute deviation        |
|MAPE (Mean Absolute % Error)  |Scale-independent error           |
|RMSE (Root Mean Squared Error)|Penalizes large errors            |
|SMAPE (Symmetric MAPE)        |Symmetric percentage error        |
|CRPS                          |Probabilistic forecast calibration|
|sMAPE                         |M4/M5 competition standard        |

-----

## 10. Tabular Data / Classical ML

|Model              |Type             |Typical Use              |Strengths                        |
|-------------------|-----------------|-------------------------|---------------------------------|
|★ XGBoost          |Gradient Boosting|Classification/Regression|Speed, regularization            |
|★ LightGBM         |Gradient Boosting|Classification/Regression|Fast, large datasets             |
|CatBoost           |Gradient Boosting|Categorical-heavy data   |Native categoricals              |
|★ Random Forest    |Ensemble Bagging |General purpose          |Robust, interpretable            |
|↑ TabNet           |Deep Learning    |Tabular classification   |Attention-based feature selection|
|↑ TabPFN           |Meta-learning    |Small tabular datasets   |In-context learning, <1k rows    |
|SVM (RBF kernel)   |Kernel method    |Classification           |High-dimensional, small data     |
|Logistic Regression|Linear model     |Binary classification    |Fast, interpretable baseline     |

### Key Metrics

|Task                 |Primary Metrics                                |
|---------------------|-----------------------------------------------|
|Binary Classification|Accuracy, AUC-ROC, F1, Precision, Recall, MCC  |
|Multi-class          |Macro/Micro F1, Cohen’s Kappa, Confusion Matrix|
|Regression           |MAE, MSE, RMSE, R², MAPE                       |
|Ranking              |NDCG, MAP, MRR                                 |

-----

## 11. Recommendation Systems

|Model                               |Architecture                |Notes                           |
|------------------------------------|----------------------------|--------------------------------|
|★ Collaborative Filtering (ALS/SVD) |Matrix Factorization        |Classic, implicit feedback      |
|Neural Collaborative Filtering (NCF)|MLP + embeddings            |Replaces dot-product with MLP   |
|★ Wide & Deep                       |Linear + DNN                |Google Play App store           |
|DeepFM                              |FM + DNN                    |Implicit feature interactions   |
|BERT4Rec                            |Transformer (BERT)          |Sequential recommendation       |
|★ SASRec                            |Transformer (autoregressive)|Self-attentive sequential       |
|TwoTower                            |Dual encoder                |Large-scale retrieval (YouTube) |
|LightGCN                            |Graph Neural Network        |Collaborative filtering on graph|

### Key Metrics

- **Precision@K, Recall@K, NDCG@K, Hit Rate@K** — Ranking quality
- **MRR (Mean Reciprocal Rank)** — First relevant result rank
- **Coverage, Diversity** — Catalog usage
- **CTR / Conversion Rate** — Business metrics

-----

## 12. Video Generation & Understanding

|Model               |Organization|Architecture               |Notes                |
|--------------------|------------|---------------------------|---------------------|
|↑ Sora              |OpenAI      |DiT (Diffusion Transformer)|Up to 1-min HD video |
|↑ Veo 2             |Google      |Diffusion Transformer      |4K, cinematic quality|
|★ Runway Gen-3 Alpha|Runway      |Diffusion                  |Commercial standard  |
|CogVideoX           |Zhipu AI    |DiT                        |Open weights         |
|Wan 2.1             |Alibaba     |Diffusion Transformer      |Open-source, leading |
|AnimateDiff         |—           |Motion module + SD         |Open-source animation|

### Video Understanding

|Model                 |Task                                      |
|----------------------|------------------------------------------|
|VideoMAE / VideoMAE V2|Video classification (pre-training)       |
|InternVideo2          |Multi-task video understanding            |
|★ CLIP4Clip           |Video-text retrieval                      |
|TimeSformer           |Video classification via divided attention|

### Key Metrics

- **FVD (Fréchet Video Distance)** — Video generation quality
- **IS (Inception Score)** — Clip-level quality
- **Top-1/5 Accuracy** — Video classification (Kinetics-400/600/700)
- **mAP** — Action detection

-----

## 13. Multimodal Models (Vision-Language)

|Model                   |Organization   |Architecture          |Capabilities                         |
|------------------------|---------------|----------------------|-------------------------------------|
|★ GPT-4V / GPT-4o       |OpenAI         |LLM + vision encoder  |Image/video understanding, generation|
|★ Claude 3.5 Sonnet/Opus|Anthropic      |LLM + vision encoder  |Complex visual reasoning             |
|★ Gemini 1.5 Pro        |Google         |Multimodal Transformer|Video, audio, image, text            |
|↑ LLaVA-1.6 / LLaVA-Next|Various        |CLIP + Vicuna/Llama   |Open-source VQA                      |
|InternVL 2.5            |Shanghai AI Lab|ViT + LLM             |SOTA open-source VLM                 |
|Qwen-VL / Qwen2-VL      |Alibaba        |ViT + Qwen            |Multilingual, OCR-capable            |
|Phi-3.5-Vision          |Microsoft      |SigLIP + Phi          |Efficient SLM                        |
|↑ Pixtral Large         |Mistral        |ViT + Mistral         |Open weights                         |

### Key Benchmarks

|Benchmark       |Measures                                 |
|----------------|-----------------------------------------|
|VQAv2           |Visual question answering                |
|TextVQA         |Text in images                           |
|DocVQA          |Document understanding                   |
|MMMU            |Multi-discipline multimodal understanding|
|MMBench / MMStar|Comprehensive VLM evaluation             |
|ChartQA         |Chart/figure understanding               |

-----

## 14. Named Entity Recognition (NER) & NLP Tasks

|Model                  |Architecture               |Notes                    |
|-----------------------|---------------------------|-------------------------|
|★ BERT / RoBERTa       |Transformer encoder        |Fine-tuned NER baseline  |
|★ SpanBERT             |Span-level BERT            |Coreference, NER         |
|DeBERTa V3             |Disentangled attention BERT|Strong NER/classification|
|★ GLiNER               |Generalist NER             |Zero-shot entity types   |
|Flair                  |LSTM + char embeddings     |Sequence tagging         |
|spaCy (en_core_web_trf)|Transformer-based          |Production NLP pipeline  |

### NLP Task Benchmarks

|Task              |Benchmark    |Metric     |
|------------------|-------------|-----------|
|NER               |CoNLL-2003   |F1         |
|POS Tagging       |Penn Treebank|Accuracy   |
|Sentiment         |SST-2        |Accuracy   |
|NLI               |MNLI / SNLI  |Accuracy   |
|Question Answering|SQuAD 2.0    |EM / F1    |
|Summarization     |CNN/DM, XSum |ROUGE-1/2/L|
|Coreference       |CoNLL-2012   |Avg F1     |

-----

## 15. Code Generation

|Model                   |Organization|HumanEval (pass@1)|Notes                   |
|------------------------|------------|------------------|------------------------|
|↑ Claude 3.5 Sonnet     |Anthropic   |~92%              |Best overall coding     |
|↑ GPT-4o                |OpenAI      |~90%              |Broad language support  |
|★ GitHub Copilot (GPT-4)|Microsoft   |—                 |IDE-integrated          |
|DeepSeek-Coder-V2       |DeepSeek    |~90.2%            |Open weights SOTA       |
|★ CodeLlama 70B         |Meta        |~53%              |Open, instruction-tuned |
|Qwen2.5-Coder-32B       |Alibaba     |~92.7%            |Open-source leader      |
|StarCoder2              |BigCode     |~46% (15B)        |Open, multi-language    |
|WizardCoder             |WizardLM    |~73%              |EVOL-Instruct fine-tuned|

### Code Benchmarks

- **HumanEval / HumanEval+** — Python function synthesis (pass@1/10/100)
- **MBPP** — Mostly Basic Python Problems
- **SWE-Bench** — Real GitHub issue resolution
- **LiveCodeBench** — Contamination-free coding benchmark
- **BigCodeBench** — Library/API-level tasks

-----

## 16. Anomaly Detection

|Model                     |Type        |Architecture         |Notes                  |
|--------------------------|------------|---------------------|-----------------------|
|★ Isolation Forest        |Unsupervised|Tree ensemble        |Fast, high-dimensional |
|LOF (Local Outlier Factor)|Unsupervised|Density-based        |Local context aware    |
|One-Class SVM             |Unsupervised|Kernel SVM           |Classic baseline       |
|Autoencoder               |DL          |Encoder-Decoder      |Reconstruction error   |
|VAE (Variational AE)      |DL          |Latent variable model|Probabilistic          |
|★ PatchCore               |DL (vision) |Memory bank + ViT    |MVTec SOTA             |
|USAD                      |DL          |Dual AE + GAN        |Time-series anomaly    |
|TranAD                    |DL          |Transformer          |Multivariate TS anomaly|

### Key Metrics

- **AUC-ROC, AUC-PR** — Threshold-free evaluation
- **F1 @ best threshold** — Classification-style
- **AUROC** — MVTec AD benchmark standard
- **PA (Point-Adjust) F1** — Time-series anomaly standard

-----

## 17. Reinforcement Learning (RL)

|Algorithm                           |Type       |Use Case                    |
|------------------------------------|-----------|----------------------------|
|★ PPO (Proximal Policy Optimization)|On-policy  |LLM RLHF, robotics, games   |
|SAC (Soft Actor-Critic)             |Off-policy |Continuous control, robotics|
|DQN / Double DQN                    |Off-policy |Discrete action spaces      |
|TD3 (Twin Delayed DDPG)             |Off-policy |Continuous control          |
|A3C / A2C                           |On-policy  |Parallel environments       |
|★ AlphaZero / MuZero                |Model-based|Board games, planning       |
|GRPO                                |On-policy  |LLM reasoning (DeepSeek-R1) |
|Dreamer V3                          |Model-based|World model RL              |

### Key Metrics

- **Cumulative Reward** — Primary RL objective
- **Sample Efficiency** — Steps to reach threshold performance
- **Human-normalized Score** — Atari benchmark standard
- **ELO / Glicko** — Game-playing agents

-----

## 18. Embedding Models

|Model                   |Organization|Dimensions|MTEB Score|Notes                   |
|------------------------|------------|----------|----------|------------------------|
|↑ text-embedding-3-large|OpenAI      |3072      |~64.6     |Top commercial          |
|★ E5-Mistral-7B         |Microsoft   |4096      |~66.6     |Open SOTA               |
|↑ GTE-Qwen2-7B          |Alibaba     |3584      |~72.0     |Current open leader     |
|★ BGE-M3                |BAAI        |1024      |~66.0     |Multilingual, multi-task|
|NV-Embed-v2             |NVIDIA      |4096      |~72.3     |MTEB leader             |
|★ BAAI/bge-large        |BAAI        |1024      |~63.6     |Efficient baseline      |
|all-MiniLM-L6           |SBERT       |384       |~56.3     |Fast, lightweight       |
|Cohere Embed v3         |Cohere      |1024      |~64.5     |Multilingual            |

### Key Metric

- **MTEB (Massive Text Embedding Benchmark)** — 56 tasks across 8 categories
  - Tasks: Retrieval, STS, Classification, Clustering, Reranking, etc.

-----

## Quick Reference: Architecture Building Blocks

|Component                        |Description                                           |Used In            |
|---------------------------------|------------------------------------------------------|-------------------|
|Multi-Head Self-Attention        |Parallel attention heads, token-to-token relationships|Transformers       |
|Positional Encoding (RoPE, ALiBi)|Inject sequence order information                     |LLMs, ViTs         |
|Layer Normalization (Pre/Post)   |Training stability                                    |All Transformers   |
|Feed-Forward Network (FFN/MLP)   |Point-wise transformation                             |All Transformers   |
|Cross-Attention                  |Query from one, K/V from another sequence             |Seq2Seq, Diffusion |
|Flash Attention 2/3              |Memory-efficient exact attention                      |Modern LLMs        |
|KV Cache                         |Inference speedup for autoregressive models           |LLM inference      |
|LoRA / QLoRA                     |Low-rank adaptation for fine-tuning                   |LLM fine-tuning    |
|Grouped Query Attention (GQA)    |Fewer KV heads for efficiency                         |Llama 3, Mistral   |
|Sliding Window Attention         |Local+global hybrid                                   |Mistral, Longformer|
|Speculative Decoding             |Draft model + verifier for faster inference           |LLM serving        |
|GELU / SiLU / SwiGLU             |Activation functions                                  |Modern LLMs        |

-----

## Benchmark Quick Reference Table

|Domain              |Primary Benchmark    |Metric                |
|--------------------|---------------------|----------------------|
|LLM General         |MMLU, HELM, BIG-Bench|Accuracy              |
|LLM Reasoning       |MATH, GSM8K, ARC     |Accuracy              |
|LLM Code            |HumanEval, SWE-Bench |pass@1, % resolved    |
|Image Classification|ImageNet-1K          |Top-1 Accuracy        |
|Object Detection    |COCO                 |mAP@[0.5:0.95]        |
|Segmentation        |COCO Panoptic        |PQ                    |
|OCR                 |IIIT5K, SVT, CUTE80  |Word Accuracy         |
|ASR                 |LibriSpeech          |WER                   |
|TTS                 |LJSpeech             |MOS                   |
|Machine Translation |WMT                  |BLEU / COMET          |
|Forecasting         |M4 / M5 Competition  |sMAPE / WRMSSE        |
|Embeddings          |MTEB                 |Average Score         |
|Video Classification|Kinetics-400         |Top-1 Accuracy        |
|VLM                 |MMMU, VQAv2          |Accuracy              |
|RL (Games)          |Atari 57             |Human-normalized Score|

-----

*Last updated: February 2025 — Models and benchmarks evolve rapidly; verify against official leaderboards (Papers With Code, LMSYS Chatbot Arena, MTEB, Open LLM Leaderboard).*