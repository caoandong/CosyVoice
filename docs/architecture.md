# CosyVoice 3.0 Architecture Deep Dive

This document explains the CosyVoice 3.0 architecture in two layers at the same time:

1. The paper-level system described in `paper.pdf`.
2. The concrete implementation released in this repository.

That distinction matters:

- The paper describes the supervised multi-task speech tokenizer, DiffRO post-training, large-scale data scaling, and speaker fine-tuning strategy.
- This repository exposes the released CosyVoice 3 runtime and supervised training recipe built around `CosyVoice3LM`, `CausalMaskedDiffWithDiT`, and `CausalHiFTGenerator`.
- The tokenizer is consumed here as `speech_tokenizer_v3.onnx`; the tokenizer training code itself is not in this repo.
- The paper's DiffRO post-training is described in the paper, but the exact open-source CosyVoice 3 DiffRO training loop is not fully present here. The repo does include DPO hooks and CosyVoice2 GRPO examples, which are related but not identical.

All exact module sizes below come from `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`, which corresponds to the released 0.5B-class CosyVoice 3 implementation. The paper also describes a larger 1.5B variant with the same overall pipeline.


## 1. The shortest accurate summary

CosyVoice 3 is a two-stage LLM TTS system:

1. A Qwen-based autoregressive language model turns text and optional prompt context into **discrete speech tokens** at **25 Hz**.
2. A causal conditional flow-matching model with a DiT backbone turns those speech tokens into **mel spectrogram frames** at **50 Hz**.
3. A causal HiFT vocoder turns the mel spectrogram into a **24 kHz waveform**.

The prompt audio is split into three different kinds of conditioning because each kind solves a different problem:

- **Prompt speech tokens**: help the LLM continue speaking in the same semantic/prosodic style space.
- **Prompt mel frames**: help the flow decoder match local acoustic texture and smoothly continue from the prompt.
- **Speaker embedding**: gives the decoder a compact global identity vector.


## 2. Big-picture architecture

```text
                              CosyVoice 3.0 Inference Stack

   Target text / instruction
             |
             v
   +-----------------------+
   | Qwen tokenizer +      |
   | special control tags  |
   +-----------------------+
             |
             v
      text tokens [1, L_text]


   Prompt wav ---------------------------------------------------------------+
      |                                                                       |
      |                                                                       |
      | 16 kHz                                                                | 24 kHz
      v                                                                       v
   +-----------------------+                                        +-----------------------+
   | Speech tokenizer v3   |                                        | Mel extractor         |
   | (ONNX, 25 Hz tokens)  |                                        | (80 bins, 50 Hz)      |
   +-----------------------+                                        +-----------------------+
      |                                                                       |
      v                                                                       v
   prompt speech tokens [1, L_prompt_tok]                              prompt mel [1, L_prompt_mel, 80]
      |                                                                       |
      |                                                                       |
      | 16 kHz                                                                |
      v                                                                       |
   +-----------------------+                                                  |
   | CAMPPlus speaker      |                                                  |
   | embedding             |                                                  |
   +-----------------------+                                                  |
      |                                                                       |
      v                                                                       |
   speaker embedding [1, 192]                                                 |
      |                                                                       |
      +-----------------------------+-----------------------------------------+
                                    |
                                    v
   +--------------------------------------------------------------------------------------+
   | Stage 1: CosyVoice3LM (Qwen-based autoregressive speech-token LM)                    |
   |                                                                                      |
   | Input sequence: [SOS] [instruction/prompt text + target text] [TASK] [prompt tokens] |
   | Output: generated speech tokens [1, L_gen_tok] at 25 Hz                              |
   +--------------------------------------------------------------------------------------+
                                    |
                                    v
   +--------------------------------------------------------------------------------------+
   | Stage 2: CausalMaskedDiffWithDiT                                                     |
   |                                                                                      |
   | Inputs: prompt+generated speech tokens, prompt mel prefix, speaker embedding         |
   | Output: generated mel [1, 80, 2 * L_gen_tok] at 50 Hz                               |
   +--------------------------------------------------------------------------------------+
                                    |
                                    v
   +--------------------------------------------------------------------------------------+
   | Stage 3: CausalHiFTGenerator                                                         |
   |                                                                                      |
   | Inputs: mel -> predicted f0 -> harmonic source + ISTFT decoder                       |
   | Output: waveform [1, 960 * L_gen_tok] at 24 kHz                                      |
   +--------------------------------------------------------------------------------------+
```

The key timing ratios are:

- speech token rate: `25 tokens / second`
- mel frame rate: `50 frames / second`
- waveform sample rate: `24000 samples / second`

So one speech token corresponds to:

- `1 / 25 = 40 ms`
- `2` mel frames
- `960` waveform samples at 24 kHz


## 3. Paper view vs repo view

### 3.1 Paper view

The paper presents CosyVoice 3 as a scaled-up, post-trained TTS system with:

- a new multi-task supervised FSQ speech tokenizer built on MinMo
- zero-shot LM + CFM pretraining
- DiffRO post-training
- pronunciation inpainting
- text-normalization self-training
- instruction-following training
- speaker fine-tuning with capability transfer

### 3.2 Repo view

This repo implements the production model stack as:

- `cosyvoice/llm/llm.py` -> `CosyVoice3LM`
- `cosyvoice/flow/flow.py` -> `CausalMaskedDiffWithDiT`
- `cosyvoice/flow/flow_matching.py` -> `CausalConditionalCFM`
- `cosyvoice/flow/DiT/dit.py` -> `DiT`
- `cosyvoice/hifigan/generator.py` -> `CausalHiFTGenerator`
- `cosyvoice/cli/frontend.py` -> runtime preprocessing
- `cosyvoice/cli/model.py` -> runtime orchestration and streaming
- `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` -> exact released dimensions

This means:

- the paper explains why the system exists in this shape
- the repo shows exactly how the released model runs


## 4. Tensor ledger

The easiest way to stay oriented is to track the core tensors and their shapes.

### 4.1 Symbols

- `B`: batch size
- `L_txt`: target text token length
- `L_ins`: instruction or prompt-text token length
- `L_prompt_tok`: prompt speech token length
- `L_gen_tok`: generated speech token length
- `L_tok_total = L_prompt_tok + L_gen_tok`
- `L_prompt_mel = 2 * L_prompt_tok`
- `L_gen_mel = 2 * L_gen_tok`
- `L_mel_total = 2 * L_tok_total`
- `L_wav = 480 * L_gen_mel = 960 * L_gen_tok`

### 4.2 Main runtime tensors

| Tensor | Shape | Where it appears | Meaning |
| --- | --- | --- | --- |
| `text` | `[1, L_txt]` | frontend -> LLM | tokenized target text |
| `prompt_text` | `[1, L_ins]` | frontend -> LLM | either prompt transcript or natural-language instruction |
| `llm_prompt_speech_token` | `[1, L_prompt_tok]` | frontend -> LLM | prompt speech tokens used by LM when applicable |
| `flow_prompt_speech_token` | `[1, L_prompt_tok]` | frontend -> flow | prompt speech tokens used by token-to-mel model |
| `prompt_speech_feat` | `[1, L_prompt_mel, 80]` | frontend -> flow | prompt mel prefix |
| `embedding` | `[1, 192]` | frontend -> flow | speaker embedding from CAMPPlus |
| LM hidden sequence | `[1, L_seq, 896]` | CosyVoice3LM | Qwen hidden states |
| generated speech tokens | `[1, L_gen_tok]` | LLM output | 25 Hz discrete speech representation |
| token embedding | `[1, L_tok_total, 80]` | flow input | embedding of prompt+generated speech tokens |
| `mu` after repeat | `[1, L_mel_total, 80]` | flow input | token-conditioned frame-level conditioning |
| `cond` | `[1, L_mel_total, 80]` | flow input | prompt mel in prefix, zeros elsewhere |
| flow output mel | `[1, 80, L_gen_mel]` | flow output | generated mel spectrogram |
| vocoder output wav | `[1, L_wav]` | final output | synthesized speech |

### 4.3 Important fixed dimensions from the released config

- speech token vocabulary: `6561`
- LM hidden size: `896`
- speaker embedding size: `192`
- flow token embedding size: `80`
- DiT hidden size: `1024`
- DiT depth: `22`
- DiT heads: `16`
- token-to-mel ratio: `2`
- sample rate: `24000`

The vocabulary size `6561` is consistent with an FSQ codebook of size `3^8`, which matches the paper's finite-scalar-quantization framing.


## 5. Component-by-component deep dive

### 5.1 Speech tokenizer

#### What it is for

The speech tokenizer is the interface between raw audio and the discrete sequence space used by the LLM.

It answers this question:

> "What compact token sequence should represent this audio so that an LLM can model it autoregressively?"

#### Paper design

In the paper, the tokenizer is trained by inserting an FSQ bottleneck into MinMo:

```text
speech
  |
  v
Voice Encoder 1 (first 12 Transformer blocks)
  |
  v
intermediate hidden states H
  |
  v
Projdown -> bounded rounding / FSQ -> Projup
  |
  v
Voice Encoder 2 + MinMo LLM
  |
  v
multi-task supervision:
  ASR, LID, SER, AED, speaker analysis
```

The intuition is:

- MinMo already knows a lot about speech.
- FSQ forces that knowledge through a small discrete bottleneck.
- Multi-task supervision makes the bottleneck preserve not only lexical content but also language, emotion, vocal events, and speaker-related information.

That is exactly why CosyVoice 3 speech tokens are more expressive than "just ASR tokens".

#### Quantization idea

The paper's tokenizer does:

1. project hidden states down into a low-rank FSQ space
2. quantize each dimension with bounded rounding
3. map the quantized vector to a single discrete index

Conceptually:

```text
continuous hidden state H
   -> low-rank continuous code
   -> per-dimension scalar quantization
   -> one discrete token id
```

#### Repo runtime path

At inference time, the repo does not retrain this tokenizer. It loads a frozen ONNX tokenizer:

- `speech_tokenizer_v3.onnx`
- prompt audio is resampled to `16 kHz`
- Whisper-style `128`-bin log-mel features are extracted
- the ONNX model returns discrete speech tokens

In `cosyvoice/cli/frontend.py`, the flow is:

```text
prompt wav
  -> load at 16 kHz
  -> whisper.log_mel_spectrogram(..., n_mels=128)
  -> speech_tokenizer_v3.onnx
  -> prompt speech tokens
```

The tokenizer runs at `25 Hz`, so:

- `1 second of speech ~= 25 tokens`
- `4 seconds of prompt ~= 100 tokens`

#### Why the tokenizer matters so much

CosyVoice 3 is only as good as its discrete interface.

If the speech tokens lose:

- emotion
- speaking style
- language identity
- pronunciation detail
- vocal events like breath/laughter

then the LLM can never put them back.

So the tokenizer is not just preprocessing. It is the information bottleneck that defines what the rest of the system is allowed to know.


### 5.2 Text tokenizer and control vocabulary

#### What it is for

The text side is not plain BPE only. CosyVoice 3 deliberately extends the text vocabulary so it can control more than just lexical content.

#### Repo implementation

`CosyVoice3Tokenizer` extends the Qwen tokenizer with:

- `<|endofprompt|>`
- fine-grained vocal tags like `[breath]`, `[laughter]`
- emphasis tags like `<strong> ... </strong>`
- many English phoneme tags
- many Chinese pinyin tags

This directly reflects two paper goals:

1. **instruction following**
2. **pronunciation inpainting**

#### Why `<|endofprompt|>` matters

For CosyVoice 3, natural-language instructions are prepended to the synthesis text, and `<|endofprompt|>` marks where instruction ends and synthesis content begins.

The implementation enforces this:

- `CosyVoice3LM.inference` asserts that `<|endofprompt|>` is present

That design is important because the model needs a hard boundary between:

- meta-instruction
- actual content to speak

Without that boundary, instruction tokens and content tokens would mix ambiguously.


### 5.3 CosyVoice3LM: the text-to-speech-token language model

#### Core idea

The LLM does not generate waveforms or mels.

It generates **speech tokens**.

So its job is:

```text
text / instruction / prompt context
    ->
discrete speech tokens
```

This is the content planner and coarse prosody planner of the system.

#### Concrete implementation

The released config uses:

- `CosyVoice3LM`
- backed by `Qwen2Encoder`
- hidden size `896`
- speech token vocabulary `6561`
- output head size `6561 + 200`

Why `+200`?

Because the speech embedding table also reserves extra special IDs, including:

- `SOS`
- `EOS`
- `TASK`
- `FILL`
- other reserved stop IDs

Only IDs in the true speech-token range are passed downstream as acoustic tokens.

#### Important architectural difference from CosyVoice 1 style models

CosyVoice 3 does **not** feed a speaker embedding into the LLM.

That is visible in the code:

- `CosyVoice3LM.forward` uses text tokens, instruction tokens, and speech tokens
- it does not consume the `embedding` tensor
- speaker identity is enforced mainly through prompt speech tokens and downstream flow conditioning

This is a strong architectural choice:

- the LLM focuses on content/prosody token modeling
- the flow/vocoder stack handles acoustic realization and identity anchoring

#### LLM input sequence at inference

For standard zero-shot inference, the LLM sees:

```text
[SOS]
[prompt_text + target_text]
[TASK]
[prompt_speech_tokens]
```

with embeddings:

- text tokens use Qwen token embeddings
- `SOS` and `TASK` use the speech embedding table
- prompt speech tokens use the speech embedding table

In tensor form:

```text
text_emb                : [1, L_ins + L_txt, 896]
sos_emb                 : [1, 1, 896]
task_id_emb             : [1, 1, 896]
prompt_speech_token_emb : [1, L_prompt_tok, 896]

LM input:
[1, 1 + (L_ins + L_txt) + 1 + L_prompt_tok, 896]
```

#### Natural-language instruction handling

In CosyVoice 3, "prompt text" is overloaded in a useful way:

- in zero-shot cloning, it can be the transcript of the prompt speech
- in instruction mode, it can be an instruction string ending with `<|endofprompt|>`

The LM does not care which one it is. It only sees a text prefix to condition generation.

#### Training sequence formats: unistream and bistream

This part is easy to miss, but it is one of the most important design ideas in the repo.

The training code can construct two different sequence formats.

#### Unistream format

```text
[SOS] [instruction] [text] [TASK] [speech tokens] [EOS]
```

This is the normal left-to-right setup.

#### Bistream format

The model can also interleave chunks of text tokens and speech tokens using `mix_ratio = [5, 15]`:

```text
[SOS]
[instruction]
[text x 5] [speech x 15] [FILL]
[text x 5] [speech x 15] [FILL]
...
[remaining text] [TASK] [remaining speech] [EOS]
```

This is what enables incremental text-input decoding for streaming text generators.

Intuition:

- the model learns that it does not need the full text before starting speech-token generation
- it can consume text and emit speech in alternating chunks

So the system is not only "streaming audio output". It is also capable of **bi-streaming** text-to-audio behavior.

#### Sampling

During inference, the LM uses repetition-aware sampling (`ras_sampling`):

- top-p sampling
- top-k restriction
- repetition check over a short window

This is a stability patch for long autoregressive acoustic-token generation. Without it, speech-token LMs can get trapped in loops or repeated acoustic patterns.

#### Runtime suppression of silence collapse

`CosyVoice3Model` also keeps a list of "silent/breath-like" token IDs and caps long consecutive runs.

This is a pragmatic runtime safeguard:

- the LM is allowed to output silence-like tokens
- it is not allowed to get stuck outputting them forever


### 5.4 Token-to-mel: CausalMaskedDiffWithDiT

#### What this stage does

The LLM output is still too coarse to synthesize directly.

So the second stage turns speech tokens into actual acoustic frames:

```text
speech tokens -> mel spectrogram
```

This stage is where:

- local acoustic detail
- voice texture
- prompt continuation smoothness
- frame-level timing

are realized.

#### Why CosyVoice 3 removed the old length regulator

The paper explicitly says CosyVoice 3 removes the complicated text encoder and length regularization module used earlier.

The reason is visible in the code:

- speech tokens already live in a speech-aligned space
- the rate mismatch is fixed and simple: `25 Hz tokens -> 50 Hz mel`
- so the model just repeats each token representation twice

In other words:

```text
one speech token -> two mel frames
```

That is much simpler than learning text-to-frame duration from scratch.

#### Step-by-step flow input construction

For inference, the flow module does this:

```text
1. concatenate prompt speech tokens + generated speech tokens
2. embed token ids with an 80-d embedding table
3. apply a small pre-lookahead conv module
4. repeat each token state 2 times to reach mel frame rate
5. build a conditioning mel tensor whose prefix is the prompt mel and whose suffix is zeros
6. project speaker embedding 192 -> 80
7. run conditional flow matching / DiT to sample mel frames
8. discard the prompt prefix and keep only generated mel
```

#### Exact shapes through the flow stack

```text
prompt_token + gen_token         : [1, L_tok_total]
token embedding                  : [1, L_tok_total, 80]
after PreLookaheadLayer          : [1, L_tok_total, 80]
after repeat_interleave(x2)      : [1, L_mel_total, 80]

prompt mel prefix                : [1, L_prompt_mel, 80]
cond full tensor                 : [1, L_mel_total, 80]
speaker embedding                : [1, 192] -> linear -> [1, 80]

decoder input mu                 : [1, 80, L_mel_total]
decoder cond                     : [1, 80, L_mel_total]
decoder output full mel          : [1, 80, L_mel_total]
cropped generated mel            : [1, 80, L_gen_mel]
```

#### The pre-lookahead layer

`PreLookaheadLayer` is a small causal-ish convolutional refinement block that peeks a few tokens into the future.

In the released config:

- `pre_lookahead_len = 3`

So when streaming, each chunk can use a tiny amount of right context without needing the entire future utterance.

That is the compromise:

- enough future context to avoid obvious chunk boundaries
- not enough future context to destroy streaming

#### Why the flow has three conditioning inputs

The flow decoder receives:

- `x`: the noised mel it is denoising
- `mu`: the token-conditioned frame-level prior
- `cond`: prompt mel prefix
- `spks`: speaker embedding

Each one solves a different problem:

- `mu` says what should be spoken
- `cond` says how the beginning should acoustically continue from the prompt
- `spks` says whose voice it should sound like

This separation is cleaner than trying to force all acoustic identity and prompt continuity into the LM alone.


### 5.5 Conditional flow matching and the DiT backbone

#### What flow matching is doing here

The target mel is `x1`.

Training samples a noise tensor `z`, a random time `t`, and forms an interpolated state `y`.

The model predicts the velocity field needed to move `y` toward the real mel.

In code form:

```text
y = (1 - (1 - sigma_min) * t) * z + t * x1
u = x1 - (1 - sigma_min) * z
pred = estimator(y, mask, mu, t, spks, cond)
loss = MSE(pred, u)
```

So the network is not directly predicting "the clean mel".
It is predicting how the current noisy sample should move.

#### Why that is a good fit here

This gives CosyVoice 3:

- non-autoregressive frame generation inside the acoustic stage
- strong conditional control from tokens + prompt mel + speaker embedding
- a clean streaming-friendly decoder because the estimator is causal/chunk-aware

#### DiT input layout

The DiT does not receive text tokens directly.

Its `InputEmbedding` concatenates:

- current noisy mel `x`
- prompt-conditioning mel `cond`
- token-conditioned frame prior `mu`
- repeated speaker embedding

Per frame, the concatenated feature size is:

```text
80 (x) + 80 (cond) + 80 (mu) + 80 (speaker) = 320
```

Then a linear projection maps that to:

- `1024` hidden channels

So the DiT is operating in a frame-level latent space that already contains all relevant conditions.

#### DiT block structure

The estimator is:

- hidden size `1024`
- depth `22`
- heads `16`
- head dimension `64`
- feed-forward multiplier `2`

Conceptually:

```text
input frames + time embedding
   ->
22 DiT blocks
   ->
final adaptive layer norm
   ->
80-channel velocity prediction
```

#### Why DiT is a strong architectural choice here

Compared with older conv-heavy diffusion backbones, DiT is attractive because:

- attention models long-range frame dependencies better
- scaling depth/width is straightforward
- chunk masks make streaming possible

The paper explicitly says the DiT backbone was strong enough that older, more complicated support modules could be removed.


### 5.6 Classifier-free guidance in the flow decoder

The flow decoder uses classifier-free guidance (CFG).

#### Training side

With probability `training_cfg_rate = 0.2`, the decoder randomly drops:

- `mu`
- `spks`
- `cond`

for a training sample.

That forces the model to learn both:

- conditional behavior
- unconditional behavior

#### Inference side

At inference, the decoder duplicates the batch internally:

- one conditional path
- one dropped-conditioning path

and combines them as:

```text
guided = (1 + cfg_rate) * conditional - cfg_rate * unconditional
```

with:

- `inference_cfg_rate = 0.7`

Why do this?

- stronger conditioning fidelity
- better adherence to prompt and speaker attributes
- sharper mel generation


### 5.7 Streaming-aware flow design

The streaming path has several small but important tricks.

#### Static chunk size

The released config sets:

- token chunk size: `25`
- token-to-mel ratio: `2`
- DiT static chunk size: `50` mel frames

So the DiT is trained to process chunks that correspond to about:

- `25 tokens`
- `50 mel frames`
- `1 second of audio`

#### Random streaming/full-context training

During flow training, the code randomly chooses:

- full-context mode
- streaming mode

with a 50/50 split.

That is important because the same decoder must work well in both:

- offline full-utterance synthesis
- online chunked synthesis

#### Deterministic noise for chunk consistency

`CausalConditionalCFM` uses a fixed stored random noise tensor at inference instead of fresh noise each time.

That is a subtle but important choice.

If chunked decoding used different random noise every time, recomputing prefixes could drift from full-utterance decoding. Fixed noise makes chunk recomputation much more stable.


### 5.8 CausalHiFTGenerator: mel-to-waveform

#### What it does

The vocoder turns the generated mel into actual waveform samples.

But it does not do that by plain transposed-conv decoding only.

It explicitly models excitation:

```text
mel -> f0 -> harmonic source -> source-filter style waveform generator
```

#### Why that helps

This is good for:

- pitch stability
- voiced/unvoiced structure
- naturalness
- streaming robustness

#### Step-by-step

1. Predict `f0` from mel using a causal convolutional network.
2. Upsample `f0` to waveform time resolution.
3. Generate harmonic source excitation from `f0`.
4. Run the HiFT decoder, fusing:
   - mel-conditioned decoder features
   - source branch features
5. Predict ISTFT magnitude/phase-like components.
6. Invert with ISTFT to waveform.

#### Exact temporal ratio

From the config:

- mel hop size is `480` samples at `24 kHz`
- so one mel frame equals `480 / 24000 = 20 ms`

Therefore:

- `50 mel frames / second`
- `480 waveform samples / mel frame`
- `960 waveform samples / speech token`

#### Vocoder internal upsampling

The released vocoder uses:

- upsample rates `[8, 5, 3]`
- ISTFT hop `4`

The product is:

```text
8 * 5 * 3 * 4 = 480
```

which exactly matches the mel hop size.

That exact alignment is deliberate. It keeps the acoustic stack temporally coherent.

#### Streaming-specific behavior

The causal vocoder uses:

- a small right lookahead in `conv_pre`
- `finalize=True/False` handling
- cache-aware speech offsets in `CosyVoice3Model`

In the CosyVoice 3 runtime wrapper, the system often recomputes a larger mel prefix but only returns the newly exposed waveform segment. That makes chunk joins smoother than trying to decode each chunk in total isolation.


## 6. Mode-specific conditioning behavior

The runtime supports several modes. The core stack is the same, but what each stage sees changes.

| Mode | LLM conditions on | Flow conditions on | Why |
| --- | --- | --- | --- |
| SFT | target text / instruction | speaker embedding (usually speaker-level), no prompt audio required | for speaker-specific models |
| Zero-shot | prompt text + target text + prompt speech tokens | prompt speech tokens + prompt mel + speaker embedding | strongest cloning mode |
| Cross-lingual | target text only | prompt speech tokens + prompt mel + speaker embedding | avoid leaking prompt lexical content into target language |
| Instruct2 | instruction text + target text | prompt speech tokens + prompt mel + speaker embedding | control style via text while cloning acoustics from prompt |

Two especially important details:

### Cross-lingual mode

`frontend_cross_lingual` deletes:

- `prompt_text`
- `llm_prompt_speech_token`

from the LLM path.

That is deliberate.

If the LLM saw prompt transcript or prompt speech tokens in cross-lingual mode, it could try to continue the prompt content instead of only borrowing speaker/style information.

### Instruct2 mode

`frontend_instruct2` keeps prompt audio for the flow stage but removes prompt speech tokens from the LLM stage.

That also makes sense:

- the instruction text should drive the linguistic/prosodic control
- the prompt audio should drive acoustics and identity


## 7. End-to-end inference walkthrough

This section follows the actual runtime path in the repo.

### 7.1 Frontend preprocessing

The frontend does three separate extractions from prompt audio:

```text
prompt wav
  -> text-independent speech tokens (16 kHz -> Whisper log-mel -> tokenizer ONNX)
  -> prompt mel features      (24 kHz -> 80-bin mel)
  -> speaker embedding        (16 kHz -> fbank -> CAMPPlus ONNX)
```

At the same time it:

- normalizes text
- optionally splits long text into smaller synthesis units
- tokenizes target text or instruction text with the extended Qwen tokenizer

For zero-shot mode, there is an additional alignment fix:

```text
token_len = min(mel_len / 2, speech_token_len)
```

and both prompt mel and prompt speech tokens are truncated to agree on the exact `2:1` mel-to-token ratio.

That small detail is crucial. If prompt tokens and prompt mel were misaligned, the flow stage would get inconsistent prefix conditioning.

### 7.2 LLM prefix construction

The LM receives a prefix sequence built from:

- `SOS`
- prompt/instruction text
- target text
- `TASK`
- optional prompt speech tokens

and then autoregressively emits speech tokens one by one.

The LM output is not text. It is a token stream in the speech-token vocabulary.

### 7.3 Runtime token generation thread

`CosyVoice3Model.tts` starts a background thread:

```text
llm_job(...)
```

That thread:

1. runs the LLM
2. appends generated speech token IDs into a shared list
3. marks completion when `EOS` or another stop token is hit

So the main thread can already start audio synthesis before the full token sequence is done.

### 7.4 Main thread: polling and chunk synthesis

The main thread periodically checks whether enough tokens are available.

For CosyVoice 3 streaming:

- initial token hop: `25`
- max hop: `100`
- scale factor: `2`

So it starts by synthesizing about 1 second chunks, then can increase chunk size.

Conceptually:

```text
while LM still generating:
    if enough new tokens exist:
        run token2wav on current prefix
        yield only the newly exposed waveform part
```

#### Why prefix recomputation is used

This is a very important implementation detail.

The streaming path does not synthesize each chunk from scratch in isolation.
Instead, it often reruns the flow/vocoder on a larger prefix and then returns only the new audio part.

That is why chunk transitions stay stable:

- the model always sees the full available history
- the wrapper trims away already-returned speech

### 7.5 Token-to-mel inference

When `token2wav` calls the flow stage:

1. prompt tokens and generated tokens are concatenated
2. token embeddings are produced
3. pre-lookahead refinement is applied
4. each token state is duplicated into 2 mel frames
5. prompt mel occupies the prefix of `cond`
6. the decoder runs 10 Euler flow steps with a cosine time schedule
7. the prompt mel prefix is discarded

At the end of this stage, the system has:

```text
generated mel: [1, 80, L_gen_mel]
```

### 7.6 Mel-to-wave inference

The vocoder then:

1. predicts `f0`
2. builds the harmonic source
3. synthesizes waveform samples
4. returns only the newly exposed audio segment if this is a streaming chunk

### 7.7 Final output

The final output yielded to the caller is:

```text
tts_speech: [1, L_wav]
```

at:

- `24 kHz`


## 8. Training pipeline: paper-level and repo-level

CosyVoice 3 training is best understood as several stacked training problems, not one single loop.

```text
        +-------------------------------+
        | 1. Train speech tokenizer     |
        |    (paper)                    |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        | 2. Pretrain text->token LM    |
        |    (repo + paper)             |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        | 3. Train token->mel flow      |
        |    (repo + paper)             |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        | 4. Train HiFT GAN vocoder     |
        |    (repo + paper)             |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        | 5. Post-train / preference    |
        |    optimization (paper)       |
        +---------------+---------------+
                        |
                        v
        +-------------------------------+
        | 6. Speaker fine-tune /        |
        |    continual pretrain (paper) |
        +-------------------------------+
```

### Important note

The repo training script `cosyvoice/bin/train.py` trains one module at a time:

- `llm`
- `flow`
- `hifigan`

The tokenizer training stage is outside this repo, and the exact DiffRO loop from the paper is not fully reproduced here.


### 8.1 Data pipeline before model training

#### Paper data pipeline

The paper describes a multilingual in-the-wild data pipeline:

1. speech detection and segmentation
2. noise reduction
3. ASR transcription
4. punctuation adjustment
5. volume standardization
6. filtering abnormal audio-text length ratios

This is the upstream corpus construction pipeline.

#### Repo training data pipeline

The repo training recipe assumes you already have prepared data and then performs:

```text
parquet_open
  -> tokenize text / instruction
  -> filter
  -> resample
  -> compute_fbank
  -> parse speaker embedding
  -> optionally compute Whisper fbank
  -> shuffle
  -> sort
  -> dynamic batch
  -> pad
```

For GAN training it additionally uses:

- truncation
- f0 extraction

#### Batch fields

The padded training batch can contain:

- `text_token`: `[B, L_txt_max]`
- `instruct_token`: `[B, L_ins_max]` if present
- `speech_token`: `[B, L_tok_max]` if precomputed
- `whisper_feat`: `[B, L_whisper_max, 128]` if online token extraction is used
- `speech_feat`: `[B, L_mel_max, 80]`
- `embedding`: `[B, 192]`
- `speech`: `[B, L_wav_max]` for GAN training
- `pitch_feat`: `[B, L_mel_max]` for GAN training

One subtle but important design choice:

- the batch carries both utterance-level and speaker-level embeddings
- the config chooses which one becomes `embedding`

In the released zero-shot config:

- `use_spk_embedding: False`

which means training uses utterance-level embeddings by default.

For SFT, the config comment explicitly says to switch this to speaker-level embeddings.


### 8.2 Stage 1: speech tokenizer training (paper)

This stage is described in the paper, not in this repo's training script.

#### Inputs

- raw speech
- task labels for ASR, LID, SER, AED, speaker analysis

#### Output

- discrete speech token IDs at 25 Hz

#### Loss and purpose

The tokenizer is supervised through the downstream MinMo stack so that the discrete bottleneck preserves:

- lexical content
- language identity
- emotion
- audio events
- speaker-related information

This is what makes the tokenizer suitable for downstream generative speech modeling instead of only recognition.


### 8.3 Stage 2: LLM training (repo)

The repo's `llm` training optimizes `CosyVoice3LM`.

#### Inputs

- `text_token`
- `instruct_token` if present
- `speech_token` or online-extracted speech tokens

#### What the model learns

It learns:

```text
instruction/text prefix -> speech token continuation
```

#### Exact sequence construction

The training code creates:

- unistream sequences
- bistream interleaved sequences

and computes cross-entropy over speech-token targets.

#### Target construction intuition

The model is trained so that:

- text and instruction tokens are context only
- speech tokens are the prediction target
- reserved special tokens (`EOS`, `FILL`, etc.) control sequence boundaries

#### Why this stage exists separately

This stage should learn:

- what to say
- in what approximate duration/prosodic structure
- how instructions and style markers should change the speech-token stream

It should **not** waste capacity learning fine spectral detail. That is the next stage's job.


### 8.4 Stage 3: flow training (repo)

The repo's `flow` training optimizes `CausalMaskedDiffWithDiT`.

#### Inputs

- `speech_token` or online tokenizer output
- `speech_feat`
- `embedding`

#### Step-by-step

1. Get speech tokens.
   - from `batch['speech_token']` if present
   - otherwise from the frozen tokenizer ONNX using `whisper_feat`
2. Embed speech tokens into 80-d vectors.
3. Apply the pre-lookahead layer.
4. Repeat each token representation twice to match 50 Hz mel rate.
5. Build prompt-conditioning mel `cond`.
   - half the time there is no prefix conditioning
   - otherwise a random prefix up to 30% of the utterance is revealed
6. Project the speaker embedding from 192 -> 80.
7. Randomly choose streaming or full-context mode.
8. Sample random flow time `t` and random noise `z`.
9. Train DiT to predict the flow velocity field.

#### Why random prefix conditioning is used

This is a clever training trick.

The model is not always shown the same prompt format.

Sometimes:

- it gets no acoustic prefix

Sometimes:

- it gets a partial mel prefix

That teaches the flow decoder to handle:

- pure generation
- prompt continuation
- partial-prefix continuation

with the same architecture.


### 8.5 Stage 4: HiFT GAN training (repo)

The repo's `hifigan` training actually trains the HiFT generator plus discriminator as a GAN.

#### Inputs

- `speech_feat` (mel)
- `speech` (real waveform)
- `pitch_feat` (f0 target)

#### Generator side

The generator:

1. predicts waveform from mel
2. predicts f0 from mel
3. is optimized with:
   - adversarial generator loss
   - feature matching loss
   - multi-mel reconstruction loss
   - optional TPR loss
   - f0 L1 loss

#### Discriminator side

The discriminator:

1. compares real vs generated waveforms
2. optimizes discriminator loss
3. may also use the TPR term

#### Why this stage is separate

The vocoder needs to solve a very different problem from the LM and flow model:

- waveform realism
- harmonic quality
- phase-consistent synthesis

GAN training is appropriate here, but not for the upstream text-to-token model.


### 8.6 Stage 5: DiffRO post-training (paper)

The paper introduces Differentiable Reward Optimization (DiffRO).

#### The problem it is solving

Classical RL for TTS is awkward because:

- speech-token LM output still has to pass through flow + vocoder
- those downstream stages are expensive
- speaker similarity after downstream decoding is often uniformly high, which makes reward discrimination hard

#### Core idea

Instead of optimizing generated waveform directly, DiffRO optimizes the speech-token distribution itself.

The paper pipeline is:

```text
LM token logits
   -> Gumbel-Softmax sampled soft token sequence
   -> Token2Text / ASR-like reward model
   -> differentiable reward
   -> KL-regularized optimization against reference model
```

#### Intuition

The reward asks:

> "Do these generated speech tokens still contain the text content and desired attributes?"

If yes, the LM should be nudged toward them.

#### Multi-task reward

The paper also describes adding reward models for:

- SER
- MOS prediction
- AED
- other audio-understanding tasks

This extends post-training from "say the text correctly" to "say it with the right attributes".

#### Repo note

The exact CosyVoice 3 DiffRO loop is not released here in the same form.

The closest open-source post-training hooks in this repo are:

- DPO support in `cosyvoice/bin/train.py`
- CosyVoice2 GRPO examples under `examples/grpo/cosyvoice2`

These are useful reference points, but they are not the full paper DiffRO pipeline.


### 8.7 Stage 6: speaker fine-tuning and capability transfer (paper)

The paper's SFT story is more interesting than "just fine-tune on one speaker".

It is trying to preserve and transfer:

- multilingual ability
- instruction following
- speaker identity

at the same time.

#### Paper strategy

For multilingual SFT:

- the instruction prompt explicitly mentions speaker identity and target language

For instructed SFT:

- speaker prompt and style prompt can both appear
- either one may be blank
- they are randomly masked during fine-tuning

This is meant to avoid catastrophic forgetting and improve transfer.

#### Repo-level implication

At the config level, the main visible SFT switch is:

- use speaker-level embedding instead of utterance-level embedding

That aligns with the paper idea of using stable speaker identity during fine-tuning.


## 9. Why each architectural decision makes sense

This section turns the architecture into plain engineering logic.

### 9.1 Why use a discrete speech-token LM at all?

Because it cleanly separates:

- long-range linguistic/autoregressive planning
- short-range acoustic rendering

An autoregressive waveform or mel model would make streaming and scaling much harder.

### 9.2 Why use a better tokenizer instead of only scaling the LM?

Because the LM cannot model information that the tokenizer discards.

If the tokenizer keeps richer prosody/emotion/event cues, the LM and flow model inherit a better latent language.

### 9.3 Why remove the old text encoder and length regulator?

Because once speech tokens already encode speech-time structure:

- the LM handles text-to-token generation
- the flow model only needs token-to-frame upsampling

The fixed `25 Hz -> 50 Hz` ratio makes simple repetition possible.

That reduces complexity and removes a common source of alignment fragility.

### 9.4 Why send prompt speech tokens to the LM?

Because prompt speech tokens live in the same discrete space as generated speech tokens.

That gives the LM an easy way to continue:

- speaking rate
- style
- rhythm
- token-space acoustic semantics

### 9.5 Why also send prompt mel to the flow model?

Because the flow model is responsible for local acoustics, not the LM.

Prompt mel gives it:

- spectral texture
- local continuity at the prompt boundary
- a direct acoustic anchor

### 9.6 Why keep a separate speaker embedding if prompt audio already exists?

Because prompt audio is a sequence, but speaker identity is also useful as a compact global vector.

The embedding gives the decoder a stable identity summary even when the prompt is short or noisy.

### 9.7 Why use DiT for token-to-mel?

Because this stage needs:

- high-capacity conditional generation
- long-range frame modeling
- scalable depth/width
- chunk-mask friendliness for streaming

DiT fits all four.

### 9.8 Why use CFG in the flow decoder?

Because prompt fidelity matters.

CFG lets the model trade:

- mode coverage
- condition adherence

without retraining a second model.

### 9.9 Why use an f0-aware vocoder?

Because TTS quality depends strongly on excitation structure.

Predicting f0 and generating a harmonic source helps the vocoder keep:

- pitch stable
- voicing natural
- prosody more realistic

### 9.10 Why keep the runtime chunked and recomputational?

Because perfect streaming is not only about latency.

It is also about:

- avoiding audible seams
- keeping chunk outputs consistent with full-context decoding

Recomputing a growing prefix and returning only the new segment is a practical way to get both.


## 10. Practical shape walkthrough with a concrete example

Suppose:

- prompt audio is 4 seconds long
- target generation is 6 seconds long

Then approximately:

- prompt tokens: `4 * 25 = 100`
- generated tokens: `6 * 25 = 150`
- prompt mel frames: `200`
- generated mel frames: `300`
- generated waveform samples: `300 * 480 = 144000`

The main tensors would look like:

```text
text tokens                  : [1, L_txt]
prompt text tokens           : [1, L_ins]
prompt speech tokens         : [1, 100]
prompt mel                   : [1, 200, 80]
speaker embedding            : [1, 192]

LM input hidden states       : [1, 1 + L_ins + L_txt + 1 + 100, 896]
generated speech tokens      : [1, 150]

flow token sequence          : [1, 250]
flow frame prior after x2    : [1, 500, 80]
flow output mel              : [1, 80, 300]

final waveform               : [1, 144000]
```

That example is useful because it shows the stage separation clearly:

- LM thinks in `150` coarse acoustic tokens
- flow expands them into `300` mel frames
- vocoder expands those into `144000` audio samples


## 11. Implementation map

If you want to trace the exact path in code, these are the files that matter most.

### Frontend and orchestration

- `cosyvoice/cli/cosyvoice.py`
- `cosyvoice/cli/frontend.py`
- `cosyvoice/cli/model.py`

### Text and speech token modeling

- `cosyvoice/tokenizer/tokenizer.py`
- `cosyvoice/llm/llm.py`

### Token-to-mel

- `cosyvoice/flow/flow.py`
- `cosyvoice/flow/flow_matching.py`
- `cosyvoice/flow/DiT/dit.py`
- `cosyvoice/flow/DiT/modules.py`
- `cosyvoice/transformer/upsample_encoder.py`

### Vocoder

- `cosyvoice/hifigan/generator.py`
- `cosyvoice/hifigan/f0_predictor.py`
- `cosyvoice/hifigan/hifigan.py`

### Training and data

- `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`
- `examples/libritts/cosyvoice3/run.sh`
- `cosyvoice/bin/train.py`
- `cosyvoice/dataset/processor.py`
- `cosyvoice/utils/executor.py`
- `cosyvoice/utils/train_utils.py`


## 12. Final mental model

If you remember only one diagram, remember this one:

```text
                                  CosyVoice 3.0

    text / instruction
            |
            v
    +------------------+          prompt wav
    | Qwen-based LM    |<-------------+
    | (AR, 25 Hz)      |              |
    +------------------+              |
            |                         |
            v                         |
    speech tokens                     |
            |                         |
            v                         |
    +------------------+              |
    | Causal DiT CFM   |<-------------+---- prompt mel prefix
    | (token -> mel)   |<-------------+---- speaker embedding
    +------------------+
            |
            v
         mel frames
            |
            v
    +------------------+
    | Causal HiFT      |
    | (mel -> waveform)|
    +------------------+
            |
            v
       24 kHz speech
```

The cleanest intuition is:

- the **LM decides the discrete speech plan**
- the **flow model paints that plan into acoustic frames**
- the **vocoder turns the frames into pressure waves**

And the whole CosyVoice 3 improvement story is:

- better speech tokens
- more diverse data
- stronger text-to-token LM
- stronger token-to-mel decoder
- better post-training

all working together around the same staged pipeline.
