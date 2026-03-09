# Voice Changer Design

This document defines the voice changer feature for CosyVoice using the **existing training-free pipeline** that is already present in the repository.

The updated goal is:

- input: `source_audio`, `ref_audio`
- output: audio that preserves the **content and approximate timing/cadence** of `source_audio`
- speaker identity: taken from `ref_audio`
- implementation constraint: **training-free**

This design intentionally does **not** require perfect word-level or sample-level timing preservation. It uses the current voice-conversion path as the product feature.


## 1. Short answer

The repository already contains a training-free voice changer:

- `cosyvoice/cli/cosyvoice.py::inference_vc`
- `cosyvoice/cli/frontend.py::frontend_vc`
- `cosyvoice/cli/model.py::vc_job`

That path should be treated as the official implementation.

Conceptually it works like this:

```text
source_audio
  -> speech tokenizer
  -> source speech tokens

ref_audio
  -> prompt speech tokens
  -> prompt mel
  -> speaker embedding

source speech tokens + target voice conditioning
  -> flow
  -> vocoder
  -> converted waveform
```

The important product decision is:

- use the current VC pipeline as-is
- do not add a new model
- do not add training
- do not insert the LLM into the loop
- accept that timing preservation is **strong but not exact**


## 2. Why this is the right design

### 2.1 It already matches the intended use case

The requested behavior is:

- keep what is being said from `source_audio`
- change who is saying it to the speaker from `ref_audio`

That is exactly what the current VC path is built for.

Unlike the standard TTS path, the VC path:

- does not tokenize text and regenerate a new speech-token sequence
- does not rely on the autoregressive LLM
- directly reuses the **source speech-token sequence**

That is the main reason it preserves source timing and cadence reasonably well without additional machinery.

### 2.2 It is fully training-free

The current pipeline only uses frozen components:

- `speech_tokenizer_v3.onnx`
- CAMPPlus speaker embedding model
- frozen flow model
- frozen HiFT vocoder

No weights are updated.

So the feature already satisfies the strongest engineering constraint:

- **no fine-tuning**
- **no adaptation**
- **no RL**
- **no new training recipe**

### 2.3 It is simpler and lower risk

The previous version of this design doc proposed explicit timing-lock and prosody-lock control layers.
That is no longer necessary for the product goal.

Those additions would have increased:

- implementation complexity
- latency
- maintenance burden
- failure surface

without being required by the updated requirement.


## 3. Existing pipeline review

### 3.1 Current frontend path

`frontend_vc(source_speech_16k, prompt_wav, resample_rate)` currently does four things:

1. Extract `prompt_speech_token` from `ref_audio`
2. Extract `prompt_speech_feat` from `ref_audio`
3. Extract target `embedding` from `ref_audio`
4. Extract `source_speech_token` from `source_audio`

Returned model input:

```text
{
  source_speech_token,
  flow_prompt_speech_token,
  prompt_speech_feat,
  flow_embedding
}
```

This is exactly the right conditioning package for a training-free voice changer.

### 3.2 Current runtime path

At runtime:

- `inference_vc(...)` calls `frontend_vc(...)`
- `model.tts(...)` sees that `source_speech_token` is present
- `vc_job(...)` bypasses LLM generation and writes the source tokens directly into the synthesis queue

In other words:

```text
source tokens are used directly as the content plan
```

This is the key design property that makes the current VC feature preferable to any text-based reconstruction path.

### 3.3 What the current VC path already preserves well

Because the source speech-token stream is reused directly, the current pipeline tends to preserve:

- spoken content
- phrase structure
- pause placement, approximately
- rough duration
- rough cadence

It will generally preserve timing much better than:

- ASR -> text -> zero-shot TTS

because that alternative regenerates timing from text.

### 3.4 What the current VC path does not guarantee

The current VC path should not be documented as providing:

- exact word-level timing lock
- exact sample count match
- exact pause duration match
- exact f0 contour preservation

Why:

- source speech tokens run at `25 Hz`, so timing is represented on a coarse `40 ms` grid
- the flow regenerates mel
- the vocoder regenerates waveform detail

So the correct claim is:

- **strong timing/cadence preservation**
- **not exact timing reconstruction**


## 4. Updated product requirement

The voice changer feature should now be framed as:

> Given `source_audio` and `ref_audio`, generate speech with the content and approximate timing/cadence of `source_audio`, but with the speaker identity of `ref_audio`, using only the frozen existing CosyVoice VC pipeline.

This wording is important because it aligns with what the current implementation can actually support.


## 5. Proposed feature definition

### 5.1 Inputs

- `source_audio`
  - the speech whose linguistic content, phrasing, and approximate timing should be preserved
- `ref_audio`
  - a target voice sample that defines the speaker identity of the output

Optional future input:

- `stream`
- `speed`

but the default feature should just expose the existing VC behavior cleanly.

### 5.2 Output

- converted waveform at CosyVoice output sample rate
- target speaker timbre from `ref_audio`
- source content from `source_audio`
- approximate timing/cadence continuity from `source_audio`

### 5.3 Non-goals

This feature should explicitly **not** promise:

- perfect word boundary preservation
- exact prosodic imitation
- exact waveform length equality
- exact emotional transfer from source

If those are needed later, they should be treated as a separate "timing-locked VC" project, not part of this feature.


## 6. High-level architecture

```text
                    Existing Training-Free Voice Changer

source_audio ----------------------------------------------------+
                                                                 |
                                                                 v
                                                      +----------------------+
                                                      | Speech Tokenizer v3  |
                                                      | -> source tokens     |
                                                      +----------+-----------+
                                                                 |
                                                                 |
ref_audio -------------------------------------------------------+-----------------------------+
                                                                                              |
                                                                                              v
                                                      +-----------------------------------------------+
                                                      | Reference conditioning                         |
                                                      |                                               |
                                                      | - prompt speech tokens from ref_audio         |
                                                      | - prompt mel from ref_audio                   |
                                                      | - speaker embedding from ref_audio            |
                                                      +----------------------+------------------------+
                                                                             |
                                                                             v
                                                      +-----------------------------------------------+
                                                      | Existing VC core                               |
                                                      |                                               |
                                                      | source tokens                                 |
                                                      |   + target prompt tokens/mel/embedding        |
                                                      |   -> flow -> vocoder -> converted waveform    |
                                                      +-----------------------------------------------+
```


## 7. Why the LLM should stay out of the loop

The standard CosyVoice 3 TTS architecture includes:

- tokenizer
- LLM
- flow
- vocoder

The VC path intentionally skips the LLM.

That is correct and should not be changed.

Why:

- the LLM is useful when generating speech tokens from text
- for voice changing, the source audio already provides speech tokens
- using the LLM would reintroduce timing drift
- using the LLM would increase latency and complexity

So the design rule is:

```text
voice changer = source token passthrough + target speaker conditioning
```


## 8. Training-free properties

The feature is training-free in a strict sense.

### Allowed components

- frozen source speech tokenizer
- frozen prompt speech tokenizer
- frozen speaker embedding extractor
- frozen flow model
- frozen HiFT vocoder

### Not allowed

- any checkpoint updates
- LoRA
- prompt tuning
- RL / DPO / GRPO
- test-time adaptation that modifies weights

Simple preprocessing is still allowed:

- resampling
- segmentation
- trimming
- optional stitching logic

Those do not violate the training-free constraint.


## 9. Expected behavior

### 9.1 What users should expect

Users should expect:

- same words, or very close
- same speaking order and rough phrasing
- similar overall duration
- similar pause rhythm
- noticeably different speaker identity matching `ref_audio`

### 9.2 What users should not expect

Users should not expect:

- exact frame-by-frame timing preservation
- exact source pitch contour
- exact emotional shape transfer
- perfect preservation under very noisy source audio

This should be part of product documentation, not hidden as an implementation detail.


## 10. Recommended API

The existing public path is already good:

- `inference_vc(source_wav, prompt_wav, stream=False, speed=1.0)`

If we want a more product-facing name, add a thin alias:

- `inference_voice_change(source_audio, ref_audio, stream=False, speed=1.0)`

Internally it should just call the current VC path.

### Recommended behavior

```python
def inference_voice_change(source_audio, ref_audio, stream=False, speed=1.0):
    return self.inference_vc(source_audio, ref_audio, stream=stream, speed=speed)
```

This keeps the implementation honest:

- no new algorithm
- no new model path
- no duplicated inference logic


## 11. Optional lightweight improvements

These are acceptable because they do not change the core design.

### 11.1 Long-audio segmentation

The tokenizer currently rejects a single extraction longer than `30s`.

So for production use, a lightweight wrapper should:

- segment long `source_audio`
- run existing VC on each segment
- stitch outputs back together

This is a packaging improvement, not a model change.

### 11.2 Better silence handling

If needed, a wrapper can preserve long source silences more faithfully by:

- detecting long silent spans in `source_audio`
- processing voiced segments only
- reinserting source-like silence between converted segments

Again, this is optional and inference-only.

### 11.3 Reference-audio cleanup

Since speaker conditioning quality depends on `ref_audio`, a wrapper can improve robustness by:

- trimming long leading/trailing silence
- choosing a cleaner reference clip
- warning on very short or noisy reference audio


## 12. Acceptance criteria

The feature should be accepted if it satisfies these practical conditions.

### 1. Speaker transfer

The output should sound like the speaker from `ref_audio`, not like the original source speaker.

### 2. Content preservation

ASR on converted output should stay close to ASR on `source_audio`.

### 3. Timing similarity

The output should preserve:

- similar total duration
- similar phrase timing
- similar pause rhythm

Exact match is not required.

### 4. Training-free implementation

The implementation must use only:

- frozen shipped models
- inference-time preprocessing and orchestration


## 13. Limitations

These limitations should be documented.

### Limitation 1: timing is approximate, not exact

Because the current VC path reuses `25 Hz` speech tokens and regenerates mel/waveform, some timing drift is expected.

### Limitation 2: source-speaker leakage can remain

Speech tokens are not perfectly speaker-invariant, so some source-speaker traits may survive.

### Limitation 3: very noisy source audio will degrade conversion

If `source_audio` is noisy, reverberant, or heavily overlapped, content preservation and cadence preservation can degrade.

### Limitation 4: reference quality matters

If `ref_audio` is too short, too noisy, or heavily expressive, target speaker conditioning can become unstable.

### Limitation 5: long audio needs wrapper logic

Very long audio should be segmented and stitched externally or in a thin wrapper because the tokenizer path is not meant for a single >30s chunk.


## 14. Final recommendation

Treat the **existing VC pipeline** as the feature.

Do not redesign it into a timing-locked system.
Do not add a new model path.
Do not use the LLM.
Do not train anything.

The recommended product stance is:

```text
CosyVoice voice changer is a training-free source-token-based voice conversion pipeline that preserves source content and approximate timing/cadence while transferring the target speaker identity from a reference audio clip.
```

In one line:

```text
Use `source_audio` for content and rough timing, use `ref_audio` for speaker identity, and rely on the current `inference_vc` path end-to-end.
```
