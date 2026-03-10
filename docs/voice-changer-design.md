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


## 14. How to run the voice changer

This repository already contains sample audio files that are ready to use for the existing VC pipeline:

- `./data/input_audio.wav`
  - use this as `source_audio`
  - this is the speech whose words, phrasing, and rough cadence will be preserved
- `./data/ref_voice.mp3`
  - use this as `ref_audio`
  - this is the target voice sample whose speaker identity will be transferred

Ignore `./data/._input_audio.wav`.
That file is only a macOS metadata artifact and is not real audio.

### 14.1 Run from the repository root

All commands below assume your working directory is the repository root:

```sh
cd /path/to/CosyVoice
```

### 14.2 Make sure dependencies are already installed

Use the normal CosyVoice environment described in the main README.
If you have already run CosyVoice inference before, you can usually skip this step.

At minimum, the runtime here assumes:

- Python can import `torch`
- Python can import `torchaudio`
- Python can import `cosyvoice`
- the local checkout includes `third_party/Matcha-TTS`

### 14.3 Make sure the CosyVoice 3 model exists locally

The runnable path described here uses a CosyVoice 3 checkpoint directory.
It does not need to live under `pretrained_models/` specifically.
For example, either of these layouts works:

- `pretrained_models/Fun-CosyVoice3-0.5B`
- `checkpoints/Fun-CosyVoice3-0.5B`

If your chosen directory already exists, keep using it.
If it does not exist, download it first.

Hugging Face download:

```sh
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='checkpoints/Fun-CosyVoice3-0.5B',
)
PY
```

ModelScope download:

```sh
python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='checkpoints/Fun-CosyVoice3-0.5B',
)
PY
```

### 14.4 Verify that the bundled sample audio files are present

Before running inference, confirm that the repo already has the two audio files this document uses:

```sh
ls -lh ./data/input_audio.wav ./data/ref_voice.mp3
```

Those are the exact files that should be plugged into the existing VC path.

### 14.5 Run the existing VC pipeline with the bundled sample files

The simplest ready-to-run command is a one-shot Python script:

```sh
python - <<'PY'
import sys
import torchaudio

sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel

model_dir = 'checkpoints/Fun-CosyVoice3-0.5B'
source_audio = './data/input_audio.wav'
ref_audio = './data/ref_voice.mp3'

cosyvoice = AutoModel(model_dir=model_dir)

for i, result in enumerate(
    cosyvoice.inference_vc(source_audio, ref_audio, stream=False)
):
    output_path = f'./data/voice_changed_{i}.wav'
    torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
    print(output_path)
PY
```

This is the direct mapping from the design to the implementation:

- `source_audio = ./data/input_audio.wav`
- `ref_audio = ./data/ref_voice.mp3`
- `cosyvoice.inference_vc(source_audio, ref_audio, stream=False)`
- save output to `./data/voice_changed_0.wav`

### 14.6 What happens step by step when you run that command

1. `AutoModel(model_dir=...)` loads the frozen CosyVoice 3 inference stack.
2. `inference_vc(source_audio, ref_audio, ...)` calls `frontend_vc(...)`.
3. `frontend_vc(...)` loads `source_audio` and `ref_audio`.
4. `ref_audio` is used to extract:
   - prompt speech tokens
   - prompt speech features / mel-style conditioning
   - speaker embedding
5. `source_audio` is used to extract:
   - source speech tokens
6. The runtime detects that `source_speech_token` is present and enters the VC path instead of the text-to-speech LLM path.
7. `vc_job(...)` passes the source speech-token sequence directly into the synthesis queue.
8. The frozen flow model converts those source tokens plus the target-speaker conditioning into acoustic features.
9. The frozen vocoder converts those acoustic features into waveform samples.
10. `torchaudio.save(...)` writes the converted result to `./data/voice_changed_0.wav`.

Operationally, this means:

- content comes mainly from `source_audio`
- target speaker identity comes from `ref_audio`
- timing and cadence are usually similar to `source_audio`
- timing is not guaranteed to be exactly identical

### 14.7 Expected output

For the bundled sample files, the first output file will normally be:

- `./data/voice_changed_0.wav`

If the generator yields more than one result, the script will also create:

- `./data/voice_changed_1.wav`
- `./data/voice_changed_2.wav`
- and so on

In the common non-streaming case, you should expect a single file.

### 14.8 Optional: adjust runtime behavior

The current VC API already exposes two useful controls:

- `stream=False`
  - recommended default for offline conversion
- `speed=1.0`
  - keep at `1.0` unless you explicitly want faster or slower output pacing

Example:

```python
cosyvoice.inference_vc(source_audio, ref_audio, stream=False, speed=1.0)
```

For the feature described in this document, `stream=False` and `speed=1.0` should be treated as the default product behavior.

### 14.9 Optional fallback if MP3 decoding is unavailable

If your local `torchaudio` build cannot decode MP3, convert the reference file once and use WAV instead:

```sh
ffmpeg -i ./data/ref_voice.mp3 ./data/ref_voice.wav
```

Then change the run command to:

```python
ref_audio = './data/ref_voice.wav'
```

The rest of the pipeline stays exactly the same.

### 14.10 Practical constraints while running it

The current tokenizer path does not support extracting speech tokens from a single audio chunk longer than `30s`.

So in practice:

- the bundled sample files are the easiest ready-to-go demo path
- short single-utterance source audio is the safe default
- long source audio should be segmented before conversion if you want production robustness

### 14.11 Minimal success criterion

The run should be considered successful if:

- the command finishes without raising an exception
- `./data/voice_changed_0.wav` is written
- the output says the same words as `./data/input_audio.wav`
- the output sounds like the speaker from `./data/ref_voice.mp3`
- the output duration and phrasing are close to the source, even if not exact


## 15. Final recommendation

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
