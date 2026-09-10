# FASHN VTON 1.5 support in stable-diffusion.cpp

**Status: architectural investigation and implementation plan only. No implementation changes have been made.**

Investigation date: 2026-09-08.

## 1. Executive recommendation

Implement FASHN VTON 1.5 as a **new model family with a dedicated try-on inference entry point**, not as a FLUX checkpoint alias, a LoRA, ControlNet, or an ordinary img2img workflow.

The strongest reuse opportunity is unusually good: the existing `Flux::DoubleStreamBlock`, `Flux::SingleStreamBlock`, modulation, Q/K normalization, timestep MLP, and final projection closely match the released FASHN implementation. The missing work is mainly the architecture around those blocks, image conditioning, model identification, sampling semantics, and preprocessing.

Recommended sequence:

1. Establish a pinned PyTorch reference and prepared-input fixtures.
2. Add strict checkpoint identification and native FASHN graph execution.
3. Match individual layers and a complete conditional/unconditional forward pass.
4. Add the exact FASHN Euler/CFG loop and a prepared-input C API/CLI.
5. Validate real try-on output at the actual 576 x 864 model canvas.
6. Optimize memory, precision, and repeated work only after parity.
7. Add the server contract.
8. Optionally add a separate native preprocessing executable using ONNX Runtime and OpenCV.

The first deliverable should be **native GGML execution of the actual FASHN model using reference-prepared inputs**. It must not be advertised as a Python-free raw-image pipeline until pose detection, parsing, image transforms, and mask processing are also provided natively.

Two important limits:

- This VM is suitable for CPU development and correctness work, but its SD 1.5 smoke-test time is not a useful FASHN latency predictor.
- The main FASHN checkpoint/code is Apache-2.0, but the reference human parser has a separate NVIDIA SegFormer license with a non-commercial research/evaluation limitation. Treat deployment and redistribution of that component as a separate licensing decision.

## 2. Sources and evidence boundaries

The analysis is based on these exact revisions, not just repository README descriptions:

| Component | Revision |
|---|---|
| Local stable-diffusion.cpp | `d04e8950c1ec8d30248cbe996682b3182fb1adf6` |
| Its GGML submodule | `e20c3a14aa70ee84ca58499814206dd08d8026bc` |
| FASHN VTON source | `7c0f10af3f91ad4048fe9729c470a13ef905d25a` |
| FASHN VTON Hugging Face checkpoint | `7720683168567eb5a2a4c67f15116c6e29c83ded` |
| FASHN human-parser source | `f2771f2fb8655349e87e2869bbde7ace0bd06f2c` |
| FASHN human-parser Hugging Face model | `1f80c34dbab321c5730dda5c3fea279fd3e97498` |
| FASHN DWPose Hugging Face models | `548b5df25b84d9f4aac0611dfa1c2a7a12f15571` |

The FASHN checkpoint was inspected using HTTP byte ranges: the 8-byte safetensors length prefix and its **39,560-byte JSON header**. The approximately 1.94 GB tensor payload was not downloaded or executed.

This establishes the actual names, shapes, dtypes, tensor counts, and block counts. It does **not** establish numeric parity, weight-value ranges, throughput, peak memory, or image quality in a proposed C++ implementation.

The FASHN Python dependency declarations contain lower bounds rather than a complete environment lock. Before implementing parity tests, lock PyTorch, torchvision, NumPy, Pillow, OpenCV, transformers, safetensors, the parser package, and ONNX Runtime. A pinned model commit alone does not pin the preprocessing/runtime numerics.

All proposed paths, APIs, switches, and code fragments below are **design proposals** unless explicitly identified as existing. Code sketches describe integration and math; they are not claimed to have been compiled.

## 3. What FASHN actually runs

### 3.1 End-to-end pipeline

The reference pipeline is:

```text
person RGB image                         garment RGB image
      |                                        |
PIL aspect-fit pre-resize, max dimension 864, no upsampling
      |                                        |
DWPose on BGR                       DWPose on BGR for model photo
      |                             or dummy pose for flat-lay
      |                                        |
grayscale pose raster                grayscale pose raster
      |                                        |
human-parser segmentation            human-parser segmentation
      |                                        |
clothing-agnostic image              category-isolated garment image
(unchanged when segmentation-free)   (unchanged when flat-lay)
      |                                        |
OpenCV aspect-fit resize + symmetric black pad to 576 x 864
      |                                        |
RGB / one-channel pose normalization: byte / 127.5 - 1
      |                                        |
      +-------------- FASHN MMDiT -------------+
                           |
               Gaussian RGB noise, t = 0
                           |
         shifted ascending time schedule, Euler + CFG
                           |
                  RGB pixels at t = 1
                           |
           clamp, float-to-byte conversion, unpad
```

There is no CLIP, T5, text prompt encoder, VAE, latent scaling constant, image-captioning model, or garment-text encoder in this network. The names `img` and `txt` in its transformer are inherited from FLUX; **`txt` is the garment-image token stream**. [F-model] [F-pipeline]

The pose is an input image channel. It is not a ControlNet residual and is not a list of keypoints passed directly to the transformer.

### 3.2 Exact released configuration

| Item | Value |
|---|---|
| Model canvas | Height 864, width 576 |
| Output/input noisy channels | 3 RGB channels |
| Target/person patch input | 7 channels: noisy RGB + clothing-agnostic RGB + person pose |
| Garment patch input | 4 channels: garment RGB + garment pose |
| Patch kernel and stride | 12 x 12 |
| Patch grid | 72 rows x 48 columns |
| Tokens per image stream | 3,456 |
| Joint attention sequence | 6,912 |
| Hidden width | 1,280 |
| Attention heads | 10 |
| Head dimension | 128 |
| MLP expansion | 4; intermediate width 5,120 |
| Target-only patch mixer | 4 single-stream blocks |
| Double-stream core | 8 blocks |
| Joint single-stream core | 16 blocks |
| RoPE axes | `[16, 56, 56]` |
| RoPE theta | 10,000 |
| Timestep embedding | 256 sin/cos features; input time multiplied by 1,000 |
| Category embedding | 4 rows x 1,280; null=0, tops=1, bottoms=2, one-pieces=3 |
| Guidance-distillation embedding | Absent in the released model |
| Final token output | 432 = 3 x 12 x 12 |
| Checkpoint dtype | Every recorded tensor is BF16 |
| Checkpoint tensor count | 366 |
| Stored scalar count | 971,814,240 |

The scalar count includes `patch_mixer_token`, a 432-element buffer that is explicitly unused in inference. The trainable-weight scalar count excluding that buffer is **971,813,808**.

Do not infer resolution, RoPE theta, or the axis partition solely from tensor shapes. The checkpoint has no separate configuration file; those values come from the pinned source. Shapes validate the preset but do not fully specify it.

Some pose-shape unit tests use a height of 768 as a test fixture. That is not the released model's configured height; the actual `TryOnModel` constructor uses 864.

### 3.3 Actual checkpoint families

| Family | Tensor count | Scalar count | BF16 payload bytes |
|---|---:|---:|---:|
| `double_blocks` | 192 | 472,170,496 | 944,340,992 |
| `single_blocks` | 128 | 393,445,376 | 786,890,752 |
| `x_patch_mixer` | 32 | 98,361,344 | 196,722,688 |
| `t_embedder` | 4 | 1,968,640 | 3,937,280 |
| `y_embedder` | 1 | 5,120 | 10,240 |
| `x_embedder` | 2 | 1,291,520 | 2,583,040 |
| `garment_embedder` | 2 | 738,560 | 1,477,120 |
| `final_layer` | 4 | 3,832,752 | 7,665,504 |
| `patch_mixer_token` | 1 | 432 | 864 |

Total payload: **1,943,628,480 bytes**. Including the length prefix and header, the safetensors file is **1,943,668,048 bytes**.

### 3.4 Forward-pass tensor contract

Use `B=1` initially. Here, shapes are in **PyTorch order**:

| Stage | Shape |
|---|---|
| Noisy RGB `x` | `[B, 3, 864, 576]` |
| Clothing-agnostic image | `[B, 3, 864, 576]` |
| Person pose | `[B, 1, 864, 576]` |
| Target channel concatenation | `[B, 7, 864, 576]` |
| Target patch projection | `[B, 1280, 72, 48]` |
| Target tokens | `[B, 3456, 1280]` |
| Garment plus garment pose | `[B, 4, 864, 576]` |
| Garment tokens | `[B, 3456, 1280]` |
| Time plus category vector | `[B, 1280]` |
| Patch-mixer output | `[B, 3456, 1280]` |
| Double-stream outputs | Two tensors `[B, 3456, 1280]` |
| Joint single-stream input | `[B, 6912, 1280]`, garment first |
| Target-only tokens after slicing | `[B, 3456, 1280]` |
| Final projection | `[B, 3456, 432]` |
| Unpatchified velocity | `[B, 3, 864, 576]` |

The output is a **pixel-space flow velocity**, not a finished RGB image, epsilon, or SD's v-prediction parameterization.

## 4. Important numerical semantics

### 4.1 Conditional versus unconditional inference

FASHN's reference `forward_for_cfg` duplicates the batch and uses a boolean keep mask.

Conditional branch:

- Preserve noisy RGB.
- Preserve current timestep.
- Preserve clothing-agnostic image, both pose maps, and garment RGB.
- Use the requested category, 1-3.

Unconditional branch:

- Preserve the **same** noisy RGB and timestep.
- Replace all four normalized conditioning images with **exact floating-point zero**.
- Replace the category with integer zero.
- Retain the normal projection layers and their biases.
- Retain all garment tokens and joint-attention sequence length.

Consequences:

- Black PNG pixels normalize to `-1`, not zero.
- A gray fill of 127 normalizes to approximately `-0.0039215686`, not zero.
- A flat-lay's blank garment pose is an all-`-1` normalized image, not the unconditional zero pose.
- The null category uses the learned row zero of `y_embedder`; it is not an all-zero category embedding.
- Null garment embeddings are not necessarily zero because the convolution has a bias.
- Do not reuse the conditional patch-mixer output for the unconditional branch.

No unconditional text prompt or separate unconditional checkpoint is involved.

### 4.2 RoPE is spatial for both streams

For both images, the position at patch row `r`, column `c` is:

```text
[0, r, c]
```

Both grids use the same coordinates. Do not give garment tokens FLUX's all-zero text positions, a distinct reference-image index, shifted columns, or a mask distinguishing two image domains.

A subtle source detail: the Python forward concatenates positional embeddings as `[x_pe, g_pe]`, while attention concatenates actual tokens as `[garment, target]`. At the supported resolution, both position arrays are identical, so this has no numerical effect. Preserve the two identical grids; do not reinterpret this as permission to use different image sizes or new reference indexing.

The Python RoPE frequencies are calculated in float64 and then cast to float32. The current C++ `Rope::rope` uses float arithmetic and `powf`. Reuse the existing layout and attention implementation, but compare the generated tables. If strict reference matching requires a float64 table generator, add a **FASHN-local generator**, not an unvalidated global change to FLUX. [F-model] [S-rope]

### 4.3 Timestep direction and velocity sign

FASHN starts from noise at `t=0` and reaches the image at `t=1`.

For `N` steps, define:

```text
u_i = i / N
a   = exp(-mu)
t_i = a*u_i / (1-u_i + a*u_i)
mu  = 1.5 by default
```

This is algebraically equivalent to the reverse schedule in the reference's `get_rf_schedule`.

For four steps, the schedule is approximately:

```text
t:     0, 0.069227785, 0.182425524, 0.400978973, 1
sigma: 1, 0.930772215, 0.817574476, 0.599021027, 0
```

The direct update is:

```text
v_guided = v_uncond + cfg * (v_cond - v_uncond)
x_next   = x + (t_next - t) * v_guided
```

The last step uses `v_cond` alone by default. `skip_cfg_last_n_steps` defaults to 1 and should be an explicit parameter, not an undocumented modification of the user's CFG scale.

The existing FLUX flow denoiser uses `c_out = -sigma` and passes decreasing noise time. That is not directly compatible.

If a future integration adapts FASHN to the generic sigma-based Euler sampler:

```text
sigma = 1 - t
D(x, sigma) = x + sigma * v_fashn(x, 1-sigma)
(x-D)/sigma = -v_fashn
```

Therefore its denoiser scalings would be `c_skip=1`, `c_out=+sigma`, `c_in=1`. With decreasing sigma this reproduces the correct ascending-time update. Both the sign and the time transform matter.

Do not substitute `MiniT2IFlowDenoiser`: although it also reverses time, it uses different output scaling and initializes from noise multiplied by 2. [F-sampling] [F-pipeline] [S-denoisers]

### 4.4 Precision and output conversion

The reference uses float32 on CPU and on CUDA devices without BF16 support. It uses BF16 on compatible CUDA devices.

This is more than a weight-storage distinction. In the BF16 path, time vectors, conditioning, state updates, and activations can be rounded differently. A native F32-activation implementation with BF16 weights should not promise bit-identical results to the CUDA BF16 reference.

The reference clamps the final floating image to `[-1,1]`, maps it to `[0,1]`, and uses torchvision's conversion to byte images. The existing C++ image conversion uses round-to-nearest via `value*255 + 0.5`; torchvision's floating conversion normally truncates after scaling to 255. Pin and verify the torchvision behavior in the reference environment.

For pixel-level output parity, use a FASHN-specific final conversion rather than changing the existing global converter. Compare float output before byte conversion so a one-level rounding difference does not obscure model correctness.

## 5. Existing support versus missing support

| Requirement | Current stable-diffusion.cpp support | Required work |
|---|---|---|
| Safetensors and BF16 parsing | Present | Reuse; add family detection and strict validation |
| GGUF import/export | Present | Round-trip FASHN names/shapes; apply a deliberate quantization policy |
| CPU execution, AVX2 | Present | Reuse current build and GGML CPU backend |
| CUDA/Vulkan and other backends | Infrastructure present | Validate the new graph on each backend before claiming support |
| Linear, embedding, LayerNorm, RMSNorm | Present | Configure and compose them for FASHN |
| SiLU and tanh GELU | Present | Preserve FASHN activation choice |
| FLUX-style double-stream blocks | Present and closely matching | Reuse with width=1280, heads=10, MLP=4, biases on |
| FLUX-style single-stream blocks | Present and closely matching | Reuse for both patch mixer and core |
| Multiaxis RoPE and attention | Present | Generate FASHN's two spatial grids; validate precision |
| Flash attention | Present, with runtime support check and fallback | Enable only after forward parity; measure actual selected path |
| Patchify/unpatchify | Present in `DiT` | Use 12 x 12 and channel-first patch flattening |
| 7-channel and 4-channel patch projections | Generic operations exist; no FASHN model wiring | Add FASHN patch-embedding wrappers |
| Four target-only patch-mixer blocks | Primitive exists; architecture absent | Add distinct `x_patch_mixer.0..3` |
| Category conditioning | Generic embedding exists; no try-on category interface | Add row lookup and conditional-drop semantics |
| Pixel-space output | Precedents exist, including `FakeVAE` | Add FASHN-specific scale/channel/encoding handling |
| Image-conditioned, no-text generation | No FASHN entry path | Add a dedicated prepared-input entry point |
| FASHN schedule/CFG | General Euler/CFG exists, exact contract absent | Add an isolated reference-matching sampler |
| DWPose/YOLOX preprocessing | Not part of the current try-on/runtime pipeline | Reference preparation first; optional ORT frontend later |
| FASHN human parser | Absent | Separate dependency/export/port decision |
| Category-aware agnostic masking | Absent as this exact algorithm | Reuse reference first; native port requires its own tests |
| CLI/server try-on inputs | Absent | Add explicit mode and schemas |
| FASHN layer/image parity tests | Absent | Add fixtures and focused test targets |

No new fundamental GGML operation is obviously required for the main FASHN transformer. That is a source-level conclusion about expressing its graph, not proof that every dtype/backend combination will run correctly.

### 5.1 Why not simply use FluxRunner?

`FluxRunner` assumes a context-token input and calls `context->ne[1]` when constructing positions. Its outer model expects FLUX-specific input projections and optional text/vector/guidance components. FASHN has different patch projections and no text encoder.

Changing its width and depths is insufficient. Adding FASHN to `sd_version_is_flux` would also trigger unrelated text-encoder, VAE, reference-image, and denoiser choices.

Reuse **blocks**, not the entire FLUX workflow.

### 5.2 Why not SD3's MMDiTRunner?

"MMDiT" names a general architectural family. The current SD3 runner targets SD3 checkpoint structure and conditioning. FASHN's double/single-block topology and names are derived from FLUX. SD3's runner is not the appropriate outer model.

### 5.3 Existing block reuse is already a repository convention

`hunyuan.hpp` includes `model/diffusion/flux.hpp` and constructs `Flux::MLPEmbedder`, `DoubleStreamBlock`, `SingleStreamBlock`, and `LastLayer`.

Follow that precedent initially. A separate extraction into a shared `flux_blocks.h` is optional later and should not be mixed into the first FASHN feature unless necessary. Such an extraction must preserve FLUX2/Chroma/Hunyuan behavior and all constructor options. [S-hunyuan]

## 6. Proposed integration architecture

```text
Reference Python preparer            Optional native preparer
  DWPose + parser + transforms          ORT + OpenCV
              \                         /
               prepared input contract
                         |
       CLI adapter / server request / C API caller
                         |
       generate_try_on(sd_ctx_t*, sd_try_on_params_t*, ...)
                         |
        normalize bytes and validate fixed input canvas
                         |
           FASHN-specific sampling runtime
                         |
                DiffusionParams
           + FashnVTONDiffusionExtra
                         |
                 FashnVTONRunner
                         |
          FashnVTONModel + reused Flux blocks
                         |
        existing GGMLRunner / ModelManager / backends
                         |
          pixel conversion and person-output crop
```

The core library should not call Python, launch shell commands, download preprocessing models implicitly, or require ONNX Runtime just to execute already-prepared FASHN inputs.

### 6.1 Proposed files

| File | Responsibility |
|---|---|
| `src\model\diffusion\fashn_vton.h` | Configuration, patch embedding, model graph, runner |
| `src\runtime\fashn_vton.h` | Prepared tensor contract and sampling declarations |
| `src\runtime\fashn_vton.cpp` | Exact schedule, sequential CFG/Euler, validation/conversion helpers |
| `src\model.h` | Dedicated version and model-family predicate |
| `src\model_loader.cpp` | Detection before generic FLUX; model-specific conversion policy |
| `src\name_conversion.cpp` | Normalize known FASHN roots without rewriting internal names |
| `src\model\diffusion\model.hpp` | Add `FashnVTONDiffusionExtra` to the existing variant |
| `src\stable-diffusion.cpp` | Context initialization, capabilities, new public API dispatch, lifecycle |
| `include\stable-diffusion.h` | New additive C API types/functions |
| `src\model\vae\vae.hpp` | FASHN scale=1 if retaining the weightless `FakeVAE` adapter |
| `examples\common\common.h/.cpp` | Mode registration and shared request ownership/validation |
| `examples\common\try_on.h/.cpp` | Prefer a separate prepared-input adapter over growing generic img-gen parsing |
| `examples\cli\main.cpp` | Try-on mode options/dispatch/output metadata |
| `examples\cli\CMakeLists.txt` | Include any new example-common source files |
| `examples\server\runtime.h/.cpp` | Try-on capability and request structures |
| `examples\server\async_jobs.h/.cpp` | Owned try-on job payload, worker dispatch, cancellation |
| `examples\server\routes_sdcpp.cpp` | Try-on endpoint and capability metadata |
| `examples\server\CMakeLists.txt` | Include new shared example sources if needed |
| `scripts\prepare_fashn_vton.py` | Reference-compatible raw-image preprocessing, not a model implementation |
| `scripts\export_fashn_vton_reference.py` | Development-only weight/tensor fixture exporter |
| `tests\test_fashn_vton_*.cpp` | Proposed native metadata/graph/sampler tests |
| `tests\test_fashn_vton_reference.py` | Proposed comparisons against reference fixtures |
| `docs\fashn_vton.md` | Supported workflow, exact model, preprocessing/license limits, examples |
| README and relevant CLI/server docs | Feature registration and explicit support boundaries |

New headers should use `.h`, include guards, C++17, and the current model-specific `*Config` conventions.

The root CMake file already glob-includes `src\model\*\*.h/.cpp` and `src\runtime\*.h/.cpp`. A header-based new model and runtime source fit the existing source list. New example sources and proposed tests still need deliberate CMake wiring.

## 7. Checkpoint identification and configuration

### 7.1 Detection must precede generic FLUX

Currently, prefixed keys such as:

```text
model.diffusion_model.double_blocks.0.img_attn.qkv.weight
```

set the loader's `is_flux` flag. Without a FASHN-specific decision, a FASHN checkpoint supplied through `--diffusion-model` reaches the generic FLUX classification.

The raw checkpoint supplied through `-m` also needs explicit recognition because it does not carry that canonical prefix.

Support these two representations first:

```text
x_embedder.proj.weight
model.diffusion_model.x_embedder.proj.weight
```

Do not require a filename such as `fashn.safetensors` to identify the architecture.

Use two-stage recognition:

1. Identify the FASHN family from its characteristic roots, especially `garment_embedder.proj`, `x_patch_mixer`, and `y_embedder`.
2. Validate the exact released 1.5 preset before allocating/loading model tensors.

A recognizable but incomplete or differently shaped FASHN checkpoint must produce an explicit unsupported/malformed-FASHN error, not silently fall back to FLUX.

### 7.2 Sentinel shapes

| Key | PyTorch shape | Loader/GGML dimensions |
|---|---|---|
| `x_embedder.proj.weight` | `[1280,7,12,12]` | `[12,12,7,1280]` |
| `garment_embedder.proj.weight` | `[1280,4,12,12]` | `[12,12,4,1280]` |
| `t_embedder.mlp.in_layer.weight` | `[1280,256]` | `[256,1280]` |
| `y_embedder.weight` | `[4,1280]` | `[1280,4]` |
| `double_blocks.0.img_attn.qkv.weight` | `[3840,1280]` | `[1280,3840]` |
| `double_blocks.0.img_mod.lin.weight` | `[7680,1280]` | `[1280,7680]` |
| `single_blocks.0.linear1.weight` | `[8960,1280]` | `[1280,8960]` |
| `single_blocks.0.linear2.weight` | `[1280,6400]` | `[6400,1280]` |
| `final_layer.linear.weight` | `[432,1280]` | `[1280,432]` |
| `final_layer.adaLN_modulation.1.weight` | `[2560,1280]` | `[1280,2560]` |
| `patch_mixer_token` | `[1,1,432]` | `[432,1,1]` |

The safetensors reader already reverses dimension metadata. **Do not transpose all weight payloads again.**

### 7.3 Configuration blueprint

```cpp
struct FashnVTONConfig {
    int64_t hidden_size = 1280;
    int64_t mlp_hidden_size = 5120;
    int num_heads = 10;
    int head_dim = 128;
    int patch_size = 12;
    int input_height = 864;
    int input_width = 576;
    int patch_mixer_depth = 4;
    int double_depth = 8;
    int single_depth = 16;
    int category_rows = 4;
    int time_embed_dim = 256;
    int theta = 10000;
    std::vector<int> axes_dim = {16, 56, 56};

    static FashnVTONConfig detect_from_weights(
        const String2TensorStorage& tensors,
        const std::string& prefix);

    bool validate_v15(
        const String2TensorStorage& tensors,
        const std::string& prefix,
        std::string* error) const;
};
```

`detect_from_weights` should infer dimensions and block counts where possible, following `docs\model_config.md`. `validate_v15` must verify the complete preset rather than treating default field values as evidence of support.

Validation must cover:

- Contiguous indices 0-3, 0-7, and 0-15 for the three block families.
- Both streams in every double block.
- Bias presence and exact sizes.
- Q/K norm vectors of 128 elements.
- Every required tensor, not just the sentinels.
- `hidden_size == num_heads * head_dim`.
- Axis sum equals head dimension and each axis dimension is even.
- MLP and final-output dimensional consistency.
- Missing or unexpected guidance-embedding variants.
- Fixed model canvas; custom source configurations are not implicitly supported.
- No normalization-name collision after prefix canonicalization.

The existing `ModelManager::validate_registered_tensors` checks the registered tensors against metadata. It does not by itself establish an exact file-to-model key-set match. Add a FASHN-specific key-set check: **365 inference tensors plus the known optional 432-element unused buffer**. Reject or explicitly diagnose additional tensors belonging to an unsupported architecture variant.

Do not globally ignore every name containing `token`. Specifically recognize the exact unused `patch_mixer_token` buffer for this family.

### 7.4 Weight naming policy

Preserve all FASHN suffixes:

```text
t_embedder.mlp.in_layer.weight
y_embedder.weight
x_embedder.proj.weight
garment_embedder.proj.weight
x_patch_mixer.0.linear1.weight
double_blocks.0.img_mod.lin.weight
single_blocks.0.linear1.weight
final_layer.linear.weight
```

Only prepend/remove the recognized canonical model root. There is no need to rename `t_embedder` to `time_in` or `garment_embedder` to `txt_in`; the new model block can register the original names directly.

Keep `sd_version_is_flux(VERSION_FASHN_VTON_1_5)` false. Add a dedicated predicate and include the new version in `sd_version_is_dit`.

## 8. Native graph design and code sketches

### 8.1 Internal tensor order

`sd::Tensor` and GGML use lowest/most-contiguous dimension first:

```text
PyTorch RGB:       [B,3,H,W]       -> sd/GGML [W,H,3,B]
PyTorch tokens:    [B,L,D]         -> sd/GGML [D,L,B]
PyTorch category:  [B]            -> sd/GGML [B], integer
PyTorch PE:        [B,1,L,64,2,2]  -> existing Rope matrix [2,2,64,L], B=1
```

Concatenate input image channels on GGML dimension 2. Concatenate token sequences on GGML dimension 1. Concatenate Q/K/V token axes using the existing FLUX block implementation.

Do not infer the logical batch rank from `ggml_n_dims`: trailing singleton dimensions can disappear. Restore output to logical `[576,864,3,1]`.

### 8.2 Patch embeddings: avoid an accidental F16-only reference path

The existing `Conv2d` supports these kernel/stride/channel shapes, but its parameter allocation currently fixes convolution weights to F16. That is a precision difference from a strict all-F32 CPU reference.

A useful FASHN-local solution is to express the non-overlapping convolution as:

```text
DiT::patchify(input, 12, 12, patch_last=true)
    -> linear projection using a reshaped view of the 4D kernel
```

This is mathematically the same convolution with stride equal to kernel size, no padding, and no dilation. Keep the registered weight **4D** so it matches the original checkpoint; only reshape it into a matrix view inside the graph.

Illustrative block:

```cpp
class FashnPatchEmbed : public UnaryBlock {
    int64_t in_channels_;
    int64_t hidden_size_;
    int patch_size_;

protected:
    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensors,
                     const std::string prefix) override {
        const ggml_type type =
            get_type(prefix + "weight", tensors, GGML_TYPE_F32);
        // Family validation restricts this kernel to supported floating types.
        params["weight"] = ggml_new_tensor_4d(
            ctx, type, patch_size_, patch_size_,
            in_channels_, hidden_size_);
        params["bias"] = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, hidden_size_);
    }

public:
    FashnPatchEmbed(int64_t in_channels,
                   int64_t hidden_size,
                   int patch_size)
        : in_channels_(in_channels),
          hidden_size_(hidden_size),
          patch_size_(patch_size) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* image) override {
        auto patches = DiT::patchify(
            ctx->ggml_ctx, image, patch_size_, patch_size_, true);
        auto weight = ggml_reshape_2d(
            ctx->ggml_ctx, params.at("weight"),
            in_channels_ * patch_size_ * patch_size_,
            hidden_size_);
        return ggml_ext_linear(
            ctx->ggml_ctx, patches, weight, params.at("bias"), true);
    }
};
```

The flat patch index is:

```text
q = ((channel * 12) + patch_row) * 12 + patch_column
```

This must match both convolution flattening and final unpatchification. `patch_last=true` in the existing helper matches this order.

This wrapper should not pretend to support arbitrary quantized convolution tensors or LoRA injection initially. Validate permitted types and reject incompatible options. Later, compare this implementation with a native-convolution fast path and select based on actual performance/precision.

### 8.3 Model composition

The proposed `FashnVTONModel` should register these children:

```cpp
// Constructor sketch inside a GGMLBlock-derived FashnVTONModel.
blocks["x_embedder.proj"] =
    std::make_shared<FashnPatchEmbed>(7, 1280, 12);
blocks["garment_embedder.proj"] =
    std::make_shared<FashnPatchEmbed>(4, 1280, 12);
blocks["t_embedder.mlp"] =
    std::make_shared<Flux::MLPEmbedder>(256, 1280, true);
blocks["y_embedder"] = std::make_shared<Embedding>(4, 1280);

for (int i = 0; i < 4; ++i) {
    blocks["x_patch_mixer." + std::to_string(i)] =
        std::make_shared<Flux::SingleStreamBlock>(1280, 10, 4.0f, i);
}
for (int i = 0; i < 8; ++i) {
    blocks["double_blocks." + std::to_string(i)] =
        std::make_shared<Flux::DoubleStreamBlock>(
            1280, 10, 4.0f, i, true);
}
for (int i = 0; i < 16; ++i) {
    blocks["single_blocks." + std::to_string(i)] =
        std::make_shared<Flux::SingleStreamBlock>(1280, 10, 4.0f, i);
}
blocks["final_layer"] =
    std::make_shared<Flux::LastLayer>(1280, 12, 3);
```

In implementation, replace these literals with validated `config` fields. The explicit values here show how the current constructor signatures map to FASHN.

Required settings: no pruned modulation, no shared modulation, no YakMLP, no SwiGLU, biases enabled, QKV bias enabled, tanh-approximate GELU, non-affine LayerNorm with epsilon `1e-6`.

The final-layer constructor takes patch size and output channels; pass `(1280,12,3)`, not `(1280,12,432)`.

### 8.4 Forward graph outline

The following sketch omits routine child casts and graph allocation; each named block is the registered child above:

```cpp
auto target_input = ggml_concat(gctx, noisy_rgb, ca_rgb, 2);
target_input = ggml_concat(gctx, target_input, person_pose, 2);
auto target = x_embedder->forward(ctx, target_input);

auto garment_input = ggml_concat(gctx, garment_rgb, garment_pose, 2);
auto garment = garment_embedder->forward(ctx, garment_input);

auto time_features = ggml_ext_timestep_embedding(
    gctx, timesteps, 256, 10000, 1000.0f);
auto vec = time_mlp->forward(ctx, time_features);
auto label = category_embedder->forward(ctx, category_ids);
label = ggml_reshape_2d(gctx, label, 1280, 1); // B=1 milestone
vec = ggml_add(gctx, vec, label);

for (const auto& mixer : patch_mixers) {
    target = mixer->forward(ctx, target, vec, target_pe);
}
for (const auto& block : double_blocks) {
    auto streams = block->forward(ctx, target, garment, vec, joint_pe);
    target = streams.first;
    garment = streams.second;
}

const int64_t garment_tokens = garment->ne[1];
auto joint = ggml_concat(gctx, garment, target, 1);
for (const auto& block : single_blocks) {
    joint = block->forward(ctx, joint, vec, joint_pe);
}
auto target_only = ggml_cont(
    gctx, ggml_ext_slice(gctx, joint, 1,
                        garment_tokens, joint->ne[1]));
auto patch_velocity = final_layer->forward(ctx, target_only, vec);
return DiT::unpatchify(gctx, patch_velocity, 72, 48, 12, 12, true);
```

`target_pe` contains 3,456 spatial positions. `joint_pe` contains two identical spatial grids. No causal mask or padding-token mask is used; the padded pixels remain ordinary image inputs.

### 8.5 Typed extra parameters

Extend the existing `DiffusionExtraParams` variant rather than overloading `context`, `y`, or `ref_latents` with unexplained positional meanings:

```cpp
struct FashnVTONDiffusionExtra {
    const sd::Tensor<float>* ca_image = nullptr;
    const sd::Tensor<float>* garment_image = nullptr;
    const sd::Tensor<float>* person_pose = nullptr;
    const sd::Tensor<float>* garment_pose = nullptr;
    const sd::Tensor<int32_t>* garment_categories = nullptr;
};
```

Keep noisy RGB in `DiffusionParams::x` and time in `DiffusionParams::timesteps`.

The runner should check the variant and required tensors, log meaningful errors, and return an empty `sd::Tensor<float>` on failure. Validate the public request before graph construction; do not use assertions as the only protection against bad user input.

### 8.6 Runner lifecycle

Follow existing runners:

- Detect config once, validate it, initialize model parameters with the canonical prefix.
- Use `make_input` and `set_backend_tensor_data`; do not assume tensors are host-addressable on all backends.
- Build the graph with `new_graph_custom`, expand the final output, and use `GGMLRunner::compute`.
- Restore trailing singleton dimensions on returned RGB velocity.
- Register parameters with `ModelManager`.
- Use request-level RAII to call `runner_end()` after all steps, on success, failure, or cancellation.
- Start with request-scoped buffers and no cross-request condition cache.
- Use `capture_tensor` and the existing backend-evaluation callback for diagnostic fixture comparison.

During sampling, call the base runner with `auto_runner_end=false` so it can reuse workspace/weights across steps. The enclosing request owns the corresponding cleanup.

Backend graph-plan caching must not accidentally reuse stale input bindings between conditional and unconditional passes. Same shapes do not mean same values.

For the patchify-plus-linear embedding, specifically exercise parameter discovery through the reshaped weight view: lazy loading, mmap, and graph-cut execution must resolve the view back to the registered 4D parameter. Do not register a second independently owned 2D copy with the same checkpoint name.

## 9. Sampling implementation plan

### 9.1 Recommended initial design: an isolated direct-time sampler

Implement the short FASHN loop in `src\runtime\fashn_vton.cpp`, called by the dedicated try-on entry point. Reuse the existing RNG, tensors, runner, logging/progress, and cancellation facilities.

This avoids teaching the generic text/image-guidance machinery that an empty text condition can still require a fully populated, learned null-image branch.

Do not replace or globally modify `sample_euler`, `FluxFlowDenoiser`, or SD's CFG behavior for this feature.

A later consolidation into the general sampler is possible using the sigma adapter derived in section 4.3. It is not necessary for the first correct implementation and must not be combined with the direct-time path in a way that transforms time/sign twice.

### 9.2 Schedule helper sketch

```cpp
static bool make_fashn_timesteps(int steps,
                                float mu,
                                std::vector<float>& times) {
    times.clear();
    if (steps <= 0 || !std::isfinite(mu)) {
        LOG_ERROR("FASHN needs positive steps and a finite time shift");
        return false;
    }
    const float a = std::exp(-mu);
    if (!std::isfinite(a) || a <= 0.0f) {
        LOG_ERROR("FASHN time shift is outside the supported numeric range");
        return false;
    }
    times.resize(static_cast<size_t>(steps) + 1);
    times.front() = 0.0f;
    times.back() = 1.0f;
    for (int i = 1; i < steps; ++i) {
        const float u = static_cast<float>(i) / steps;
        times[i] = (a * u) / ((1.0f - u) + a * u);
    }
    for (size_t i = 1; i < times.size(); ++i) {
        if (!std::isfinite(times[i]) || times[i] <= times[i - 1]) {
            LOG_ERROR("FASHN timestep schedule is not strictly increasing");
            times.clear();
            return false;
        }
    }
    return true;
}
```

The algebraically stable formula avoids `1/t` at the endpoints. It may round slightly differently from PyTorch's original operation sequence. Test against recorded float32 schedules, including 1, 4, 20, 30, and 50 steps. For exact diagnostic runs, allow fixture timesteps through an internal test interface.

Apply a documented resource limit to externally supplied step counts before allocation; do not let a server request allocate an unbounded vector.

### 9.3 Sequential CFG blueprint

Precompute normalized conditional tensors and exact-zero unconditional tensors once per request.

```cpp
// Structural sketch: run_branch checks graph output shape and reports errors.
for (int i = 0; i < steps; ++i) {
    if (cancel_all_requested()) {
        return cancelled_result();
    }

    auto vc = run_branch(x, times[i], conditional_inputs);
    if (vc.empty()) {
        LOG_ERROR("FASHN conditional forward failed at step %d", i);
        return failed_result();
    }

    const bool skip_cfg =
        skip_cfg_last_n_steps > 0 &&
        i >= steps - skip_cfg_last_n_steps;

    sd::Tensor<float> velocity;
    if (skip_cfg || guidance_scale == 1.0f) {
        velocity = std::move(vc);
    } else {
        auto vu = run_branch(x, times[i], unconditional_inputs);
        if (vu.empty()) {
            LOG_ERROR("FASHN unconditional forward failed at step %d", i);
            return failed_result();
        }
        velocity = vu + (vc - vu) * guidance_scale;
    }

    x += velocity * (times[i + 1] - times[i]);
    report_completed_step(i + 1, steps);
}
```

`run_branch`, cancellation-result helpers, and progress helpers above are proposed local integration helpers, not existing public APIs.

Check cancellation before both expensive branches. Shape/non-finite checks must not silently replace failed predictions with zeros.

The reference computes both branches even on the final step and then discards the unconditional output. Omitting that unused evaluation is mathematically equivalent. At 30 steps and default settings, sequential execution needs **59 branch evaluations**, instead of 60 branch-equivalents in the batched reference. With CFG=1, only 30 conditional evaluations are needed.

Sequential branches are recommended initially because `FluxRunner` itself asserts a single-image batch and existing shared blocks have not been validated here for FASHN's doubled batch. Add fused batch-CFG only after explicit B=2/B=8 tests.

### 9.4 Multiple samples and RNG

Support `num_samples=1..4` at the API level, initially by running independent B=1 trajectories.

For parity, generate the complete initial noise for all requested samples as one logical `[B,3,864,576]` draw and then split it. Do not silently change the reference's batch RNG ordering to a per-image `seed+i` policy.

The existing `MT19937RNG` explicitly aims to imitate PyTorch CPU `randn`, making `CPU_RNG` the natural first choice. Still, use exported noise for parity fixtures: matching seed numbers across PyTorch CPU, CUDA, native GGML RNG, dtypes, and versions is not a sufficient guarantee.

Separate:

- Same-seed reproducibility within the native build.
- Same-initial-noise numerical parity with the reference.
- Cross-device exact RNG compatibility, only if tested.

## 10. Preprocessing plan

### 10.1 Three independently named support levels

| Level | Inputs | Runtime dependencies | Completion meaning |
|---|---|---|---|
| A: Native prepared-input inference | Four model-ready images + category + crop metadata | stable-diffusion.cpp/GGML only | Main FASHN model is native |
| B: Reference preprocessing workflow | Raw person and garment images | Python, original pose/parser stack, then native inference | End-to-end local try-on works, preprocessing is not native |
| C: Native raw-image workflow | Raw person and garment images | Optional native ORT/OpenCV preprocessing tool plus native inference | No Python required at inference time |

A fourth, much larger project would be porting YOLOX, RTMPose/DWPose, and SegFormer themselves into GGML. Do not make that a prerequisite for the main model port, and do not imply level C is entirely GGML.

### 10.2 Exact mode/dependency matrix

| Person mode | Garment photo | Person parsing affects output? | Garment parsing affects output? | Garment pose |
|---|---|---|---|---|
| Segmentation-free | Flat-lay | No | No | Blank raster |
| Segmentation-free | Model-worn | No | Yes | DWPose |
| Masked/agnostic | Flat-lay | Yes | No | Blank raster |
| Masked/agnostic | Model-worn | Yes | Yes | DWPose |

The actual reference eagerly initializes the parser and computes both segmentation maps in every mode. When a map is unused, a future implementation can skip that computation without changing image inputs. Do this only after prepared-input equivalence is demonstrated.

For a parser-free initial raw-image mode, segmentation-free person + flat-lay garment is the obvious candidate. It still needs person pose detection; it is not a completely preprocessing-free workflow.

### 10.3 Reference-preparation script

Create a preparer that follows the source operations but does not instantiate the approximately 1B-parameter try-on model merely to discover `(864,576)`.

It should:

1. Load RGB images in a documented orientation/color policy.
2. Use the pinned transforms, DWPose wrapper, and parser package.
3. Set the fixed canvas from the validated 1.5 preset.
4. Save lossless RGB/one-channel PNG files.
5. Save a schema-versioned manifest with category, mode, padding, image dimensions, hashes, and source/model/dependency revisions.
6. Optionally save normalized float32 tensors and initial noise for development tests.
7. Fail explicitly on missing weights or preprocessing errors.

Proposed manifest:

```json
{
  "schema": "sdcpp.fashn-vton.prepared.v1",
  "model_family": "fashn-vton-1.5",
  "canvas": {"width": 576, "height": 864},
  "category": "tops",
  "garment_photo_type": "model",
  "segmentation_free": true,
  "images": {
    "ca": "ca.png",
    "garment": "garment.png",
    "person_pose": "person_pose.png",
    "garment_pose": "garment_pose.png"
  },
  "person_padding": {"left": 0, "top": 0, "right": 0, "bottom": 0},
  "normalization": "uint8_div_127.5_minus_1"
}
```

The zero padding values above are an example for a matching aspect ratio, not defaults to use for every input.

Keep raw-image provenance and hashes in a sidecar without storing sensitive personal file paths in generated PNG metadata by default. Relative manifest paths resolve against the manifest's directory.

### 10.4 Resize and padding semantics

Preserve the source's two distinct resize stages:

- Before detection/parsing: PIL LANCZOS, aspect fit inside 864 x 864, no upsampling; dimensions truncated with `int`.
- Before the transformer: OpenCV aspect fit into width=576/height=864, with upsampling permitted.

For RGB in the second stage:

- Downsampling: `INTER_AREA`.
- Upsampling: `INTER_LANCZOS4`.
- Exact scale 1: unchanged.

For pose rasters in the second stage:

- `INTER_NEAREST_EXACT`.

Padding:

- Constant black, zero byte value.
- Left/top receive integer floor of half the required padding.
- Right/bottom receive the remainder.
- Record padding from the person/clothing-agnostic image only for output unpadding.

The reference **crops off padding after generation**. It does not resize the image back to the original photograph's dimensions. The internal canvas is 576 x 864, but the returned image may be narrower or shorter.

Do not silently use the generic SD image-resize helper, letterbox defaults from a different detector, or a convenient bicubic interpolation as a substitute.

Orientation handling should be explicit: the VTON pipeline receives PIL images and does not itself apply a global EXIF transpose. If the new raw-image frontend normalizes EXIF orientation, the reference fixture must use the same normalized input. Do not call that an exact upstream behavior without matching the ingress policy.

### 10.5 DWPose details that must not be simplified

The source calls a YOLOX-L detector, then an ONNX pose model:

- Detector input: 640 x 640, top-left image placement, padding value 114, OpenCV linear resize, float32 CHW.
- Detector decode: strides 8/16/32, exponentiated width/height predictions, class-aware NMS.
- Thresholds: NMS 0.45, initial score 0.1, final person confidence greater than 0.3 and class ID zero.
- If no box remains, pose preprocessing uses the full image.
- Pose input dimensions are taken from the ONNX session, not guessed from the filename.
- Pose crop: bounding-box padding factor 1.25, aspect-corrected affine transform, linear interpolation.
- Preserve the exact BGR/RGB convention used by these exported models; do not add an extra "helpful" channel swap.
- Pose normalization uses the source's `[123.675,116.28,103.53]` mean and `[58.395,57.12,57.375]` standard deviation.
- SimCC decode: argmax of X/Y outputs, split ratio 2, confidence from the lower of X/Y maxima, then inverse coordinate mapping.
- Compute the neck point and map MMPose indices to OpenPose indices.
- Keypoint visibility threshold: 0.3.
- Use the reference single-person candidate-selection heuristic.

The existing native `YOLOv8` ADetailer detector is **not** a drop-in implementation of YOLOX-L.

Drawing is part of the learned conditioning:

- Body: grayscale values from `linspace(20,240,18)`, ellipse/polygon limbs, stick width 4, limb factor 0.6, circles of radius 4.
- Hands: edge values from 160 to 220, thickness 2, keypoint circles value 240/radius 4.
- Face: circles value 200/radius 3.
- The render order and overlap semantics matter.

A standard RGB OpenPose skeleton converted to grayscale is not equivalent.

The current best-candidate function contains an assumption about using the first candidate's validity mask when computing candidate areas. Preserve it for reference parity or document an intentional behavioral divergence; do not mix a preprocessing "fix" into model parity debugging. [F-dwpose] [F-dwpose-det] [F-dwpose-pose] [F-drawing]

### 10.6 Human parsing and masked mode

The parser is SegFormer-B4 with a MiT-B4 encoder and 18-class decoder. The inspected model config has:

```text
stage depths:       [3,8,27,3]
stage hidden sizes: [64,128,320,512]
attention heads:    [1,2,5,8]
spatial reductions: [8,4,2,1]
patch sizes:        [7,3,3,3]
strides:            [4,2,2,2]
decoder width:      768
```

Its native preprocessing requirement is distinct from VTON preprocessing:

- Resize directly to 384 x 576 using OpenCV `INTER_AREA`.
- Normalize RGB with ImageNet mean/std.
- Run the segmentation model.
- Bilinearly upsample **logits** to the input image's pre-resized dimensions with `align_corners=false`.
- Then argmax class IDs.

Do not argmax at low resolution and resize labels instead.

Category selection is:

| Category | Coverage | Garment labels |
|---|---|---|
| tops | upper | top, dress, scarf |
| bottoms | lower | skirt, pants, belt |
| one-pieces | full | top, dress, scarf, skirt, pants, belt |

For model-worn garments, replace pixels outside selected labels with byte value 127.

For masked person mode, port the full existing algorithm:

- Add arms/torso for upper/full coverage and legs for lower/full coverage.
- Scale spatial parameters relative to image height / 864.
- Buffer dilation using `max(1,int(4*scale))`.
- Bounding-rectangle mask.
- Contour-following mask using an elliptical dilation, signed-distance transforms, Gaussian blur, and hole filling.
- Brush radius `max(1,int(18*scale))`.
- Remove bounded-mask expansion farther than the scaled threshold of 100.
- Asymmetric dilation: left/right `int(33*scale)`, up/down `int(16*scale)`.
- Preserve identity labels and appropriate limbs/hands/feet.
- Final mask is the buffer mask OR the expanded mask excluding identity regions.
- Fill masked pixels with 127.

Keep the exact mask operations and order; a generic box mask, SAM mask, or ordinary inpaint mask is a different preprocessing distribution. [F-agnostic] [F-masks] [P-parser] [P-labels]

### 10.7 Optional native preprocessing executable

Propose an off-by-default target such as:

```text
SD_BUILD_VTON_PREPROCESSOR=ON
examples\try_on_preprocess\
```

It would link ONNX Runtime and OpenCV separately from the main library and emit the same prepared manifest.

Use the official DWPose/YOLOX ONNX files. Export the pinned parser to ONNX as a separate, versioned preparation step, retaining pre/postprocessing in the native frontend. An ONNX export is **not yet validated** by this investigation.

Validate the parser export's logits and masks, not just successful session construction. Package licenses independently. Do not automatically fetch a different parser with different label IDs.

## 11. Public C API and context integration

### 11.1 Additive API sketch

Prefer new types and functions over appending fields to the existing unversioned `sd_img_gen_params_t` or `sd_ctx_params_t`. Existing applications must not be required to pass a larger old struct.

```c
enum sd_try_on_category_t {
    SD_TRY_ON_TOPS = 1,
    SD_TRY_ON_BOTTOMS = 2,
    SD_TRY_ON_ONE_PIECES = 3
};

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} sd_try_on_crop_t;

typedef struct {
    size_t struct_size;
    sd_image_t ca_image;
    sd_image_t garment_image;
    sd_image_t person_pose;
    sd_image_t garment_pose;
    enum sd_try_on_category_t category;
    int steps;
    float guidance_scale;
    float time_shift_mu;
    int skip_cfg_last_n_steps;
    int64_t seed;
    int num_samples;
    bool crop_output;
    sd_try_on_crop_t output_crop;
} sd_try_on_params_t;

SD_API void sd_try_on_params_init(sd_try_on_params_t* params);
SD_API bool sd_ctx_supports_try_on(const sd_ctx_t* ctx);
SD_API bool generate_try_on(sd_ctx_t* ctx,
                            const sd_try_on_params_t* params,
                            sd_image_t** images_out,
                            int* num_images_out);
```

This sketch follows the current bool/out-parameters generation style. The core accepts **prepared** inputs. A raw RGB photograph is not accepted merely because it has the expected dimensions.

The public category enum intentionally does not expose the internal null row as a normal user category.

`struct_size` provides a basis for compatible extension. Implement size validation before accessing fields and document the policy for future larger structures.

### 11.2 Defaults and validation

Defaults:

```text
steps                   30
guidance_scale          1.5
time_shift_mu           1.5
skip_cfg_last_n_steps   1
seed                    42
num_samples             1
crop_output             false, unless provided by a prepared manifest
```

Validate:

- Correct model family and initialized context.
- Non-null buffers, all images exactly 576 x 864.
- RGB images exactly 3 channels; poses exactly 1 channel.
- Category exactly 1, 2, or 3.
- Positive bounded step count; finite guidance and time-shift values.
- Non-negative guidance scale.
- `0 <= skip_cfg_last_n_steps <= steps`.
- `1 <= num_samples <= 4`.
- Valid crop rectangle inside the model canvas, using overflow-safe bounds.
- No unsupported sampler, init-noise strength, ControlNet, LoRA, video, hires, or text-conditioning request.

On failure: initialize outputs to null/zero, log the reason, and return false. Never report success with an uninitialized or fallback black image.

Input buffers are borrowed only for the call. Output allocation/freeing should follow the current library's image ownership convention; callers and example owners free each image data buffer and the returned array. If adding a public result-free helper is desirable, make it additive and document its scope rather than silently changing ownership.

Support existing cancellation semantics: abort all versus finish the current sample and skip later samples. Document whether partial results are returned under each mode.

### 11.3 Context loader changes that are easy to miss

The current initializer unconditionally creates/registers a conditioner, and parts of flash-attention setup dereference it. For FASHN:

- Instantiate `FashnVTONRunner` and leave `cond_stage_model` absent.
- Guard conditioner registration, memory configuration, and global flash-attention setup.
- Do not initialize CLIP/T5/VLM models or require their weights.
- Retain a weightless `FakeVAE` adapter if that minimizes common lifecycle assumptions; set its scale to 1 and image channels to 3.
- Do not run prepared conditioning through VAE encoding or text-conditioning machinery.
- Do not use `FakeVAE` to apply an additional normalization to already-normalized inputs.
- Skip the generic denoiser initialization/use for the dedicated FASHN runtime.
- Keep shared weight registration, metadata validation, mmap, backend ownership, and memory accounting.
- Reject incompatible external model components rather than constructing unused ones.
- Update `model_version_to_str` and its alignment with the version enum.

The current capability rule treats every non-video model as supporting generic image generation. Change it for FASHN: expose `try_on`, and reject generic `generate_image`/`img_gen` until there is an explicit compatible contract. This prevents entry into code that assumes a text conditioner exists.

Audit public default-sampler/scheduler getters and server capability generation so they do not publish misleading generic defaults for a try-on-only context. FASHN defaults come from `sd_try_on_params_init`.

Optional previews should show current RGB state or a documented predicted-clean preview without invoking a learned VAE. Do not treat velocity itself as an RGB image.

## 12. CLI and server plan

### 12.1 CLI

Add `TRY_ON` to the example mode enum and update both `modes_str` and `SD_ALL_MODES_STR`, help text, dispatch, and validation.

Use an explicit prepared-bundle option, for example `--try-on-inputs`, so users do not confuse prepared tensors with ordinary `--init-img`/`--ref-image`.

Proposed commands, **after implementation**:

```powershell
Set-Location C:\source\stable-diffusion.cpp

python .\scripts\prepare_fashn_vton.py `
  --person C:\data\person.png `
  --garment C:\data\garment.png `
  --category tops `
  --garment-photo-type model `
  --segmentation-free `
  --weights-dir C:\models\fashn-preprocessing `
  --output-dir .\outputs\fashn-case

.\build\bin\Release\sd-cli.exe `
  --mode try_on `
  --diffusion-model C:\models\fashn-vton-1.5\model.safetensors `
  --try-on-inputs .\outputs\fashn-case\inputs.json `
  --steps 30 --cfg-scale 1.5 --seed 42 --threads 8 `
  --mmap --diffusion-fa `
  --output .\outputs\fashn-result.png
```

For a native integration smoke test, start with one step and CFG=1 at the full supported canvas. That only checks execution; it is not a quality result.

Do not reuse the earlier 256 x 256 SD smoke test as a FASHN test: it is neither the supported canvas nor divisible by patch size 12.

Mode-specific defaults need explicit handling. Existing generic generation defaults must not accidentally supply 20 steps, 512 x 512, strength 0.75, or a text CFG default. Preserve explicitly supplied scalar options; do not overwrite `--steps` after argument parsing.

Implementation options are a small mode-specific defaults pass with tracking of explicitly set values, or a dedicated try-on argument adapter. Do not register duplicate `--steps` options in two option tables.

No dummy prompt should be required. Reject irrelevant prompt/sampler/hires options rather than silently ignoring them.

Use lossless PNG for prepared bundles. The current local build has `SD_WEBP=OFF` and `SD_WEBM=OFF`; the upstream FASHN example images are WebP. The reference preparer can read them with Pillow and write PNG without changing the native model build.

### 12.2 Server

Add a proposed endpoint:

```text
POST /sdcpp/v1/try_on
```

Example request shape:

```json
{
  "ca_image": "<base64 RGB PNG, already resized/padded>",
  "garment_image": "<base64 RGB PNG, already processed>",
  "person_pose": "<base64 one-channel PNG>",
  "garment_pose": "<base64 one-channel PNG>",
  "category": "tops",
  "steps": 30,
  "guidance_scale": 1.5,
  "time_shift_mu": 1.5,
  "skip_cfg_last_n_steps": 1,
  "seed": 42,
  "num_samples": 1,
  "output_format": "png"
}
```

Add an optional explicit crop object to match the prepared manifest. Keep the API's RGB/grayscale decoding strict; do not turn a color skeleton into a different grayscale representation.

Integrate with the current asynchronous job system:

- `AsyncJobKind::TryOn`.
- A request object that owns all decoded images.
- Queue admission and existing limits.
- Worker dispatch to `generate_try_on`.
- Progress, cancellation, result encoding, and retention.
- The existing 202 response/polling pattern.

Publish actual capabilities:

```text
mode: try_on
prepared_inputs: true
raw_image_preprocessing: false, unless the native frontend is integrated
text_prompt: false
categories: tops, bottoms, one-pieces
canvas: 576 x 864
```

Do not initially overload OpenAI image edits or A1111 img2img routes with hidden category/pose semantics. Those handlers currently have different validation and request contracts. Return a clear unsupported-mode error for them on a FASHN-only context.

Frontend changes are separate work because the embedded frontend is a submodule. A new backend endpoint does not automatically create a functional browser try-on UI.

Do not log base64 inputs, personal images, or unnecessary local paths. Apply byte/dimension/step/sample limits before expensive work, and do not accept arbitrary server-local paths as a substitute for uploaded inputs.

## 13. Precision, conversion, and optimization

### 13.1 Precision ladder

Use this order:

1. CPU F32 reference from the released BF16 checkpoint, with values expanded losslessly to F32.
2. Native F32 weights/activations, manual attention, no approximation caches.
3. Original BF16 checkpoint loading with native execution.
4. Flash-attention comparison.
5. Q8_0 transformer matrices.
6. Q5/Q4 only after explicit quality comparisons.
7. CUDA/Vulkan and other backend validation.

The current generic converter protects tensors whose names include `x_embedder`, `t_embedder`, `y_embedder`, and `final_layer` from conversion. A `--type f32` invocation therefore does **not** necessarily produce an entirely F32 file.

For the strict reference milestone, use a development exporter that explicitly casts every floating tensor to F32 and preserves names/shapes:

```python
# Development preparation sketch, not executed during this investigation.
from safetensors.torch import load_file, save_file

state = load_file(input_checkpoint, device="cpu")
expanded = {
    name: value.float().contiguous() if value.is_floating_point() else value
    for name, value in state.items()
}
save_file(expanded, output_checkpoint)
```

Keep the production path capable of reading the official safetensors directly. A GGUF conversion should be optional, not a prerequisite.

### 13.2 Quantization policy

Initially keep these in floating precision:

- Both patch kernels and their biases.
- Time embedding MLP.
- Category table.
- Q/K norm scales.
- All biases.
- Final projection and final modulation.
- Preferably modulation matrices for the first quantized comparison.

Quantize attention/MLP matrices in the double blocks, single blocks, and patch mixer first.

Extend the model-specific policy to recognize `garment_embedder`; it is not named in the current explicit protected-embedder list. The 4D kernel's lowest dimension of 12 already prevents many block-quantized formats, but relying on an incidental divisibility failure is not a clear model policy.

The hidden matrix dimensions 1280, 3840, 5120, 6400, and 8960 are compatible with common 32/256 block widths. Validate each tensor's actual supported type and backend; do not infer end-to-end support just from divisibility.

GGUF round-trip tests should preserve canonical names, shape metadata, the unused-buffer policy, and detection. Avoid adding a new GGUF metadata subsystem merely to carry an architecture label when existing shape/name detection suffices. If provenance metadata is added, use the existing reader/writer plumbing and define behavior when it is absent.

### 13.3 Attention and memory

At the joint sequence length, one F32 score matrix costs:

```text
B * heads * L * L * 4
= B * 10 * 6912 * 6912 * 4 bytes
= B * 1.779785 GiB
```

That is **one intermediate**, not total model memory. Batched conditional/unconditional inference doubles it. MLP/QKV intermediates, activations, workspace, weights, and allocator overhead are additional.

The current flash helper:

- Uses existing backend support checks.
- Casts K/V to F16.
- Requests F32 precision for the flash operation.
- Falls back to explicit attention if necessary.

Consequently `--diffusion-fa` is not proof that flash was selected, and enabling it is also a numerical change to validate.

Graph cutting/layer offload cannot by itself make an indivisible quadratic attention operation smaller. If full-resolution memory remains a problem, investigate backend flash support before attempting a new attention kernel.

### 13.4 Safe caching opportunities

After parity:

- Cache spatial PE tables per shape/config.
- Cache conditional and unconditional initial garment embeddings separately.
- Cache category vectors.
- Optionally split target patch projection into changing noisy-RGB contribution and static person/pose contribution, with the bias added exactly once.

Do **not** cache the garment stream through double blocks: its values change with time and attention to the changing target image.

Do **not** reuse modulation-dependent patch-mixer output across timesteps.

Condition cache keys must include request identity/input content, conditional versus unconditional branch, category where relevant, dtype/backend, and any weight-adapter epoch. Clear request-owned caches on cancellation/failure.

EasyCache, TeaCache-style approximation, TaylorSeer, skip-layer guidance, LoRA, and arbitrary schedulers are separate optional capabilities, not defaults for the parity milestone.

## 14. Validation and reference fixtures

### 14.1 Test infrastructure reality

This checkout has no local `test` directory, and its root `SD_BUILD_TESTS` option is commented out. Do not claim that an existing comprehensive model unit-test suite will catch this work.

Propose a small native test target using the existing compiler/CMake and a Python comparison tool using the reference project's existing pytest ecosystem. Test infrastructure changes must be explicit. No new build/test dependencies were installed for this planning task.

Keep binary fixtures and full weights out of the repository. Store metadata/small synthetic fixtures in tests where practical; gate full-checkpoint tests behind explicit local paths or artifact setup.

### 14.2 Fixture layers

| Level | Fixture/check | Primary failure isolated |
|---|---|---|
| Metadata | All names, shapes, dtypes, expected key set | Detection/canonicalization/missing weights |
| Integer layout | Patchify/unpatchify round trip with coordinate-coded pixels | Channel/patch/sequence permutation |
| Preprocessing | Lossless prepared RGB/pose arrays and padding | Resize, skeleton, masks, normalization |
| Primitive | Time embedding, category lookup, norm, modulation, RoPE | Local numerical mismatch |
| Block | One patch mixer, one double block, one single block | QKV order, attention, residual/gating |
| Full forward | Same `x`, time, conditions; conditional and null velocity | Whole-network wiring |
| Euler step | Same velocity/noise/schedule | Sign, direction, CFG, updates |
| Full trajectory | Shared initial noise and all timesteps | Accumulated numerical differences |
| End to end | Real person/garment, final floats, PNG and crop | Workflow correctness |

### 14.3 What to export from PyTorch

Record:

- Original checkpoint revision and file hash.
- All preprocessing inputs, maps, and transform metadata.
- Initial noise, normalized image tensors, category IDs, and times.
- Target/garment patch embeddings.
- Time-only vector, category vector, and combined conditioning vector.
- RoPE tables.
- Output after patch mixers 0 and 3.
- Both streams after double blocks 0 and 7.
- Joint output after single blocks 0 and 15.
- Final patch projection and unpatchified velocity.
- Both `v_c` and `v_u`.
- Guided velocity and updated RGB state for selected steps.
- Final unclipped/clipped float images and cropped byte output.

Use the reference model directly for model tests; do not run detector/parser inference every time a transformer block is compared.

A useful direct oracle sketch:

```python
model = TryOnModel().float().eval()
model.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)

# Inputs/noise below are recorded fixtures, not regenerated in the C++ test.
with torch.inference_mode():
    conditional = model(
        noisy, times,
        ca_images=ca,
        garment_images=garment,
        person_poses=person_pose,
        garment_poses=garment_pose,
        garment_categories=categories,
    )["x"]
    unconditional = model(
        noisy, times,
        ca_images=torch.zeros_like(ca),
        garment_images=torch.zeros_like(garment),
        person_poses=torch.zeros_like(person_pose),
        garment_poses=torch.zeros_like(garment_pose),
        garment_categories=torch.zeros_like(categories),
    )["x"]
```

Hooks can export intermediate blocks without changing model math. Pin the attention implementation/precision policy used by the oracle. A diagnostic non-flash PyTorch attention path may be useful, but record it as an oracle setting.

### 14.4 Comparison metrics

Use, at minimum:

```text
max_abs = max(abs(native-reference))
rmse = sqrt(mean((native-reference)^2))
relative_l2 = norm(native-reference) / max(norm(reference), 1e-12)
```

Initial acceptance targets, to establish in the first parity milestone:

- Integer permutation and crop tests: exact equality.
- PNG prepared inputs produced by the same pinned preprocessing path: exact byte equality.
- Normalization: float32 agreement within approximately 1e-7.
- Time schedules: absolute difference at most approximately 2e-7 for standard settings.
- F32 primitive/single-block comparisons: begin with `atol=1e-4`, `rtol=1e-4`, report max/RMSE too.
- Full F32 forward: aim for relative L2 no greater than `1e-3`, plus finiteness and shape checks.
- Output byte conversion: exact for identical floating inputs under the chosen conversion rule.

These are proposed engineering gates, not measured results or guaranteed tolerances. Diagnose the earliest diverging tensor before changing a threshold. Record any tolerance adjustment with evidence about accumulation/backend behavior.

For BF16 and quantization, retain F32 baseline comparisons and add garment-detail, identity, color, and artifact evaluation on a fixed representative set. Do not certify a Q4 model from a single plausible output or choose permissive pixel tolerances that hide a wrong condition path.

### 14.5 Mandatory edge and regression cases

Test:

- All three garment categories.
- Both flat-lay and model-worn garments.
- Segmentation-free and masked person modes.
- Normalized blank pose `-1` versus dropped pose `0`.
- Category zero in the internal unconditional path, but rejection through the public user category.
- CFG=0, CFG=1, CFG=1.5; the reference still uses conditional output in skipped final CFG steps.
- `skip_cfg_last_n_steps` of 0, 1, and all steps.
- One-step schedule endpoints and no division by zero.
- Bad dimensions/channels, missing inputs, empty/truncated/incorrect checkpoints.
- Missing interior block and unexpected extra block.
- Already-prefixed versus raw checkpoint names and repeated normalization.
- Non-finite parameters, invalid crop rectangles, oversized server requests.
- Cancellation between branches and between samples.
- Two different requests on the same context with no stale condition reuse.
- Sample batch sizes 1 and 4 with deterministic noise policy.
- BF16/F32/GGUF round trips and permitted quantization exclusions.

Regress existing families where code is shared: SD 1.5 loading/inference, a FLUX-family graph/load case, Hunyuan/shared-block coverage, and a pixel-space model path when changing `FakeVAE` or dimension helpers. Keep test selection focused on the actual changed surfaces.

## 15. This machine: feasibility and performance expectations

Previously inspected hardware:

```text
Windows 11 x64 VM
AMD EPYC 7763 exposure: 8 cores / 16 logical processors
AVX2/FMA available; AVX-512 unavailable
64 GB system RAM
No exposed compute GPU
```

Weight storage:

- Released BF16 payload: approximately **1.810 GiB**.
- Expanded F32 payload: approximately **3.620 GiB**.
- Neither number includes activations or preprocessing models.

The RAM is sufficient for a serious CPU porting effort with B=1 and controlled workspace. Do not guarantee peak process memory before a graph measurement.

The runtime bottleneck will likely be the long-sequence transformer, not just the roughly 1B weight count. There are 24 joint-attention core blocks over 6,912 tokens, plus four 3,456-token patch mixers, and normally two forward branches per step.

Use the machine for:

- Metadata and input validation.
- Layout/unit tests.
- Reduced-shape **diagnostic** block tests.
- Individual full-shape forward passes.
- One-step full-canvas smoke tests.
- Limited end-to-end reference/native comparisons.

Reduced test shapes are not supported production resolutions. They must not replace the full 576 x 864 acceptance test.

Do not promise interactive generation. The model card's H100 timing and the earlier SD 1.5 256 x 256 result cannot be extrapolated reliably. A full 30-step CPU generation could be very slow; actual timing should be measured and reported only after implementation.

Benchmark F32 versus BF16 versus Q8, 8 versus 16 threads, and flash versus manual attention. On this AVX2-only VM, BF16 storage support does not imply native BF16 arithmetic acceleration.

There is no need to install CUDA or alter system GPU drivers on this VM for the first milestones.

## 16. Licensing and packaging

The repository/model card mark FASHN VTON 1.5 as Apache-2.0 and attribute FLUX-derived components. Preserve applicable notices when porting or adapting code.

DWPose/YOLOX are separately attributed. The reference parser explicitly inherits the NVIDIA SegFormer license; section 3.3 limits use to non-commercial research/evaluation for ordinary users.

This means:

- "FASHN is Apache-2.0" is not enough to describe the complete reference pipeline's licensing.
- Keeping preprocessing optional is useful architecturally, but does not remove license obligations when the restricted parser is actually used.
- Do not bundle or automatically download the parser as if it were an unrestricted runtime component.
- Commercial deployment needs a separate rights review or an appropriately licensed alternative, whose output distribution must then be evaluated.
- Do not imply that swapping a parser preserves numerical parity.

For packaging, keep the native prepared-input runtime usable without Python, ONNX Runtime, OpenCV, or parser weights. Make the optional preprocessing tool/dependencies explicit and separately documented. This is an engineering licensing flag, not a legal opinion about a particular deployment.

## 17. Milestones with exit criteria

| Milestone | Work | Exit criterion |
|---|---|---|
| M0: Reproducible oracle | Pin source/models/dependencies; prepare examples; export metadata/noise/layers | Reference forward and preprocessing artifacts are reproducible |
| M1: Loader | Dedicated family, prefix handling, strict config/key validation | Official/raw/prefixed metadata accepted; malformed variants rejected; no CLIP/VAE requirement |
| M2: Native graph | Patch embeddings, time/category, mixers, reused double/single blocks, unpatchify | Primitive/block/full-forward comparisons pass with shared inputs |
| M3: Sampling | Exact schedule, null-image CFG, final-step policy, RNG/ownership/cancellation | One-step and multi-step float trajectories match defined gates |
| M4: Local workflow | Prepared manifest, additive C API, CLI, output crop/metadata/docs | Real reference-prepared person/garment produces correctly sized try-on output natively |
| M5: Precision/performance | Original BF16, flash, Q8, workspace/cache measurements | Documented quality/memory/latency comparisons; no regression from F32 baseline |
| M6: Server | Capability, route, async ownership/worker/cancel/result | Prepared-input try-on works through server without misusing img2img/OpenAI routes |
| M7: Native preprocessing | ORT/OpenCV tool, parser export if licensed/needed, transform/mask port | Raw-image-to-prepared-input parity and end-to-end local execution without Python |
| M8: Additional backends/features | GPU backends, fused CFG, additional quantization, optional integrations | Each feature has its own parity/quality/performance evidence |

Dependencies: M1-M3 depend on M0; M4 depends on M3; M5 and M6 depend on M4; M7 can use the prepared-input contract independently after M0 but is not a substitute for M2-M3.

For reviewable changes, keep loader/API, model graph, sampler, CLI, server, and native preprocessing in separate focused changesets. Avoid a simultaneous general model-runtime refactor.

No implementation time estimate is asserted here: the graph has strong reuse, but reference parity and native preprocessing are separate uncertainty sources. Estimate scheduling after M0/M2 establish the numerical and backend issues.

## 18. Principal failure modes and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Generic FLUX detection | Missing text/projection weights; wrong model initialized | Family recognition before FLUX, then strict preset validation |
| Wrong time/sign | Noise increases or image never forms | Shared-noise one-step fixture and explicit ascending-time math |
| Wrong null branch | CFG distorts output despite plausible conditional forward | Exact-zero conditions and category row zero; compare both branches |
| Wrong pose representation | Clothing/person alignment fails | Use exact grayscale raster pipeline, not RGB OpenPose |
| Wrong patch order | Scrambled colors/tiles with successful execution | Coordinate-coded round-trip tests and patch-convolution comparison |
| Treating garment as text | Spatial conditioning lost | Two spatial RoPE grids and original garment tokens |
| Missing patch mixer | "Almost FLUX" graph produces poor results | Validate/run all four `x_patch_mixer` blocks |
| F16-only patch wrapper | F32 oracle never reaches tight parity | FASHN-local patchify-plus-linear representation |
| Global helper change | Existing FLUX/SD models regress | Keep family-specific numerics/conversion local |
| Quadratic attention memory | Large allocation or poor CPU behavior | B=1 sequential CFG, inspect flash support and real workspace |
| Stale condition cache | Second request resembles first | Request-scoped state and explicit cache invalidation |
| Resize/crop shortcuts | Misalignment or wrong output dimensions | Lossless prepared fixtures; distinguish canvas from final crop |
| Same-seed assumption | Outputs differ before first forward | Export and share initial noise |
| Generic conversion assumptions | Protected tensors keep unexpected dtype | Inspect converted headers and use explicit F32 reference exporter |
| Parser licensing | Whole-pipeline license described incorrectly | Separate component rights and optional packaging |

## 19. Definition of done

For the first native FASHN release:

- The official checkpoint is recognized by architecture, not its filename.
- No text encoder or learned VAE is required.
- All 365 inference tensors are correctly consumed; only the documented unused buffer is ignored.
- Tensor layouts, both CFG branches, RoPE, the patch mixer, and output unpatchification are verified.
- Time runs from zero to one with the correct flow update and last-step CFG policy.
- A real full-canvas try-on is generated from reference-prepared inputs.
- Category, cropping, ownership, errors, and cancellation behave as documented.
- The CLI and C API clearly distinguish prepared-input support from raw-image preprocessing.
- No unsupported GPU throughput, precision, sampler, quantization, UI, or licensing claims are made.
- The existing SD 1.5 workflow remains usable.

For a later "fully native raw-image try-on" claim, add native preprocessing parity and its dependency/license requirements to the above.

## 20. Source index

The following permalinks are the principal code references used in this plan:

- [FASHN architecture and forward pass][F-model]: block definitions, patch embeddings, CFG dropout, category/time embedding, stream order, unpacking.
- [FASHN pipeline][F-pipeline]: preprocessing order, precision, sampler, CFG skip, final crop.
- [FASHN schedule][F-sampling]: time shift and reverse schedule.
- [FASHN transforms][F-transforms]: PIL/OpenCV resize, padding, unpadding.
- [FASHN image/tensor conversion][F-tensor]: normalization and patch unpacking.
- [FASHN agnostic/garment processing][F-agnostic] and [mask algorithms][F-masks].
- [DWPose wrapper][F-dwpose], [YOLOX processing][F-dwpose-det], [pose processing][F-dwpose-pose], and [drawing][F-drawing].
- [FASHN model card][F-card] and [checkpoint artifact][F-weights].
- [Parser implementation][P-parser], [labels][P-labels], and [configuration][P-config].
- [Parser license statement][P-readme] and [NVIDIA license][P-license].
- [Current FLUX blocks/runner][S-flux] and [Hunyuan reuse precedent][S-hunyuan].
- [Patchify/unpatchify][S-dit], [RoPE][S-rope], and [generic model blocks][S-blocks].
- [GGML attention helper][S-attention] and [runner interfaces][S-runner].
- [Version classification][S-model], [model loader][S-loader], and [name conversion][S-names].
- [Context/generation runtime][S-runtime], [denoisers/samplers][S-denoisers], and [guidance interfaces][S-guidance].
- [Model-manager validation][S-manager] and [conversion policy/export][S-convert].
- [Public C API][S-api], [example-common types][S-common], and [CLI dispatch][S-cli].
- [Server routes][S-server], [async jobs][S-jobs], and [CMake configuration][S-cmake].
- [Current output byte conversion][S-preprocessing] and [CPU RNG][S-rng].
- [Model config conventions][S-config] and [contribution conventions][S-contributing].

[F-model]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/tryon_mmdit.py
[F-pipeline]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/pipeline.py#L140-L339
[F-sampling]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/utils/sampling.py
[F-transforms]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/preprocessing/transforms.py
[F-tensor]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/utils/tensor.py
[F-agnostic]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/preprocessing/agnostic.py
[F-masks]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/preprocessing/masks.py
[F-dwpose]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/dwpose/dwpose.py
[F-dwpose-det]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/dwpose/onnxdet.py
[F-dwpose-pose]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/dwpose/onnxpose.py
[F-drawing]: https://github.com/fashn-AI/fashn-vton-1.5/blob/7c0f10af3f91ad4048fe9729c470a13ef905d25a/src/fashn_vton/dwpose/utils.py
[F-card]: https://huggingface.co/fashn-ai/fashn-vton-1.5/blob/7720683168567eb5a2a4c67f15116c6e29c83ded/README.md
[F-weights]: https://huggingface.co/fashn-ai/fashn-vton-1.5/blob/7720683168567eb5a2a4c67f15116c6e29c83ded/model.safetensors
[P-parser]: https://github.com/fashn-AI/fashn-human-parser/blob/f2771f2fb8655349e87e2869bbde7ace0bd06f2c/src/fashn_human_parser/parser.py
[P-labels]: https://github.com/fashn-AI/fashn-human-parser/blob/f2771f2fb8655349e87e2869bbde7ace0bd06f2c/src/fashn_human_parser/labels.py
[P-config]: https://huggingface.co/fashn-ai/fashn-human-parser/blob/1f80c34dbab321c5730dda5c3fea279fd3e97498/config.json
[P-readme]: https://github.com/fashn-AI/fashn-human-parser/blob/f2771f2fb8655349e87e2869bbde7ace0bd06f2c/README.md#license
[P-license]: https://github.com/NVlabs/SegFormer/blob/master/LICENSE
[S-flux]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model/diffusion/flux.hpp#L201-L764
[S-hunyuan]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model/diffusion/hunyuan.hpp#L308-L345
[S-dit]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model/diffusion/dit.hpp#L9-L78
[S-rope]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model/common/rope.hpp
[S-blocks]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model/common/ggml_block.hpp#L141-L451
[S-attention]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/core/ggml_extend.cpp#L529-L669
[S-runner]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/core/ggml_runner.h
[S-model]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model.h
[S-loader]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model_loader.cpp#L387-L620
[S-names]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/name_conversion.cpp
[S-runtime]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/stable-diffusion.cpp
[S-denoisers]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/runtime/denoiser.hpp#L1251-L1488
[S-guidance]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/runtime/guidance.h
[S-manager]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/model_manager.cpp#L677-L711
[S-convert]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/convert.cpp
[S-api]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/include/stable-diffusion.h
[S-common]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/examples/common/common.h
[S-cli]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/examples/cli/main.cpp
[S-server]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/examples/server/routes_sdcpp.cpp#L350-L483
[S-jobs]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/examples/server/async_jobs.h
[S-cmake]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/CMakeLists.txt
[S-preprocessing]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/runtime/preprocessing.hpp#L28-L37
[S-rng]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/src/core/rng_mt19937.hpp
[S-config]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/docs/model_config.md
[S-contributing]: https://github.com/leejet/stable-diffusion.cpp/blob/d04e8950c1ec8d30248cbe996682b3182fb1adf6/CONTRIBUTING.md
