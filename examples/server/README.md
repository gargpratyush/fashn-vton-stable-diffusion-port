# Example

The following example starts `sd-server` with a standalone diffusion model, VAE, and LLM text encoder:

```
.\bin\Release\sd-server.exe --diffusion-model  ..\models\diffusion_models\z_image_turbo_bf16.safetensors --vae ..\models\vae\ae.sft  --llm ..\models\text_encoders\qwen_3_4b.safetensors --diffusion-fa --offload-to-cpu -v --cfg-scale 1.0
```

What this example does:

* `--diffusion-model` selects the standalone diffusion model
* `--vae` selects the VAE decoder
* `--llm` selects the text encoder / language model used by this pipeline
* `--diffusion-fa` enables flash attention in the diffusion model
* `--offload-to-cpu` reduces VRAM pressure by keeping weights in RAM when possible
* `-v` enables verbose logging (equivalent to `--log-level verbose`)
* `--cfg-scale 1.0` sets the default CFG scale for generation

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

After the server starts successfully:

* the web UI is available at `http://127.0.0.1:1234/`
* the native async API is available under `/sdcpp/v1/...`
* the compatibility APIs are available under `/v1/...` and `/sdapi/v1/...`

If you want to use a different host or port, pass:

```bash
--listen-ip <ip> --listen-port <port>
```

# FASHN Try-on

From the repository root, start experimental FASHN try-on without text encoders or a VAE:

```powershell
.\build\bin\Release\sd-server.exe --diffusion-model model.safetensors --rng cpu --type bf16 --diffusion-fa -t 8 --listen-ip 127.0.0.1
```

Use `POST /sdcpp/v1/try_on` with embedded prepared PNGs or configured `raw_inputs`. Sampling controls belong
in request JSON, not generic generation CLI defaults. See
[the API contract](api.md#fashn-prepared-input-try-on) and
[preparation instructions](../../docs/fashn_vton.md#native-raw-image-preparation).
Open `/try-on` on the server for a standalone prepared-input upload UI with
progress, cancellation and downloads. This page is embedded without a pnpm
build and does not modify the frontend submodule. With `SD_FASHN_PREPROCESS=ON`,
add `--try-on-dwpose-dir weights\dwpose` to enable in-memory raw HTTP/UI uploads.
Worn-garment and person-masking modes additionally require
`--try-on-parser-dir weights\human-parser-onnx-eval --accept-parser-research-license`
and remain restricted to authorized non-commercial research/evaluation.
The optional `sd-fashn-prepare` tool uses the same native pipeline.
See the [raw input contract](api.md#fashn-raw-image-try-on-opt-in).
`--type bf16` reduces
resident matrix memory while computation remains F32. Active cancellation is
cooperative and can wait for a full model forward. Keep the example server on
a trusted local interface; this does not add authentication.

# Frontend

## Build with Frontend

The server can optionally build the web frontend and embed it into the binary as `gen_index_html.h`.

### Requirements

Install the following tools:

* **Node.js** ≥ 20
  https://nodejs.org/

* **pnpm** ≥ 10
  Install via npm:

```bash
npm install -g pnpm
```

Verify installation:

```bash
node -v
pnpm -v
```

### Install frontend dependencies

Go to the frontend directory and install dependencies:

```bash
cd examples/server/frontend
pnpm install
```

### Build the server with CMake

Enable the frontend build option when configuring CMake:

```bash
cmake -B build -DSD_SERVER_BUILD_FRONTEND=ON
cmake --build build --config Release
```

If `pnpm` is available, the build system will automatically run:

```
pnpm run build
pnpm run build:header
```

and embed the generated frontend into the server binary.

## Frontend Repository

The web frontend is maintained in a **separate repository**, https://github.com/leejet/sdcpp-webui.

If you want to modify the UI or frontend logic, please submit pull requests to the **frontend repository**.

This repository (`stable-diffusion.cpp`) only vendors the frontend periodically. Changes from the frontend repo are synchronized:

* approximately **every 1–2 weeks**, or
* when there are **major frontend updates**

Because of this, frontend changes will **not appear here immediately** after being merged upstream.

## Using an external frontend

By default, the server uses the **embedded frontend** generated during the build (`gen_index_html.h`).

You can also serve a custom frontend file instead of the embedded one by using:

```bash
--serve-html-path <path-to-index.html>
```

For example:

```bash
sd-server --serve-html-path ./index.html
```

In this case, the server will load and serve the specified `index.html` file instead of the embedded frontend. This is useful when:

* developing or testing frontend changes
* using a custom UI
* avoiding rebuilding the binary after frontend modifications

# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-server -h
```
