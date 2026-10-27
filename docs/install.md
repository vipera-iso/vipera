---
title: Install
description: Step-by-step installation, model download, troubleshooting, and uninstall.
---

# Install Vipera

This guide is for getting Vipera running on a normal laptop or desktop. You do
not need to know Python, and you do not need a GPU. You will use a terminal for
a few copy-paste commands.

> **Install `vipera-ai`, not `vipera`.**
> The project published on PyPI as `vipera` is an unrelated PDF tool. Vipera's
> distribution is `vipera-ai`; every command below uses that name.

## What you need

| Requirement | Details |
|---|---|
| Python | **3.11 or 3.12**. Use 3.12 if you are installing it fresh. Avoid 3.13/3.14: prebuilt `llama-cpp-python` packages may not exist yet. |
| Disk space | At least 10 GB free. Models are several GB each. |
| Memory | 8 GB RAM is enough for a small model; 16 GB is comfortable. |
| Internet | Needed to install and download a model. After that, inference runs offline. |
| GPU | Optional. CPU works, but generation is slower. |

You do **not** need Docker, an NVIDIA GPU, or administrator rights to install
Vipera itself. Installing Python may require an administrator password.

## Step 1 — Install Python 3.12

**Windows:** download the installer from
[python.org/downloads](https://www.python.org/downloads/). On the first screen,
tick **“Add python.exe to PATH”**, then install.

**macOS:** download the macOS installer from
[python.org/downloads](https://www.python.org/downloads/) and run it. If you use
Homebrew, `brew install python@3.12` works too.

**Linux:** use your distribution's package manager. For Ubuntu 24.04:

```bash
sudo apt install python3.12 python3.12-venv
```

Check that it worked. Open a terminal (PowerShell on Windows) and run:

```bash
python3.12 --version
```

You should see `Python 3.12.x`. On Windows, `py -3.12 --version` is equivalent.

## Step 2 — Create a private environment

This keeps Vipera and its dependencies out of the rest of your system.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv "$env:USERPROFILE\vipera-env"
& "$env:USERPROFILE\vipera-env\Scripts\Activate.ps1"
```

If PowerShell refuses to run the activation script, allow it for this window
only, then run the activation command again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**macOS / Linux:**

```bash
python3.12 -m venv ~/vipera-env
source ~/vipera-env/bin/activate
```

Your prompt now starts with `(vipera-env)`. That means the environment is
active. You must activate it again every time you open a new terminal.

## Step 3 — Install Vipera

Choose one option.

### Option A — install from PyPI (no Git)

```bash
python -m pip install "vipera-ai[llama,gguf,nvidia]"
```

The command is the same in PowerShell and in bash/zsh, once the virtual
environment from Step 2 is active. `python -m pip` picks the interpreter of that
environment, so no path juggling is needed.

If PyPI is unreachable, download
<https://github.com/vipera-iso/vipera/archive/refs/heads/main.zip>, extract it,
and run `python -m pip install ".[llama,gguf,nvidia]"` inside the extracted
`vipera-main` folder.

### Option B — one command with Git

If Git is installed:

```bash
python -m pip install "vipera-ai[llama,gguf,nvidia] @ git+https://github.com/vipera-iso/vipera.git"
```

This works in PowerShell and in bash/zsh. It downloads the project and installs
it in one step.

### What the extras mean

| Extra | Why it is there |
|---|---|
| `llama` | The inference runtime (`llama-cpp-python`). Without it Vipera cannot load a model. |
| `gguf` | Reads layer count and parameter metadata from model files. |
| `nvidia` | Installs `nvidia-ml-py` (the `pynvml` module) so Vipera reads NVIDIA VRAM without spawning anything. Recommended for NVIDIA cards; AMD and Intel need no Python extra. |

Installation can take several minutes because `llama-cpp-python` may compile
code on your machine.

### GPU backends

Vipera plans for any card llama.cpp can use. What you install is the only part
that differs:

| Card | Install the runtime with | Vipera reads free VRAM from |
|---|---|---|
| NVIDIA | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124` (match your CUDA version, or set `CMAKE_ARGS='-DGGML_CUDA=on'`) | NVML (the `nvidia` extra), then `llama-server --list-devices` |
| AMD, Linux | `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/rocm72`, or `CMAKE_ARGS='-DGGML_HIP=on'` | amdgpu sysfs counters, `rocm-smi`, then `llama-server` |
| AMD, Windows | `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/hip-radeon` | `rocm-smi`, then `llama-server` |
| Intel Arc, Flex, Max | `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan`, or a SYCL build: `source /opt/intel/oneapi/setvars.sh` then `CMAKE_ARGS='-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx'` | `xpu-smi`, then `llama-server` |
| CPU only | nothing extra | RAM budget |

`vipera system doctor` prints the GPU it found, the `vram source` the number
came from, and whether the installed runtime can actually see the card. A
source that ends in an estimate (for example `xpu-smi` without free-memory
reporting) means the planner is budgeting conservatively from capacity.

An AMD APU (a Ryzen desktop or mobile chip with integrated graphics) reports
shared system memory rather than dedicated VRAM; the planner treats it as
unified memory, and `system doctor` labels it that way.

## Step 4 — Check the install

```bash
vipera --version
vipera system doctor
```

Expected:

- `vipera --version` prints `vipera 1.1.0`.
- `llama-cpp-python` shows `PASS`.
- `gguf` shows `PASS`.
- The GPU line shows `WARN` on a CPU-only machine. That is normal.
- If the runtime is missing, `doctor` now prints the exact install command.

## Step 5 — Get a model

```bash
vipera model search
```

This is interactive:

1. Type a keyword, for example `llama 3.2` or `qwen`.
2. Pick a repository by number.
3. Pick a file by number. Prefer a **`Q4_K_M`** file: a good balance of quality
   and size.

Rough guidance for a laptop:

| RAM | Suggested model |
|---|---|
| 8 GB | 3–4B parameters, Q4_K_M (~2–3 GB file) |
| 16 GB | 7–8B parameters, Q4_K_M (~4–5 GB file) |
| 32 GB | 13–14B parameters, Q4_K_M (~8–9 GB file) |

The download can take a long time. When it finishes, the model is registered
and set as the default.

Already have a `.gguf` file? Import it instead:

```powershell
# Windows
vipera model import "$env:USERPROFILE\Downloads\my-model.gguf" --name my-model
```

```bash
# macOS / Linux
vipera model import ~/Downloads/my-model.gguf --name my-model
```

## Step 6 — Run it

```bash
vipera
```

The first run loads the only installed model automatically; with several
models, you get a picker. Leave this terminal open. When you see
`SYSTEM READY`, the API is at:

```text
http://127.0.0.1:8686/v1
```

Point any OpenAI-compatible client at `http://127.0.0.1:8686/v1` with any API
key value. To chat in the terminal instead of serving HTTP, stop the gateway
with `Ctrl+C` and run `vipera chat repl`. Both modes load the model into memory,
so avoid running them at the same time on a small machine.

## Next time you want to use it

Open a terminal, activate the environment, and run `vipera`:

**Windows:**

```powershell
& "$env:USERPROFILE\vipera-env\Scripts\Activate.ps1"
vipera
```

**macOS / Linux:**

```bash
source ~/vipera-env/bin/activate
vipera
```

## Troubleshooting

| Problem | What to do |
|---|---|
| `vipera` is not recognized | The environment is not active. Run the activation command for your OS, then retry. |
| `llama-cpp-python is required to load models` | Run `python -m pip install llama-cpp-python` inside the activated environment. |
| Install fails while building `llama-cpp-python` | Check `python --version` is 3.11 or 3.12. On macOS run `xcode-select --install`. On Windows install **Visual Studio Build Tools** with the “Desktop development with C++” workload. On Debian/Ubuntu: `sudo apt install build-essential cmake`. Then retry the install. |
| `No models installed` | Run `vipera model search`, or `vipera model import <path>`. |
| `Model not found: <name>` | Run `vipera model list` and use the exact alias. |
| Port 8686 is already in use | `vipera system config port 9000`, then start again. |
| Generation is very slow | Expected on CPU. Try a smaller model or a smaller quantization (`Q4_K_M`, `Q4_0`). |
| Download from Hugging Face fails | Check your network/proxy, or download a `.gguf` manually and use `vipera model import`. |
| `vipera` works, but Python cannot import `vipera` | You are outside the virtual environment. Activate it first. |

## Where things are stored

Everything lives in `~/.vipera` (Windows: `C:\Users\<you>\.vipera`):

```text
~/.vipera/
├── config.json
├── registry.json
├── models/          # downloaded .gguf files
└── logs/
```

To move the store, set `VIPERA_HOME` before starting Vipera. See
[configuration.md](configuration.md).

## Security

By default Vipera listens only on `127.0.0.1`, so other devices cannot reach it.
It has no authentication. Do not change `--host` to `0.0.0.0` on a shared or
public network without putting a reverse proxy in front of it.

## Uninstall

Deactivate the environment, then delete it and the data directory. This also
deletes downloaded models.

**Windows:**

```powershell
deactivate
Remove-Item -Recurse -Force "$env:USERPROFILE\vipera-env"
Remove-Item -Recurse -Force "$env:USERPROFILE\.vipera"
```

**macOS / Linux:**

```bash
deactivate
rm -rf ~/vipera-env ~/.vipera
```

For the full CLI surface, see [cli.md](cli.md). For the HTTP API, see
[http-api.md](http-api.md).
