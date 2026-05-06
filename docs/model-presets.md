# Model Presets

The install wizard recommends a model based on the user's machine. The recommendation is a starting point, not a hard rule. Later releases should run a local benchmark to measure tokens per second and memory pressure.

## Modes

| Mode | Default model | Alternatives | Use case |
|---|---|---|---|
| Light Mode | `qwen3:4b` | `phi4-mini`, `qwen2.5:3b` | Basic chat, summaries, light tools |
| Fast Mode | `qwen3:8b` | - | Best default for most users |
| Balanced Mode | `qwen3:14b` | `qwen3:8b` | Better summaries, planning, reasoning |
| Power Mode | `qwen3:30b` | `qwen3:32b` | Heavy agent tasks on larger machines |

## Mac Mini Recommendations

| Mac mini | Default model | Better mode | Note |
|---|---|---|---|
| 8GB RAM | `qwen2.5:3b` / `qwen3:4b` / `phi4-mini` | Not recommended for 8B | Basic chat, summaries, and light tools. There is not much room for 8B. |
| 16GB RAM | `qwen3:8b` | `qwen3:14b` if slower responses are acceptable | Best default for most Mac mini users. |
| 24GB RAM | `qwen3:14b` | `qwen3:8b` for Fast Mode | Good balance between quality and speed. |
| 32GB RAM | `qwen3:14b` | `qwen3:30b` / MoE option if it performs well | Better summaries, reasoning, and planning. |
| 64GB+ RAM | `qwen3:30b` | Larger Qwen / DeepSeek Coder / Mixtral-style models | Power users and heavier agent tasks. |

## Language

The product interface, installer, terminal output, approval dialogs, and docs are English by default.

The assistant response language defaults to `auto`. It can answer in any language supported by the selected model.

## CLI

List modes:

```bash
dmdagent models list
```

Switch modes:

```bash
dmdagent model light --pull
dmdagent model fast --pull
dmdagent model balanced --pull
dmdagent model power --pull
```

The longer equivalent commands are:

```bash
dmdagent models set-mode light
dmdagent models set-mode fast
dmdagent models set-mode balanced
dmdagent models set-mode power
```

Changing mode updates both the default assistant model and the planner model. This keeps tool routing fast on smaller machines.

Set a custom model:

```bash
dmdagent model use phi4-mini --pull
```

## API-Key And Compatible Providers

Local Ollama remains the default:

```bash
dmdagent models provider ollama
```

OpenAI-compatible providers are opt-in. The config stores only the environment
variable name, not the API key value:

```bash
export DMDAGENT_OPENAI_API_KEY="sk-..."
dmdagent models provider openai --api-key-env DMDAGENT_OPENAI_API_KEY --model gpt-4o-mini
```

DeepSeek uses its OpenAI-compatible API endpoint:

```bash
export DMDAGENT_DEEPSEEK_API_KEY="sk-..."
dmdagent models provider deepseek --api-key-env DMDAGENT_DEEPSEEK_API_KEY --model deepseek-v4-flash
```

For local compatible servers:

```bash
dmdagent models provider lmstudio --base-url http://localhost:1234/v1
dmdagent models provider vllm --base-url http://localhost:8000/v1
```

Run `dmdagent doctor` after changing providers.
