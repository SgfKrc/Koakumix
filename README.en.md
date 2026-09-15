<p align="center">
  <img src="assets/Koakumix.png" alt="Koakumix" width="220">
</p>

# Koakumix

> **Language**: English · [简体中文](README.md)

Koakumix is the **local small-model augmentation harness** of the QLH project (formerly `harness_workbench/`; split into its own repository as a git submodule on 2026-09-15).

**Value proposition**: the *same* model should perform noticeably better inside Koakumix than in a plain chat. Long-context management, cross-session memory, retrieval augmentation (RAG) and small-model composition (draft-verify) are combined to lift the usability of local small models by one notch. Koakumix does not try to be a general-purpose agent harness — it does exactly one thing: augment local small models.

## Modules

| Module | Responsibility |
| --- | --- |
| `api_layer/` | OpenAI-compatible HTTP orchestration (chat + RAG + memory + compression engine) |
| `context_engine/` | Context budgets and policies (long-context management) |
| `memory/` | Cross-session long-term memory (extract / store / retrieve) |
| `rag/` | Chunking strategy, index granularity and offline baselines |
| `eval/` `research/` | Evaluation fixtures, red-team contracts, role-asymmetry reports |
| `image/` | **Image generation** (local engine + asset manifest/contracts) and multimodal follow-up context |
| `mcp_server/` | MCP tool surface (including `image_generate`) |
| `model_profiles/` | Model profiles, capability gates and per-device model selection |
| `skills/` | Packaged capability units (SKILL.md + JSON Schema + stable entry point) |
| `cli.py` `tui.py` | Command-line and TUI entry points |
| `desktop.py` | **Desktop shell** (pywebview: serves the local API and points the native WebView at it; opt-in `desktop` extra, window icon taken from `assets/Koakumix.png`) |
| `ui_react/` | Optional web UI (React + Vite; `node_modules` is rebuilt locally and not committed) |

**Image generation lives here and only here.** The QLH main repository removed the whole SD 1.5 chain in `d18cee7` and keeps only optional multimodal *understanding*.

## Quick start

```bash
pip install -e ".[tui]"          # editable install (with the Textual TUI)
koakumix --help                  # CLI
python -m harness_workbench.tui  # TUI
```

MCP tool surface (stdio):

```bash
python -m harness_workbench.mcp_server
```

Web UI:

```bash
cd ui_react
npm install                      # rebuild node_modules locally (not committed)
npm run build                    # tsc --noEmit + vite build
npm run dev                      # dev server
```

Desktop shell (pywebview):

```bash
pip install -e ".[desktop]"                    # optional deps: pywebview + fastapi + uvicorn
koakumix-desktop --check                       # readiness report: no window, no model needed
koakumix-desktop --model path/to/model.gguf    # serve locally and open the desktop window
```

### Enabling real image generation

The local image executor is **opt-in** and fails closed. With the optional runtime
(`diffusers` + `torch`) and a verified SD asset present:

```bash
QLH_HARNESS_IMAGE_EXECUTOR=diffusers
QLH_HARNESS_IMAGE_ASSET_ROOT=models/sd15-original-v1
QLH_HARNESS_IMAGE_DEVICE=cuda          # or cpu / auto (default)
```

Without these, the API and the MCP tool report `images_unavailable` / `executor_not_enabled`
instead of pretending to be able to generate.

## Tests

```bash
python -m pytest tests -q
```

## Relationship to the QLH main repository

Koakumix is mounted as a **git submodule** at `harness_workbench/` in the QLH main repository:

- **Zero coupling at the code level**: the main repository does not `import` any module of this
  repository (only the submodule pointer plus one image-surface guard check). Inside Koakumix the
  only exception is a *lazy* import in `tools/rag_baseline.py` used for the two-sided baseline
  comparison; a regression test enforces this with an AST check.
- **Integration**: the main repository consumes Koakumix through an **OpenAI-compatible API**,
  model manifests and explicit artifact references.
- **Shared assets** (prompt sets, rubrics, RAG baseline fixtures) are synchronized as **contracts**,
  never via implicit imports.

## History

- 2026-09-14: harness established inside the QLH main repository (`harness_workbench/`), positioned as a small-model augmentation harness.
- 2026-09-15: split out with `git subtree split` (**864 commits of history preserved**); the main repository switched to a submodule. 39 tests, 4 facade scripts (`qlh_say` / `model_quotes` / `check_prompt_cache` / `role_asymmetry_report`) and a copy of `fixtures/benchmark` came along.
- After the QLH image-generation cull (`d18cee7`), **image generation is concentrated in this repository**.

## License

MIT License — see [LICENSE](LICENSE).
