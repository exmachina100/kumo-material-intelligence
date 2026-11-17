# Kumo Material Intelligence

Local-first CMF intelligence for automotive, product, and industrial design. Initial MVP includes a Material DNA schema, deterministic color analysis, browser demo, and ML training groundwork. **There is no trained material recognition model yet.**

## Run
Requires Node.js 20+ and Python 3.

```bash
npm test
npm run serve
```

Visit http://localhost:8080/apps/demo/ . Image processing occurs entirely in your browser.

## Principles
- Shared Material DNA across web, mobile, and desktop.
- Local-first operation with optional, consent-based cloud fallback.
- No unverified physical or manufacturing claims from images.
- Licensed datasets and leakage-aware evaluation.
- ONNX adapters only after validated model artifacts exist.

## Roadmap
1. Licensed dataset manifest and expert-reviewed evaluation set.
2. Compact material classifier and appearance attributes.
3. Quantized ONNX export and WebGPU/WASM inference.
4. Multi-color palette extraction and perceptual matching.
5. Integrate into Allure and test PWA/mobile capture.

## Structure
- `packages/material-dna` — shared schema
- `packages/cmf-core` — color algorithms
- `packages/inference-sdk` — analysis contract
- `apps/demo` — local-first browser demo
- `training` — experiment roadmap
- `models` — model manifest
- `tests` — automated tests
