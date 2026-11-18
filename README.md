# Kumo Material Intelligence

**Version 0.5 — working on-device CMF vision research prototype.**

[Main workstation](https://exmachina100.github.io/kumo-material-intelligence/apps/demo/) · [Material review studio](https://exmachina100.github.io/kumo-material-intelligence/apps/review/) · [Research model lab](https://exmachina100.github.io/kumo-material-intelligence/apps/research/) · [Model release criteria](docs/model-release.md)

## What works now

- Upload or capture a surface image and inspect dominant color, CIELAB, palette, and local capture-quality hints.
- Click **Identify material (AI research)** to run the actual pinned MobileNetV3-Small vision model locally using ONNX Runtime Web and WebAssembly.
- Three image crops contribute to a cautious research suggestion. If the votes disagree or the raw score is too low, the displayed identification is **unknown**, not a forced label.
- Compare the top material-family hypotheses: **ceramic, coating, leather, metal, stone, wood**. Only these six classes were included in this research training. Other families require more representative data.
- A designer must explicitly click **Confirm suggested family** (or classify manually) before the CMF Material DNA label becomes designer-authored.
- Save reviewed Material DNA records in the local library and export/import JSON. Uploaded photos are not transmitted to a server.
- Review 110 Poly Haven CC0 reference images in a separate review interface with source provenance. Import/export expert review decisions for dataset building.
- Model weights, labels and manifest are committed to GitHub; deployment validates the pinned SHA-256, source labels and research-only flags. The ONNX JS/WASM runtime is served from the same Kumo site.

## Actual model, evaluation and limits

The current model is a frozen ImageNet-pretrained MobileNetV3-Small vision backbone with a trained linear category head. It was trained on **66 catalog-labeled Poly Haven previews**, with **22 source-distinct catalog references** held out for a preliminary evaluation.

- **77.3%** agreement with weak held-out catalog labels.
- **75.7%** macro-F1 against the same weak catalog labels.
- ONNX-vs-PyTorch logit parity checked on synthetic input.
- The first Chromium end-to-end browser run successfully loaded the model and returned a research prediction for a leather reference.
- **Not validated for real-world automotive CMF recognition**, independent supplier materials, physical composition, or phone-camera imagery. Raw softmax values are uncalibrated and may be overconfident on unknown families.
- Browser and device preprocessing parity, actual unknown rejection, lighting generalization and CMF expert-verified data remain necessary for production deployment.

**The model is experimental and must never be presented as physically verified material identification.** Production integrations with Allure remain disabled until independent validation.

## Try material identification

1. Open the [CMF workstation](https://exmachina100.github.io/kumo-material-intelligence/apps/demo/).
2. Upload an image or use the mobile camera capture control.
3. Click **Identify material (AI research)**. The model is downloaded once on demand from the Kumo site and its checksum verified in the browser.
4. Review the research guess, crop agreement and alternative material hypotheses.
5. If appropriate, click **Confirm suggested family** to set a designer-authored, *unverified* material family. Otherwise use the manual family selector.

The [Research Model Lab](https://exmachina100.github.io/kumo-material-intelligence/apps/research/) also allows testing any manually selected compatible ONNX model and its labels.

## Development

Node 20+ and Python 3:

```bash
npm test
npm run serve
```

Open `http://localhost:8080/apps/demo/`. The local repo contains research model weights, but the same-origin WASM runtime is staged by the GitHub Pages workflow; for full local inference, copy the `onnxruntime-web@1.20.1/dist` runtime files to `vendor/ort`.

Dataset source code and documentation are under `training/`. The `tests/browser-vision.e2e.mjs` script is exercised by a dedicated GitHub Actions browser workflow. Training outputs are research artifacts, never automatically approved for production.

## Roadmap

1. Obtain expert-confirmed real material samples and evaluate out-of-domain unknown detection and capture lighting variation.
2. Extend labels to automotive-focused families (textile, synthetics, polymers, rubber, glass, composites), with adequate independent examples per class.
3. Calibrate model scores and measure inference latency / memory on real iOS and Android devices.
4. Validate preprocessing parity, optimize quantization, and develop a robust production inference UX.
5. Connect verified Material DNA intelligence into Allure and other Kumo tools.

## Security, license and provenance

The repo is public. Avoid adding supplier-confidential or internal OEM photographs. Model weights are used under their applicable upstream licenses; the CC0 reference sources have independent documented provenance. Before enterprise storage, add authenticated encrypted persistence, retention and access policies. This repository's own code has no public redistribution license selected yet.
