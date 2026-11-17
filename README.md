# Kumo Material Intelligence

**Version 0.4 — local-first CMF prototype + research model lab** for automotive, product, and industrial design.

[Open browser workstation source](apps/demo/index.html) · [Material review studio](apps/review/index.html) · [ONNX research lab](apps/research/index.html) · [Architecture](docs/architecture.md) · [Mobile deployment](docs/mobile-and-deployment.md)

## Implemented
- Responsive browser workstation: drag/drop image, palette swatches, image preview, manual material family tagging, browser material library, and JSON export.
- Deterministic image color extraction and multi-color palette estimation with perceptual color difference (CIEDE2000).
- Material DNA 0.1.0 schema, JS validator and provenance fields; image samples are not uploaded to servers.
- Optional inference router and injectable ONNX adapter; **no classifier weights are included**.
- Service worker/PWA manifest, Node tests and GitHub Actions CI configuration.
- 468 procedural SVG training checks, 156 synthetic PNG examples and 110 Poly Haven CC0 reference previews with provenance and separate human-review gating.\n- Reproducible Python research pipelines, including a pretrained MobileNetV3-Small frozen vision encoder + linear classifier, with research-only ONNX export.\n- Separate opt-in browser ONNX research lab, which requires manually selecting the model and labels extracted from a GitHub Actions artifact. No automatic production classification is enabled.

## Run locally
Node 20+ and Python 3 required.

```bash
npm test
npm run serve
```

Open **http://localhost:8080/apps/demo/**.

The web app operates without an API key; after a first successful load, static resources may be cached for offline use.

## Train a baseline (requires your own licensed images)
```bash
python -m pip install -r training/requirements.txt
python training/train_classifier.py --data training/data --out training/out --epochs 8 --pretrained
```

Folder format: `training/data/{train,val,test}/{family}/*.jpg`. Pretrained mode downloads TorchVision weights. Split by sample/source, not random photos. The script exports a baseline ONNX model and performance report; it does not automatically deploy weights to the browser. **Do not expect production model quality from an uncurated dataset.**

## Status / limits
- Pretrained experimental category classification has been trained (77.3% accuracy / 75.7% macro F1 **against 22 weak catalog labels**); this is **not** a validated CMF recognition model. Production classification remains disabled.
- Full PBR map generation, supplier specifications, advanced CMF reasoning and native apps: **not yet built**.
- Browser PWA and GitHub Pages deployment work; native mobile packaging, full input-tensor parity, browser inference benchmarking, and CMF expert validation remain open.
- Material finish, substrate, cost, environmental impact and compliance must not be inferred as verified facts from a photo.

## Next milestones
1. Add licensed material data and sample-grouped held-out benchmarks.
2. Train and compare encoder baselines, calibrate confidence and unknown detection.
3. Validate ONNX preprocessing parity and quantize after accuracy checks.
4. Wire the approved ONNX artifact into the browser classifier at runtime.
5. Integrate Material DNA and library into Allure, then package mobile apps.

## Privacy
All MVP analysis occurs on-device. Material metadata is saved in localStorage only when requested. Do not use this prototype to store confidential supplier information without enterprise security controls.

## License
No license is granted for reuse of this code until the repository owner explicitly chooses one. Third-party datasets and pretrained weights have independent license requirements.

## Try the research model in your browser

1. Open the completed [MobileNet vision probe workflow](https://github.com/exmachina100/kumo-material-intelligence/actions/workflows/vision-probe.yml), select the successful run and download the `materialnet-vision-probe-research-onnx` artifact ZIP.
2. Extract `vision-probe-research.onnx` and `labels.json`.
3. Open `https://exmachina100.github.io/kumo-material-intelligence/apps/research/` and select those two files and a material image.
4. Use **Run on this device**. Local ONNX/WASM inference runs in the browser; the ONNX runtime itself loads from a pinned CDN on first use. Avoid interpreting the raw softmax score as calibrated confidence.

The research model is *not* installed in the normal CMF workstation, intentionally. Browser/phone compatibility still needs direct device testing.
