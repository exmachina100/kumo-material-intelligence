# Kumo Material Intelligence

**Version 0.2 — local-first CMF prototype** for automotive, product, and industrial design.

[Open browser workstation source](apps/demo/index.html) · [Architecture](docs/architecture.md) · [Mobile deployment](docs/mobile-and-deployment.md)

## Implemented
- Responsive browser workstation: drag/drop image, palette swatches, image preview, manual material family tagging, browser material library, and JSON export.
- Deterministic image color extraction and multi-color palette estimation with perceptual color difference (CIEDE2000).
- Material DNA 0.1.0 schema, JS validator and provenance fields; image samples are not uploaded to servers.
- Optional inference router and injectable ONNX adapter; **no classifier weights are included**.
- Service worker/PWA manifest, Node tests and GitHub Actions CI configuration.
- Reproducible Python baseline training/evaluation/ONNX export script, awaiting licensed and curated training data.

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
- Actual material recognition: **not yet trained or deployed**.
- Full PBR map generation, supplier specifications, advanced CMF reasoning and native apps: **not yet built**.
- Browser PWA icon assets, native mobile packaging, CI result confirmation, and hosted deployment need finishing.
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
