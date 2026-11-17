# Training plan

1. Audit licensing and consent for material images and supplier data.
2. Define 12 broad material-family labels, plus multi-label appearance attributes.
3. Establish split by physical sample, product, source and photographer to prevent leakage.
4. Evaluate pretrained compact encoders as frozen embedding baselines.
5. Train a classification head, review errors with a CMF professional, and only then fine-tune.
6. Calibrate confidence and test unknown/out-of-distribution material images.
7. Export compatible ONNX artifact, quantize, benchmark accuracy, download size, memory and latency on mobile browsers.
8. Populate models/manifest.json with the deployed artifact's verified checksum and runtime requirements.

No weights are bundled; MVP uses deterministic color extraction only.