# MaterialNet-S model release gate

## Data
- Use only data authorized for training and distribution.
- Group train, validation and test splits by physical material sample, product and source.
- Publish per-class counts and evaluation methodology.

## Model
- Train using `python training/train_classifier.py --data training/data --pretrained` after installing `training/requirements.txt`.
- `training/package_model.py` packages the actual ONNX artifact with checksum, input preprocessing and label set.
- The optional JS ONNX adapter supports manually loaded local model files in the opt-in research browser lab at `/apps/research/`. The production CMF workstation still has no trained classifier. Release requires reviewed labels and independently validated preprocessing and calibration.

## Critical consistency checks
- Python training uses torchvision Resize(...), which normally uses bilinear interpolation. The current JS ONNX adapter now uses bilinear sampling, but does not implement TorchVision's antialias downsampling. A browser-vs-PyTorch input-tensor parity benchmark must pass before production deployment.
- Logits may not be calibrated probabilities. Report softmax as a score until calibration is evaluated on an independent set.
- Do not treat close visual similarity as proof of substrate, chemistry, manufacturing process, wear resistance or supplier certification.
- Do not accept arbitrary user-supplied ONNX model files without a trusted model distribution and integrity verification mechanism.

## Release criteria
1. Model/source license and checksum approved.
2. Holdout accuracy and per-family recall acceptable for intended domain.
3. Unknown rejection and confidence calibration evaluated.
4. Browser preprocessing and reference PyTorch inference parity tested.
5. Browser, iOS and Android resource usage benchmarked separately.
6. Human-readable release notes and rollback path.

Never set `review.approved` simply because the manifest was generated.

## Current experimental status (November 2026)

A frozen MobileNetV3-Small ImageNet vision encoder with a trained linear classification head reached 77.3% accuracy and 75.7% macro F1 against **22 held-out weak Poly Haven catalog labels**. ONNX output matched PyTorch logits to within approximately 5.25e-6 on a random input tensor. The experiment is intentionally marked `release_eligible=false` and is available only as a GitHub Actions artifact. The browser research lab can open that downloaded artifact locally; browser runtime performance has not yet been verified on physical iOS/Android devices.

Research workflow: `.github/workflows/vision-probe.yml`. Never interpret these exploratory results as calibrated accuracy on genuine automotive interiors, industrial surfaces, supplier materials or photographs from different devices.
