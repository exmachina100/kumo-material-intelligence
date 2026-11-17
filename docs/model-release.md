# MaterialNet-S model release gate

## Data
- Use only data authorized for training and distribution.
- Group train, validation and test splits by physical material sample, product and source.
- Publish per-class counts and evaluation methodology.

## Model
- Train using `python training/train_classifier.py --data training/data --pretrained` after installing `training/requirements.txt`.
- `training/package_model.py` packages the actual ONNX artifact with checksum, input preprocessing and label set.
- The optional JS ONNX adapter currently accepts a session injected by a host. Integration into the UI requires a separate browser loader and validated preprocessing parity.

## Critical consistency checks
- Python training uses torchvision Resize(...), which normally uses bilinear interpolation. The current JS ONNX adapter uses nearest-neighbor sampling. These are not equivalent; fix and test before production deployment.
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
