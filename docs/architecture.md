# Architecture decisions (v0.2)

## Trust and provenance
No physical composition, process, specification or compliance is inferred as fact from a photograph. Classification has a confidence threshold and always remains unverified until reviewed.

## Execution
- Browser: deterministic color extraction and palette segmentation run on the main thread for MVP.
- Future validated ONNX vision models are loaded through an optional inference adapter; no fake weights or hidden API calls.
- Mobile: PWA first; native inference after hardware benchmarking.
- Cloud: never automatically transmit uploaded customer imagery.

## Data
Material DNA schema 0.1.0, user corrections explicitly marked designer-authored. Image files remain in-memory; stored records contain colors and metadata, not photos.

## Privacy
The app is local-only by default. Browser localStorage is suitable for prototypes, not security-sensitive designs. Before enterprise use, implement encrypted storage, retention, and organizational access control.

## Evaluation gate for MaterialNet-S
Macro F1 / per-family recall, confusion analysis, unknown detection calibration, cold start, model download, peak memory and per-device p50/p95 inference.
