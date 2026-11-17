# Kumo CMF synthetic training set (v0.1)

468 deterministic **synthetic vector material swatches**, 13 material families × 36 each. Each family: 30 train / 3 validation / 3 test; 390/39/39 total.

**IMPORTANT:** These are *illustrative procedural patterns*, not photographs, PBR-calibrated surfaces, measured materials, or proof of classification performance. They should be used for pipeline testing, augmentation, or UI demos only. The validation and test subsets are synthetic smoke-test splits, NOT valid estimates of real-world accuracy. Synthetic generation artifacts can leak across splits. Build independent real-material, sample-grouped evaluation data before claiming CMF recognition.

Files are SVG to keep the Git repository lightweight. Convert to RGB PNGs using `python training/synthetic/convert_svg.py` (requires `pip install cairosvg pillow`). By default the converted samples go to `training/data`, which is gitignored. Then run `python training/audit_dataset.py` and, for a smoke test only, `python training/train_classifier.py --epochs 2`.

All sources were procedurally generated as part of this project; no third-party images or supplier materials included. Don't treat the family label as verified physical composition.
