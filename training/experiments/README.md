# Experimental baseline: catalog-proxy material classification

This is a small scikit-learn logistic regression over 36 image features (RGB histograms, moments and greyscale gradients), trained on a limited number of Poly Haven CC0 thumbnail previews. It is **not** a calibrated recognition model, not a MobileNet/ONNX model, and not material-composition verification.

The catalog's material-category matches are weak labels. A score on held-out *catalog labels* estimates only how well the pipeline reproduces this imperfect source's category hints. Images may be CG renders or texture-map previews rather than camera photographs. Do not advertise the score as accuracy on automotive interiors, CMF samples or consumer products.

The GitHub workflow produces a small JSON model and a report artifact for research only. No trained weights are deployed into Allure, and the Material DNA production classifier remains disabled.

To get a useful real CMF model:
1. Review source photos in `apps/review` and confirm relevant classes.
2. Add independent physical sample photos for each family with varied lighting and cameras.
3. Train a compact pretrained vision encoder with a calibration/unknown-detection study.
4. Compare against this tiny feature baseline; release only if real-world holdout performance is adequate.
