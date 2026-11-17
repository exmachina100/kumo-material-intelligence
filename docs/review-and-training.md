# Human-reviewed CMF reference data

## What is available

The Poly Haven CC0 source registry is stored at `training/external/polyhaven_cc0_staging/provenance.json`.
All catalog-discovered labels are **suggestions, not training targets**. The sources include architectural/CG textures and sometimes procedural assets, not necessarily photographed automotive substrates.

Review app: `/apps/review/` in the deployed GitHub Pages site. Source attribution to Poly Haven is shown in the UI.

## Review procedure

1. Open the review app and set a reviewer identifier.
2. Inspect each source image and, where appropriate, open the source page.
3. Choose *include* and a material family only when it is defensible; otherwise choose uncertain or exclude.
4. Assign evidence as visual-only or source-verified. Evidence is NEVER inferred from the filename.
5. Export `kumo-cmf-material-reviews.json` and securely retain the source plus reviewer's notes.
6. Do not merge labels from unresolved images into training.

## Build the reviewed dataset

```bash
python training/external/build_reviewed_dataset.py \
 --reviews path/to/kumo-cmf-material-reviews.json \
 --output training/reviewed_data --min-per-family 15
python training/audit_dataset.py --data training/reviewed_data
```

The builder fails when any included family has fewer than 15 approved samples. It verifies source license metadata, image checksums, reviewer information, and no cross-split SHA/group leakage. It produces a manifest and split folders usable by `training/train_classifier.py`.

**Important:** 15 samples per class is a structural gate, not a statistical accuracy benchmark. For real deployment, collect many more independent material samples, varied lighting/cameras, tactile and physical supplier verification where needed, and held-out real-world photographs.

## IP controls

The repo is public. CC0 references are acceptable to publish under those terms, while confidential OEM, supplier or internal CMF photographs must be stored separately under access control. Approval records should not include personally identifying reviewer data beyond a pseudonymous identifier.
