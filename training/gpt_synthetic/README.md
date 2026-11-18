# GPT Image CMF material drip

This is a **real image-generation pipeline**, not procedural texture code or a design mockup.
It uses the OpenAI Image API and **requires the repository owner's API key and billing**.

## Frequency and outputs

- GitHub Actions workflow: [.github/workflows/gpt-material-drip.yml](../../.github/workflows/gpt-material-drip.yml).
- Schedule: **every day at 03:40 UTC / 09:10 IST**, with 6 generated material texture images per day (up to ~180 per 30 days). Scheduled jobs may run late.
- Model: **gpt-image-2.5-flare**. Model and quality are explicit, never silently switched to a discontinued model.
- Manual workflow runs support 1/2/4/6/8/12 images with a hard 12-image maximum per run. Manual dispatch defaults to **dry-run** to avoid accidental spending.
- Covers 13 material-family **prompt intents**, rotating across days: textile, synthetic-leather, polymer, rubber, metal, leather, wood, stone, ceramic, glass, composite, coating, paper.
- For each batch, the workflow uploads a ZIP with the full-resolution JPEG images, metadata manifest and preview sheet to a **GitHub Release** tagged \`gpt-cmf-YYYY-MM-DD\`.
- Git repo receives a compact contact-sheet preview at \`training/gpt_synthetic/previews/YYYY-MM-DD.jpg\` and an auditable manifest at \`training/gpt_synthetic/manifests/YYYY-MM-DD.json\`. This avoids inflating Git with gigabytes of generated binaries.
- Checksums, model, prompt, date, intended label, lighting and generation batch are preserved. Duplicate/near-duplicate rejection runs within each batch.

## One-time activation required

The GitHub code is committed and the no-cost validation workflow is tested.

**To enable unattended billed generation**, open:
[Repository → Settings → Secrets and variables → Actions](https://github.com/exmachina100/kumo-material-intelligence/settings/secrets/actions)

Create a **repository secret** named \`OPENAI_API_KEY\` with a valid image-generation-enabled OpenAI API key. Keep the key out of commits, issues, and chat messages. The secret is not part of ChatGPT subscription access; **API usage is billed separately**.

Before scheduling is financially meaningful, review [OpenAI API pricing](https://platform.openai.com/pricing/) and apply an API account/project usage budget and alerts. The code limits each scheduled batch, but cannot guarantee an external monetary budget.

After configuring the secret, open
[Actions → GPT CMF material image drip](https://github.com/exmachina100/kumo-material-intelligence/actions/workflows/gpt-material-drip.yml),
select **Run workflow**, set \`dry_run = false\` and \`count = 2\` for a low-spend first real test. This run will create a Release; subsequent scheduled runs attempt six new samples daily.

The workflow deliberately **fails with a configuration error before any API request** if the secret has not been set. It is therefore **not generating autonomously until the secret is present**.

## Dataset separation and quality controls

These assets are **synthetic AI images**, not photographed physical samples, physically based material scans, supplier-verified substrates or CC0 materials. All labels are **prompt intents**, not material truth. A human CMF reviewer must verify apparent family and obvious visual defects before any training use.

They are suitable as a supplemental **training-only** research source after review. Never place AI images in validation, test or independent field-photo benchmarks. Group samples from the same source batch/prompt, avoid nearest-neighbor duplicates, and report real-source vs synthetic training ratio separately.

A synthetic dataset must not determine class definitions, industry claims, real physical properties, sustainability or safety certification. Performance gains must be measured on **held-out real camera photographs** only.

## Failure and recovery

- GitHub Releases are date-keyed. If a successful batch Release already exists, scheduled reruns skip new billable requests that day.
- Up to three transient HTTP retries with bounded exponential backoff. Incomplete batches are not published, but successful requests may already be billed.
- Full-size images are stored as GitHub Release ZIPs, not as Git blobs; Releases and Actions artifacts are not backups of confidential material assets.
- GitHub Actions schedule may be delayed. Public repositories with no activity for 60 days may have scheduled jobs disabled.
- Any edited or regenerated sample requires a new provenance record; do not convert prompt intents into ground-truth labels automatically.

## Local development

\`\`\`bash
python -m unittest discover -s tests -p 'test_gpt_material_drip.py' -v
python training/gpt_synthetic/drip.py --date 2026-11-18 --count 6 --dry-run
\`\`\`

To run with billing (after setting \`OPENAI_API_KEY\` securely in your own environment):

\`\`\`bash
python -m pip install "Pillow>=10,<13"
python training/gpt_synthetic/drip.py --date 2026-11-18 --count 2 --output /tmp/kumo-cmf-2026-11-18
\`\`\`
