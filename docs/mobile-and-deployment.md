# Mobile, web, and deployment

## Browser MVP
Run `npm run serve` from repository root, then open `http://localhost:8080/apps/demo/`.
The app stores only Material DNA metadata in localStorage. Image pixels are processed locally in memory.
Service worker caches static assets after a successful load. Installing as a PWA may require icons and full device testing on the target browser.

## Static deployment
Host the repository as a static site preserving root paths, or change the demo's absolute asset paths before GitHub Pages under a repository subpath.
If GitHub Pages is used, set the hosting base path and service-worker cache scope correctly.

## Model integration gate
The training script produces ONNX weights, but the browser classifier adapter and verified input preprocessing are still required before enabling predictions. Never report classifications when weights are absent. Do not place the dataset in public git.

## Mobile strategy
1. Audit web inference performance in Safari/Chrome/Firefox.
2. Support responsive mobile capture via file input and installable PWA.
3. Add native bridge and converted ONNX/Core ML/Android model only after runtime benchmarking.
4. Use per-model signed manifests, SHA-256 checksums and compatibility negotiation.
5. Enterprise app: encrypted persistent storage, authenticated sync and opt-in cloud processing.
