# CMF material data sources and licensing

| Source | Source URL | Intended use | Rights status |
|---|---|---|---|
| Poly Haven | https://polyhaven.com/textures | Real-world scanned textures, PBR maps, some automotive-relevant textiles, wood, leathers | CC0 assets. API service has separate Terms. |
| ambientCG | https://ambientcg.com/gallery | Real-world PBR textures and maps, particularly textiles, metal, polymer-like surfaces | CC0 assets and preview renders. Download/API details should be verified. |
| MatSynth / Adobe research | https://github.com/Code-SY95/MatSynth | Material/appearance research, multi-map PBR and rendered exemplars | **Mixed source asset licenses**: check individual records; NOT auto-ingested. |
| OpenSurfaces | https://opensurfaces.cs.cornell.edu/ | In-context material annotations, surface segmentation | Annotations CC BY 4.0; underlying source image rights need individual review. NOT auto-ingested. |
| DTD Oxford VGG | https://www.robots.ox.ac.uk/~vgg/data/dtd/ | Texture-attribute research baseline | Academic/research availability; no blanket commercial reuse permission established. NOT auto-ingested. |
| MINC | http://opensurfaces.cs.cornell.edu/publications/minc/ | Material-category image understanding baseline | Check image-level source rights and restrictions before commercial model training. NOT auto-ingested. |

## Reviewed ingestion policy
- Only curated Poly Haven assets are automatically fetched in the pilot, as **staging candidates** with provenance. Images remain labeled UNREVIEWED until human confirmation.
- Preserve exact source URI, source author, license and SHA-256; don't falsely claim a physical substrate or performance property from appearance.
- Don't mix different views of the same material asset between train and test.
- Published previews differ from physically calibrated PBR source maps. For serious training, use original 1K/2K color/roughness/normal assets and photograph-based holdouts.
- Do not commit private OEM reference photographs, confidential supplier samples, or proprietary studio files into this public GitHub repository.
- Real-material evaluation needs independent samples, product contexts, lighting variation and expert labels.

## First licensed catalog targets
- Poly Haven leather: fabric_leather_01, fabric_leather_02, leather_white, brown_leather.
- Poly Haven textiles: bi_stretch.
- Poly Haven veneer: white_oak_veneer, natural_walnut_veneer, walnut_veneer, oak_veneer_01.
- Other verified pages: brushed_concrete, anti_slip_concrete, beige_wall_001, asphalt_01, bark_brown_01.

## Training warning
A classifier that succeeds on the synthetic procedural images may merely learn the generator patterns. It is **not** an accurate CMF classifier until it passes a physically independent, real-image benchmark.
