# AnomalyDINO Similarity Explorer

Lokale Streamlit-App fuer dense Patch-Aehnlichkeit auf dem bestehenden `res=688`-AnomalyDINO-Run.

## Was die App macht

- laedt die vorhandenen Patch-Features aus `patch_feature_cache`
- zeigt ein klickbares `Query`-Bild mit Patchgitter
- setzt per Klick einen oder mehrere Query-Patches
- Klick auf denselben Patch entfernt ihn wieder
- bildet aus mehreren Query-Patches einen gewichteten Mittelwert der Patch-Features
  - Gewichtung ueber eine `Softmax` auf den Query-Anomaliescores
- unterstuetzt drei Featurequellen:
  - `Normalmap DINOv3`
  - `Albedo DINOv3`
  - `Fusion: Normalmap + Albedo`
- bei `Fusion` wird die Similarity scorebasiert gemischt:
  - `sim_fused = alpha * sim_normal + (1 - alpha) * sim_albedo`
- vergleicht den Query-Patch per `Cosine Similarity`
  - entweder gegen dasselbe Bild
  - oder gegen ein frei waehlbares anderes Zielbild
- optional nur gegen Target-Patches, deren Anomaliescore ueber dem Bildthreshold liegt
- optional mit `Positionsbias mit good-reference entfernen`
  - lernt pro Patchposition einen normalen Referenzvektor aus `good`-Bildern
  - nutzt dabei in den good-Bildern nur low-anomaly Patches `<= image_threshold`
  - projiziert diese normale Richtungs-Komponente bei Query und Target heraus
- optional mit `PCA-Subspace-Debias`
  - umschaltbar zwischen `k=2` und `k=3`
  - baut zusaetzlich pro Patchposition einen kleinen normalen Rest-Subspace aus `good`-Bildern
  - entfernt erst die mittlere Richtungs-Komponente und dann die `k` dominantesten normalen Rest-Richtungen
  - ist fuer den Vergleich gegen Positions-/Lagebias gedacht
- optional mit `I-Relief-Reweight auf Normalmap anwenden`
  - laedt den zuvor gelernten `I-Relief`-Gewichtsvektor aus dem ROI-Feature-Satz
  - skaliert damit die finalen `Normalmap`-Patchfeatures
  - normalisiert danach wieder auf `L2`
  - laesst den Retrieval-Ablauf ansonsten unveraendert, damit `ohne` vs. `mit` direkt vergleichbar bleibt
- optional mit `Multi-Layer (Layer 1-12) + I-Relief auf Normalmap verwenden`
  - laedt den `patch_feature_cache_multilayer_l1to12`
  - normalisiert pro Patch zuerst jeden Layerblock und dann den konkatenieren `9216D`-Vektor
  - laedt den dazu gelernten Multi-Layer-`I-Relief`-Gewichtsvektor
  - skaliert damit den finalen Normalmap-Patchvektor und normalisiert danach wieder auf `L2`
  - laesst Query-Selektion, Softmax-Mittelung und Cosine-Retrieval ansonsten unveraendert
  - dieser Modus laeuft auf dem Normalmap-Zweig bewusst `ohne` Positionskorrektur, damit er direkt dem zuvor gelernten Multi-Layer-Satz entspricht
- zeigt die Similarity-Heatmap als Overlay auf dem Zielbild
- zeigt optional zusaetzlich die vorhandene Target-Anomaly-Map
- listet die Top-Matches im Zielbild und zeigt deren Patch-Crops
- exportiert den aktuellen View als `PNG + JSON`

## Start

```powershell
C:\ai\AnomalyDINO\.venvAnomalyDINO\Scripts\python.exe -m streamlit run C:\ai\AnomalyDINO\anomalydino_similarity_app\app.py
```

Optional mit explizitem Experiment-Ordner:

```powershell
C:\ai\AnomalyDINO\.venvAnomalyDINO\Scripts\python.exe -m streamlit run C:\ai\AnomalyDINO\anomalydino_similarity_app\app.py -- "C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
```

Dann im Browser:

```text
http://localhost:8501
```

## Voraussetzung

Der Run muss enthalten:

- `anomaly_maps/seed=0/.../*.npy`
- `patch_feature_cache/seed=0/cache_manifest.csv`
- die `.npz` Feature-Caches aus dem Manifest
- fuer `Albedo` oder `Fusion` zusaetzlich:
  - `albedo_patch_feature_cache/seed=0/cache_manifest.csv`
  - die zugehoerigen `.npz` Albedo-Feature-Caches
- fuer `I-Relief-Reweight` zusaetzlich:
  - `roi_top10pct_centerinbox_pca2_softmax_patch_features_labeled/irelief_cosine_weighted_features/irelief_feature_scale_sqrt.npy`
- fuer `Multi-Layer (Layer 1-12) + I-Relief auf Normalmap verwenden` zusaetzlich:
  - `patch_feature_cache_multilayer_l1to12/seed=0/cache_manifest.csv`
  - die zugehoerigen Multi-Layer-`.npz`-Caches
  - `roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled/irelief_cosine_weighted_features/irelief_feature_scale_sqrt.npy`

## Export

Exporte landen unter:

```text
<experiment_dir>/similarity_query_exports/seed=<seed>/
```

Pro Export werden geschrieben:

- ein zusammengesetztes `PNG`
- eine `JSON` mit Query-, Target- und Top-Match-Metadaten
- ein `manifest.jsonl`

## Hinweis

Die Similarity-Map ist eine DINOv3-Feature-Aehnlichkeitskarte, nicht die AnomalyDINO-Anomaly-Map.

Der Schalter `Nur target patches ueber Bildthreshold vergleichen` benutzt weiterhin den
AnomalyDINO-Threshold des aktuellen `Normalmap`-Runs als Target-Maske.
