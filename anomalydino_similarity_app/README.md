# AnomalyDINO Explorer

Lokale Streamlit-App fuer dense Patch-Aehnlichkeit und Bauteil-Tests auf dem bestehenden `res=688`-AnomalyDINO-Run.

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
- optional mit `Top10%-Klassifikator: nur die 32 I-Relief-Features auf Normalmap verwenden`
  - nutzt den Multilayer-Normalmap-Zweig des `top10%`-Klassifikators
  - wendet zuerst dessen Multi-Layer-`I-Relief`-Reweight an
  - schneidet danach auf genau die gespeicherten `32` Klassifikator-Merkmale
  - normalisiert den reduzierten Vektor wieder auf `L2`
  - ist fuer einen direkten Patch-Retrieval-Vergleich gegen den Klassifikator-Merkmalsraum gedacht
- optional mit `Top1-Patch-Klassifikator auf Query-Patches auswerten`
  - baut aus den aktuell ausgewaehlten Query-Patches den gewohnten softmax-gewichteten Aggregatvektor
  - bei genau einem Query-Patch ist die Gewichtung automatisch `1.0`
  - schickt diesen Vektor danach durch den finalen `top1patch`-Endklassifikator
  - zeigt `2D/3D` sowie `p(2D)` und `p(3D)` direkt in der App an
  - ist fuer manuelles Testen des Einzelpatch-Klassifikators auf frei ausgewaehlten Fehlerpatches gedacht
- zeigt die Similarity-Heatmap als Overlay auf dem Zielbild
- zeigt optional zusaetzlich die vorhandene Target-Anomaly-Map
- listet die Top-Matches im Zielbild und zeigt deren Patch-Crops
- exportiert den aktuellen View als `PNG + JSON`
- zusaetzlich gibt es die Ansicht `Bauteil-Test`
  - kann entweder auf den im aktuellen Run bereits vorhandenen AnomalyDINO-Scores und ROI-Boxen arbeiten
  - oder externe Bilder mit frischer End-to-End-Inferenz verarbeiten
    - DINOv3-Backbone laden
    - Referenzbank aus den gespeicherten Few-Shot-Referenzbildern des Runs aufbauen
    - frische Patch-Anomaliescores berechnen
    - ROI-Boxen mit waehlbarer Extraktionslogik erzeugen:
      - `Alte Normalmap-Logik`: klassische Hysterese mit `merge_gap=3`
      - `Neue Buttons-Logik`: Hysterese mit `merge_gap=1`, Coverage-Pass und Touch-Merge
    - danach das finale `Boruta + RBF-SVM`-Endmodell pro ROI anwenden
  - nutzt fuer die ROI-Klassifikation das finale `Boruta + RBF-SVM`-Endmodell
  - entscheidet pro Bild erst `IO/NIO`, danach pro ROI `2D/3D` und am Ende auf Bauteilebene:
    - sobald irgendeine ROI `3D` ist, ist das ganze Bauteil `3D`
    - sonst ist ein `NIO`-Bauteil `2D`
  - rendert Overlays nur, wenn die Checkbox dafuer gesetzt wird
- zusaetzlich gibt es die Ansicht `Einstellungen`
  - mit dem Reiter `Anomaly Detection`
  - dort kann fuer die frische AnomalyDINO-Evaluierung zwischen zwei Inputs gewechselt werden:
    - `normalmap`
    - `normalmapalbedo`
  - `normalmapalbedo` nutzt aktuell den `buttons`-Fusion-Run
  - dort koennen pro Objekt gespeichert werden:
    - `Referenzbilder`
    - `Testbilder good`
    - `Testbilder bad`
  - neue Uploads werden der jeweiligen Kategorie hinzugefuegt
  - dieselbe Kategorie kann also mehrfach nacheinander befuellt werden
  - die Bilder werden lokal unter dem aktuellen Experiment gespeichert
  - per Button `Evaluierung starten` werden fuer alle gespeicherten Testbilder frische Bildscores berechnet
  - danach wird der beste `Bildthreshold` nach `F1` fuer `good/bad` bestimmt
  - ein feiner Schieberegler plus exakte Zahleneingabe erlauben anschliessend Live-Analyse von:
    - `Precision`
    - `Recall`
    - `F1`
    - `Accuracy`
  - per Button kann der aktuell eingestellte `Bildthreshold` bestaetigt und als aktive Konfiguration gespeichert werden
  - der externe Modus im `Bauteil-Test` verwendet dann genau diese bestaetigte Anomaly-Detection-Konfiguration
    - bestaetigter `Bildthreshold`
    - bestaetigte Referenzbilder
  - diese Seite bewertet bewusst nur `IO/NIO` auf Bildebene und nicht ROI- oder `2D/3D`-Klassifikation
  - mit dem Reiter `Testdatensatz`
    - Testbilder koennen lokal fuer ein Objekt gespeichert werden
    - pro Bild werden auf Bauteilebene gepflegt:
      - `part_id`
      - `gt_part_label` in `IO`, `2D`, `3D`
      - optionale Notiz
    - mehrere Bilder mit derselben `part_id` werden in der Evaluierung zu einem Bauteil aggregiert
    - die Test-Evaluierung nutzt bewusst die aktuell bestaetigte `Anomaly Detection`-Konfiguration:
      - aktuelle Referenzbilder
      - aktueller bestaetigter Bildthreshold
    - als ROI-Klassifikator kann zwischen
      - `Boruta`
      - `mRMR`
      gewaehlt werden
    - gespeichert werden:
      - Bildvorhersagen
      - Bauteilvorhersagen
      - Summary mit `Accuracy`, `Macro Precision`, `Macro Recall`, `Macro F1`

## Start

Abhaengigkeiten fuer die App installieren:

```powershell
pip install -r .\anomalydino_similarity_app\requirements_app.txt
```

```powershell
cd <repo-root>
python -m streamlit run .\anomalydino_similarity_app\app.py
```

Optional mit explizitem Experiment-Ordner:

```powershell
python -m streamlit run .\anomalydino_similarity_app\app.py -- ".\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
```

Dann im Browser:

```text
http://localhost:8501
```

## Voraussetzung

Der GitHub-Stand enthaelt die kleinen finalen App-Artefakte:

- `app_settings/anomaly_detection/normalmap/confirmed_threshold_config.json`
- `final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf/classifier_pipeline.joblib`
- `final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf/selected_feature_indices.npy`
- `final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf/selected_features.csv`
- `final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf/model_info.json`
- `final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rbf/classifier_pipeline.joblib`
- `final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rbf/selected_feature_indices.npy`
- `final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rbf/selected_features.csv`
- `final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rbf/model_info.json`

Fuer eine frische End-to-End-Inferenz muessen zusaetzlich die grossen Daten- und Cache-Artefakte lokal vorhanden sein. Diese werden nicht in Git versioniert:

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
- fuer `Top10%-Klassifikator: nur die 32 I-Relief-Features auf Normalmap verwenden` zusaetzlich:
  - `final_all_boxes_top10pct_multilayer_irelief_fixedk32_rbf/selected_topk_features.csv`
  - sowie weiterhin der Multi-Layer-I-Relief-Gewichtsvektor aus
    `roi_top10pct_centerinbox_multilayer_l1to12_softmax_patch_features_labeled/irelief_cosine_weighted_features/irelief_feature_scale_sqrt.npy`
- fuer `Bauteil-Test` zusaetzlich:
  - `roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1/seed=0/roi_metadata.csv`
  - `patch_feature_cache_multilayer_l1to12/seed=0/cache_manifest.csv`
  - die zugehoerigen Multi-Layer-`.npz`-Caches
  - `final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf/classifier_pipeline.joblib`
  - `final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf/selected_features.csv`

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
