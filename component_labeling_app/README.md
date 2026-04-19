# Component Labeling App

Lokale Streamlit-App zum komponentenbasierten Labeln von AnomalyDINO-Ausgaben fuer den Aufbau von `2D`-/`3D`-Patch-Memory-Baenken.

## Voraussetzungen

1. Ein fertiger AnomalyDINO-Run mit:
   - `measurements_seed=*.csv`
   - `anomaly_maps/seed=*/*.npy`
   - `patch_feature_cache/seed=*/cache_manifest.csv`
2. Eine vorbereitete Labeling-Session aus:
   - [prepare_component_labeling_session.py](../prepare_component_labeling_session.py)

## Session vorbereiten

```powershell
C:\ai\AnomalyDINO\.venvAnomalyDINO\Scripts\python.exe C:\ai\AnomalyDINO\prepare_component_labeling_session.py `
  --experiment-dir "<RUN_DIR>" `
  --seed 0 `
  --output-dir "<SESSION_DIR>"
```

Wichtige Session-Dateien:
- `component_inventory.csv`
- `sample_inventory.csv`
- `summary.json`

## App starten

```powershell
C:\ai\AnomalyDINO\.venvAnomalyDINO\Scripts\python.exe -m streamlit run C:\ai\AnomalyDINO\component_labeling_app\app.py -- --session-dir "<SESSION_DIR>"
```

Alternativ kannst du den Session-Pfad im Sidebar-Feld eintragen.

## Was die App speichert

Im Session-Ordner:

- `component_annotations.csv`
  - persistente Komponentenlabels
  - Felder u.a.:
    - `sample`
    - `component_id`
    - `label` (`2D`, `3D`, `skip`)
    - `top_k` (`1..5`)
    - `notes`
- `part_labels.csv`
  - persistente Bauteillabels
  - Felder:
    - `sample`
    - `part_label` (`2D`, `3D`, `skip`)
    - `part_notes`
- `part_labels.json`
  - JSON-Spiegel der Bauteillabels

## Memory-Bank-Export aus der App

Ueber den Button `Export Memory Banks` erzeugt die App:

- `memory_bank_export/2D-memory-bank.npy`
- `memory_bank_export/3D-memory-bank.npy`
- `memory_bank_export/selected_patches.csv`
- `memory_bank_export/summary.json`

Die exportierten Patches stammen jeweils aus den `Top-k` staerksten Patches pro gelabelter Komponente.

## MVP-Umfang

Diese erste Version kann:

- Komponenten auf dem Patchgitter anzeigen
- binaere Anomaliemaske und Komponenten mit `8er`-Nachbarschaft visualisieren
- Komponenten als `2D` / `3D` / `skip` labeln
- `top_k` pro Komponente festlegen
- Bauteile separat als `2D` / `3D` / `skip` labeln
- persistent speichern und beim Neustart fortsetzen
- `2D`-/`3D`-Memory-Baenke exportieren

Noch nicht enthalten:

- automatische Fold-Evaluation in der App
- Patch-Crop-Galerien
- Threshold-Wechsel innerhalb derselben Session
