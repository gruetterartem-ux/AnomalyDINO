# Analysis Tools

Dieser Ordner enthaelt Werkzeuge zur nachtraeglichen Analyse, Visualisierung und Ergebnisaufbereitung.
Die Skripte sind nicht Teil der eigentlichen App-Laufzeit.

## Enthaltene Werkzeuge

- `similarity_explorer_app.py`: separater Streamlit-Similarity-Explorer zur Untersuchung von Patch-Aehnlichkeiten.
- `plot_selected_feature_subset_pca2d.py`: erzeugt PCA-Scatterplots fuer ausgewaehlte mRMR- oder Boruta-Merkmale.
- `analyze_selected_feature_distribution.py`: zaehlt ausgewaehlte Merkmale nach DINOv3-Layer und Aggregationsart.
- `export_patchgrid_anomaly_map_pngs.py`: exportiert farbliche Patchgrid-Anomaly-Maps.
- `export_labeling_raw_browse_folder.py`: bereitet Bilder zur manuellen Sichtung/Label-Erstellung auf.
- `collect_patchgrid_disagreements.py`: sammelt Unterschiede zwischen Patchgrid- und ROI-Darstellungen.
- `inspect_patch_nearest_neighbor.py`: untersucht naechste Nachbarn einzelner Patches.
- `overlay_heatmaps.py`: Hilfswerkzeug zum Rendern von Heatmap-Overlays.

## Hinweis

Die finale App wird ueber `anomalydino_similarity_app/app.py` gestartet.
Die Modellbau-Skripte liegen getrennt in `model_building`.
