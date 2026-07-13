# Analysis Tools

Dieser Ordner enthaelt Werkzeuge zur nachtraeglichen Analyse, Visualisierung und Ergebnisaufbereitung.
Die Skripte sind nicht Teil der eigentlichen App-Laufzeit.

## Enthaltene Werkzeuge

- `plot_selected_feature_subset_pca2d.py`: erzeugt PCA-Scatterplots fuer ausgewaehlte mRMR- oder Boruta-Merkmale.
- `plot_test_roi_albedo_fusion_pca2d.py`: visualisiert die Test-ROIs des Normalmap-Albedo-Fusionsmodells in zwei PCA-Dimensionen.
- `plot_test_roi_albedo_fusion_pca3d.py`: erzeugt die entsprechende statische 3D-PCA-Ansicht.
- `plot_cv_roi_albedo_fusion_pca2d.py`: visualisiert den Kreuzvalidierungsdatensatz mit foldsicheren OOF-Fehlermarkierungen.
- `analyze_fusion_pr_threshold_cv.py`: erstellt die PR-Kurve und untersucht den 3D-Wahrscheinlichkeitsschwellenwert auf OOF-Daten.
- `analyze_selected_feature_distribution.py`: zaehlt ausgewaehlte Merkmale nach DINOv3-Layer und Aggregationsart.
- `export_patchgrid_anomaly_map_pngs.py`: exportiert farbliche Patchgrid-Anomaly-Maps.
- `export_labeling_raw_browse_folder.py`: bereitet Bilder zur manuellen Sichtung/Label-Erstellung auf.
- `collect_patchgrid_disagreements.py`: sammelt Unterschiede zwischen Patchgrid- und ROI-Darstellungen.
- `inspect_patch_nearest_neighbor.py`: untersucht naechste Nachbarn einzelner Patches.
- `overlay_heatmaps.py`: Hilfswerkzeug zum Rendern von Heatmap-Overlays.

## Hinweis

Die finale App wird ueber `anomalydino_app/app.py` gestartet.
Die Modellbau-Skripte liegen getrennt in `model_building`.
