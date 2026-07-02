# Model Building

Dieser Ordner enthaelt die Skripte zur Erzeugung und Bewertung der finalen ROI-Klassifikatoren.
Die App-Laufzeit liegt getrennt in `anomalydino_similarity_app`.

## Zentrale Workflows

- `mrmr_feature_selection_sweep.py`: erstellt das mRMR-Ranking und bewertet verschiedene Top-k-Merkmalsmengen mit sklearn-Klassifikatoren.
- `boruta_mrmr_prefilter_maxminmean.py`: nutzt mRMR als Vorfilter und fuehrt danach die Boruta-Selektion aus.
- `crossval_mrmr_maxminmean_rbf_svm.py`: fuehrt die fold-sichere Kreuzvalidierung fuer mRMR-Merkmale mit RBF-SVM aus.
- `crossval_boruta_maxminmean_rbf_svm.py`: fuehrt die fold-sichere Kreuzvalidierung fuer Boruta-Merkmale mit RBF-SVM aus.
- `evaluate_boruta_selected_features_sklearn.py`: bewertet ein festes Boruta-Merkmalsset mit auswählbarem sklearn-Klassifikator.
- `evaluate_boruta_precomputed_folds_sklearn.py`: bewertet bereits vorab bestimmte Boruta-Fold-Selektionen mit LogReg, linearer SVM oder RBF-SVM.
- `roi_sklearn_groupcv.py`: enthaelt die gemeinsame Group-CV-Logik und die sklearn-Pipelines fuer LogReg, lineare SVM und RBF-SVM.
- `train_final_mrmr_rbf_svm_and_render.py`: trainiert das finale mRMR-RBF-SVM-Modell und erzeugt die zugehoerigen Artefakte.
- `train_final_boruta_rbf_svm_and_render.py`: trainiert das finale Boruta-RBF-SVM-Modell und erzeugt die zugehoerigen Artefakte.

## Hilfsmodule

- `rbf_svm_utils.py`: neutrale Hilfsfunktionen fuer RBF-SVM und Metriken.
- `overlay_render_utils.py`: kleine Hilfsfunktionen zum Rendern von ROI-Overlays.

## Hinweis

I-Relief, LogReg-only, Random-Forest-only, DINOv2 und Albedo/Fusion gehoeren nicht zum finalen Modellstand.
Eindeutig unbenoetigte Varianten wurden ins Archiv verschoben.
