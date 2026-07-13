Finales Fusionsmodell

Das Modell kombiniert 152 Boruta-selektierte DINOv3-Normalmap-Merkmale mit
128 mRMR-selektierten DINOv3-Albedo-Merkmalen. Fuer beide Modalitaeten werden
die Layer 1 bis 12 sowie Min-, Max- und Mean-Aggregation verwendet. Die
Albedo-Merkmale werden an denselben Patchpositionen extrahiert, die anhand
der Normalmap als anomal bestimmt wurden.

classifier_pipeline.joblib
    Trainierte RBF-SVM inklusive StandardScaler.

selected_normal_feature_indices.npy
    Indizes der 152 Normalmap-Merkmale.

selected_albedo_feature_indices.npy
    Indizes der 128 Albedo-Merkmale.

selected_albedo_features.csv
    Beschreibung der ausgewaehlten Albedo-Merkmale nach Layer,
    Aggregationsart und Kanal.

test_metrics.json
    ROI- und Bauteilmetriken auf dem separaten Testdatensatz.
