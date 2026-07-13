# AnomalyDINO Bauteil-Test-App

Die Streamlit-App stellt die produktiven Ansichten `Bauteil-Test` und `Einstellungen`
fuer die Normalmap-basierte AnomalyDINO-Pipeline bereit.

## Funktionsumfang

### Bauteil-Test

- Verarbeitung vorhandener Run-Bilder oder externer Uploads
- frische AnomalyDINO-Inferenz fuer externe Normalmaps
- ROI-Extraktion aus auffaelligen Patches
- ROI-Klassifikation mit folgenden Modellen:
  - RBF-SVM mit Boruta-Merkmalen
  - RBF-SVM mit mRMR-Merkmalen
  - ExtraTrees mit Boruta-Merkmalen
  - RBF-SVM mit Normalmap- und Albedo-Merkmalen, nur fuer externe Uploads
- Bauteilentscheidung: Sobald mindestens eine ROI als `3D` klassifiziert wird, gilt das
  gesamte Bauteil als `3D`; andernfalls gilt ein NIO-Bauteil als `2D`.
- optionale ROI-Overlays und gemeinsamer ZIP-Download
- Export einer ROI-Testtabelle und Berechnung von ROI- und Bauteilmetriken aus einer
  nachtraeglich gelabelten Tabelle

Das Fusionsmodell erwartet paarweise Normalmap- und Albedobilder mit identischem
Dateinamen und identischer Bildgroesse. Es verwendet 152 Boruta-selektierte
Normalmap-Merkmale und 128 mRMR-selektierte Albedo-Merkmale. Die 3D-Entscheidung wird
mit dem auf foldsicheren OOF-Daten bestimmten Wahrscheinlichkeitsschwellenwert
`p(3D) >= 0.6104560494422913` getroffen. Optional kann das ROI-Overlay auf dem
Albedobild dargestellt werden.

### Einstellungen

- Referenzbilder sowie `good`- und `bad`-Testbilder verwalten
- AnomalyDINO-Bildscores neu berechnen
- Bildthreshold anhand der Evaluierungsdaten untersuchen und bestaetigen
- gespeicherte Testdatensaetze verwalten und auswerten

Die bestaetigte Referenzbank und der Bildthreshold werden bei der frischen Inferenz
im Bauteil-Test verwendet.

## Start

Im Projektordner:

```powershell
.\.venvAnomalyDINO\Scripts\python.exe -m streamlit run .\anomalydino_app\app.py
```

Alternativ nach Installation der App-Abhaengigkeiten:

```powershell
python -m pip install -r .\anomalydino_app\requirements_app.txt
python -m streamlit run .\anomalydino_app\app.py
```

Optional kann der Experimentordner explizit uebergeben werden:

```powershell
python -m streamlit run .\anomalydino_app\app.py -- ".\results_FINAL\normalmap_dinov3_vitb16_res688"
```

Die App ist anschliessend unter `http://localhost:8501` erreichbar.

## Wichtige Artefakte

Der Standard-Experimentordner ist:

```text
results_FINAL/normalmap_dinov3_vitb16_res688
```

Er enthaelt die bestaetigten Anomaly-Detection-Einstellungen, die finalen
Klassifikatorartefakte und die ausgewaehlten Merkmalsindizes. Fuer frische
End-to-End-Inferenz werden ausserdem das DINOv3-Backbone sowie lokal gespeicherte
Referenzbilder benoetigt.

Der historische Similarity Explorer und seine exklusiven Hilfsskripte wurden aus der
produktiven App entfernt und lokal unter
`C:\ai\AnomalyDINO_archive_20260702\project_root\similarity_explorer_legacy_20260713`
archiviert.
