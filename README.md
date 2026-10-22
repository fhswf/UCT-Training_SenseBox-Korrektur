# Training des KI-Korrekturmodells (Stage 1 & 2) – Urban Climate Twin Soest

Dieses Repository enthält die Trainingspipeline für das Korrekturmodell des Urban Climate Twin Soest.
Das Modell verarbeitet Mess-, Stations- und Geodaten und schätzt korrigierte Temperaturwerte für SenseBox-Messungen.
Die Pipeline umfasst Datenaufbereitung, Feature Engineering, Modelltraining und Evaluation.
Trainingsparameter und Datenquellen werden über eine YAML-Datei konfiguriert.

## Urban Climate Twin Soest

 [Urban Climate Twin (UCT) Soest](https://www.fh-swf.de/de/forschung___transfer_4/forschungsprojekte_1/forschungsprojekt_uct.php) ist ein digitaler Zwilling des Soester Stadtklimas. Er führt Messdaten, Geodaten und weitere Klimadaten zusammen und macht das Stadklima räumlich nachvollziehbar. Bürger und die Stadtverwaltung können so die Auswirkungen von Klimaveränderungen auf das Stadtgebiet besser verstehen und Maßnahmen zur Anpassung an den Klimawandel entwickeln.

## Funktionsweise

Der Einstiegspunkt `main.py` lädt die gewählte Konfiguration und startet das Training. Die Pipeline:

1. lädt Messungen sowie Quellen- und Standortdaten aus einer PostgreSQL-Datenbank,
2. verarbeitet Messwerte, Nachbarstationen, DWD-Daten und konfigurierte Geofeatures,
3. bereitet Features sowie Trainings-, Validierungs- und Testdatensätze auf,
4. trainiert ein neuronales Regressionsmodell und wertet es gegen den Testsatz aus.

Pro Trainingslauf werden Modelle und Scaler unter `output/models/` sowie Trainingsprotokolle, Auswertungen und Zusammenfassungen unter `output/results/` gespeichert.

## Projektstruktur

```text
.
├── config/          # Trainings- und Datenkonfiguration
├── data/            # Laden und Aufteilen der Daten
├── evaluation/      # Evaluationsmetriken
├── model/           # Modellarchitektur und Training
├── preprocessing/   # Datenbereinigung und Feature Engineering
├── utils/           # Trainings-, Logging- und Visualisierungshilfen
├── output/          # Modelle und Trainingsergebnisse
└── main.py          # Einstiegspunkt der Trainingspipeline
```

## Lizenz

...