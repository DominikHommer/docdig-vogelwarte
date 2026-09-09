# DocDig – Offene Punkte und Dokumentationsstand (für eine Bachelorarbeit)

Stand: September 2026, Basis ist der letzte Stand des GitLab-Projekts
`ringing-centre/docdig` der Schweizerischen Vogelwarte (Commit `80ca79f`,
9. Juli 2026). Dieses Repo ist ein privater Mirror davon.

Dieses Dokument richtet sich an eine Person, die DocDig im Rahmen einer
Bachelorarbeit weiterentwickeln oder untersuchen will. Es beantwortet drei
Fragen: Was ist da? Was ist dokumentiert? Was ist offen?

---

## 1. Worum geht es

DocDig digitalisiert handschriftlich ausgefüllte Beringungsprotokolle
(Tabellen auf gescannten A4-Seiten, Jahrgang 1972 als Testmaterial). Pro Seite
läuft eine Pipeline aus klassischer Bildverarbeitung und mehreren
Erkennungsmodellen, deren Ergebnis in einem Streamlit-Editor von einem
Menschen korrigiert und als CSV auf eine Nextcloud exportiert wird.

Kernidee: Nicht ein einzelnes OCR-Modell, sondern ein **Ensemble** aus
Tesseract, einem YOLOv8-Ziffern-Detektor, TrOCR, HTR-VT und einem eigenen
CNN für die Geschlechtsspalte. Ein Konsens-Modul stimmt ab, vergibt einen
Score und markiert unsichere Zellen für die manuelle Nachprüfung.

---

## 2. Was ist wo (Dokumentationsstand)

### 2.1 Vorhandene Dokumentation

| Ort | Inhalt | Umfang |
|---|---|---|
| `ReadMe.md` | Voraussetzungen, Modelle, Konfigurationsdateien, Pipeline-Übersicht, Spaltenschema, Editor-Architektur, Performance, numerisches Ensemble, Zell-Dictionary-Vertrag, Modulliste, Robustheit, Nextcloud-Export, Deployment | 357 Zeilen, technisch, teils Deutsch, teils Englisch |
| `tests/README.md` | Testsuite, Marker, Fixtures, warum manche Integrationstests "weich" sind | ca. 90 Zeilen |
| `config/column_schema.json` | Gedruckte Spaltenreihenfolge, Keywords, Breitenanteile | Schema mit Kommentarfeldern |
| Docstrings in `src/modules/*.py` und `src/libs/*.py` | Modulverhalten, Randfälle, Messwerte | uneinheitlich, teils sehr ausführlich |
| `.gitlab-ci.yml` | Build, Health-Check, Promote, Deploy | Pipeline ist für GitLab geschrieben |

### 2.2 Was fehlt

- **Architektur-Dokumentation** über die Pipeline-Skizze hinaus: Datenfluss,
  Zell-Dictionary als zentrale Datenstruktur, wo Scores entstehen, wie der
  Editor mit dem Session-State umgeht. Ein Architekturdiagramm existiert nicht.
- **Evaluationsmethodik**: `tools/evaluate_corpus.py` misst gegen den
  Ground-Truth-Korpus `files-3/`, aber wie die Metriken definiert sind
  (exakte Übereinstimmung, Normalisierung, was zählt als Treffer bei
  Dezimalzahlen) steht nur im Code.
- **Datensatz-Dokumentation**: Herkunft, Umfang, Labeling-Regeln für
  `files-3/`, `tests/fixtures/cells/`, den M-W-Datensatz und die
  YOLO-Trainingsdaten. Ohne das lässt sich nichts reproduzieren.
- **Trainings-Dokumentation** für die eigenen Modelle (Sexe-CNN, YOLO-Ziffern,
  Denoiser, Klassifikator): Hyperparameter, Datenaufteilung, erreichte
  Validierungswerte. Die Skripte unter `tools/train_*.py` sind da, ein
  Protokoll nicht.
- **Nutzerdokumentation** für die Beringungszentrale: Wie bedient man den
  Editor, was bedeuten ✓ und ⚠, wie setzt man den Bague-Anker.
- **Änderungshistorie**: Kein CHANGELOG, die Git-Historie ist die einzige
  Quelle. Entscheidungen (z. B. warum HTR-VT für die Altersspalte entfernt
  wurde) stehen verstreut in Commit-Nachrichten und Docstrings.
- Der Ordner `docs/` war bis zu diesem Dokument leer.

### 2.3 Was nicht im Repo liegt

Diese Dateien sind per `.gitignore` ausgeschlossen und müssen separat
beschafft werden (bei Dominik Hommer nachfragen):

| Datei / Ordner | Grösse | Wozu |
|---|---|---|
| `config/htr_vt/best_CER.pth`, `best_WER.pth` | je 408 MB | HTR-VT-Checkpoints für die Artenspalte |
| `config/pubtables1m_detection_detr_r18.pth`, `..._structure_detr_r18.pth` | je 110 MB | Table Transformer (TATR), öffentlich bei Microsoft verfügbar |
| `data/input/*.pdf` | 89 MB | Original-Scans für die Integrationstests |
| `M-W-Classification/dataset/` | 28 MB | Trainingsdaten des Sexe-CNN (m / none / w, ca. 1100 Bilder je Klasse) |
| `Digit-Recognition/DIDA/` | 470 MB | Öffentlicher Ziffern-Datensatz, nur für Retraining nötig |
| `.env` | klein | Nextcloud-Zugangsdaten, nur für den Export nötig |

Im Repo enthalten sind: alle kleinen Modelle in `config/` (Keras, YOLO), der
Ground-Truth-Korpus `files-3/` mit sechs gelabelten Seiten, sowie 152
Zell-Fixtures mit Labels unter `tests/fixtures/cells/`.

---

## 3. Gemessener Ist-Zustand

Gegen den Korpus `files-3/` (sechs Seiten, Jahrgang 1972), gemessen im Juli
2026 mit `tools/evaluate_corpus.py`:

| Spalte | Genauigkeit | Bemerkung |
|---|---|---|
| Bague (Ringnummer) | 97 % | Sequenz-Rekonstruktion ab manuellem Anker |
| Espèce (Art) | 87 % | Fuzzy-Snap gegen Artenkatalog, Schwelle 66 |
| Sexe (Geschlecht) | 71 % | 3-Klassen-CNN, kann X und Klammern nicht ausgeben |
| Aile (Flügel) | 58 % | Restfehler sind Einzelziffern-Verleser innerhalb plausibler Länge |
| Poids (Gewicht) | 43 % | wie Aile, zusätzlich Dezimalstellen |
| Age, Jour/Mois, Heure | nicht gemessen | Ground Truth lässt diese Spalten leer |

Weitere Messwerte:

- Ziffern-Backends auf 49 gedruckten Bague-Zellen: YOLO 41, Tesseract 36, CRNN 4 richtige Endziffern.
- Altersspalte auf 49 selbst gelabelten Zellen: 48 richtig (Regel "n oder u im Text → Fnd, sonst Fd" vor dem Fuzzy-Matching).
- Spaltenerkennung: nur 5 von 8 gedruckten Kopfzeilen sind per OCR lesbar, der Rest wird über Position und Breite zugeordnet.
- Laufzeit: ca. 74 s pro Seite auf CPU (vorher 96 s), rein CPU-gebunden.
- Testsuite: 189 Testfunktionen in 21 Dateien, davon Integrationstests mit echtem Scan.

---

## 4. Offene Punkte

Nach Themen gruppiert. Die Einschätzung in Klammern ist eine grobe
Aufwandsschätzung als Orientierung.

### 4.1 Erkennungsgenauigkeit

1. **Aile und Poids** liegen bei 58 % und 43 %. Die Fehler sind
   Einzelziffern-Verleser mit richtiger Stellenzahl, ein Plausibilitäts-Prior
   fängt sie nicht. Ansätze: Sequenz-Modell statt Einzelziffern, feineres
   YOLO-Training auf handschriftlichen Messwerten, Nachtrainieren von TrOCR
   auf Zell-Crops. (gross, gut messbar, klarer Forschungskern)
2. **Sexe-Klassifikator** kennt nur m / none / w. In den Protokollen kommen
   auch X, Klammern und Fragezeichen vor. Klassenraum erweitern, Datensatz
   nachlabeln. (mittel)
3. **Ditto-Zeichen** (Wiederholung des Werts aus der Zeile darüber) werden nur
   über den Tintenanteil erkannt. Die Notationen `''`, `//` und `ii` aus
   `files-3/meta.json` werden nicht unterschieden. (klein bis mittel)
4. **CRNN-Ziffernmodell** (`config/digit_model.keras`) liefert 4 von 49 und
   ist per Default abgeschaltet. Entweder auf DIDA plus eigenen Zellen neu
   trainieren oder ganz entfernen. (mittel)
5. **Spaltenerkennung** hängt an 5 lesbaren Kopfzeilen von 8. Für andere
   Jahrgänge oder Formularvarianten muss geprüft werden, ob das
   Needleman-Wunsch-Alignment noch trägt. (mittel, braucht neue Scans)
6. **Zeilenextraktion** enthält ein `FIXME` in `src/modules/row_extractor.py`:
   eine harte Mindesthöhe von 30 px für Zellen. Bei anderen Scan-Auflösungen
   bricht das. (klein)

### 4.2 Daten und Evaluation

7. **Ground Truth ist dünn**: 6 Seiten, 49 Bague-Zellen, 49 Alterszellen.
   Für belastbare Aussagen braucht es mindestens 20 bis 30 gelabelte Seiten,
   idealerweise aus mehreren Jahrgängen und von verschiedenen
   Beringer-Handschriften. Ein Labeling-Leitfaden fehlt. (mittel, vor allem
   Fleissarbeit, aber Grundlage für alles andere)
8. **Age, Jour/Mois, Heure** haben keine Ground Truth und werden nicht
   gemessen. (klein, sobald Punkt 7 läuft)
9. **Konfidenz-Kalibrierung**: Die Scores 100 / 90 / 60 / 50 aus dem Konsens
   sind Heuristiken. Ob ein Score von 90 tatsächlich 90 % Trefferquote
   bedeutet, wurde nie geprüft. Das ist direkt relevant dafür, welche Zellen
   der Mensch anschauen muss. (mittel, sauber wissenschaftlich auswertbar)
10. **Fehleranalyse pro Fehlertyp**: Verwechslungsmatrix für Ziffern,
    Analyse nach Handschrift, nach Zellposition, nach Tintenstärke. (mittel)

### 4.3 Modelle und Training

11. **YOLO-Ziffern-Detektor** wurde auf synthetischen Mehrziffern-Bildern
    trainiert (weisse Ziffern auf schwarzem 640er-Canvas). Das Preprocessing
    muss diese Verteilung exakt nachbilden, sonst findet YOLO nichts.
    Training auf echten Zell-Crops würde das entkoppeln. (gross)
12. **HTR-VT** wurde für die Altersspalte entfernt, weil es dort Rauschen
    liefert, und läuft nur noch auf der Artenspalte. Ein Feintuning auf
    die Handschriften der Beringer wurde nie versucht. (gross, braucht GPU)
13. **Trainingsprotokolle** für alle eigenen Modelle nachziehen, siehe 2.2.
    (klein)

### 4.4 Performance

14. **74 s pro Seite** auf CPU. TrOCR-Generierung dominiert. Optionen: GPU,
    kleineres OCR-Modell, Quantisierung, TrOCR nur auf unsicheren Zellen.
    Achtung: Timings nie parallel zu anderen Jobs messen, und mehr
    Torch-Threads sind bei `generate()` kontraproduktiv. (mittel)

### 4.5 Software und Betrieb

15. **CI/CD** ist auf GitLab zugeschnitten (`.gitlab-ci.yml` mit Docker-in-
    Docker, manuellem Promote und SSH-Deploy). Nach dem Umzug auf GitHub läuft
    nichts davon. Nach GitHub Actions portieren oder mindestens die Testsuite
    automatisieren. (mittel)
16. **Modellverteilung**: Die 400-MB-Checkpoints werden über Nextcloud
    verteilt, nicht über das Repo. Git LFS oder ein Release-Asset wäre
    reproduzierbarer. (klein)
17. **Abhängigkeiten**: Python 3.12, TensorFlow und PyTorch gleichzeitig,
    `opencv-python` und `opencv-contrib-python` müssen versionsgleich gepinnt
    sein (4.12.0.88), sonst bricht `HoughLinesP`. Die Installation ist fragil
    und nicht dokumentiert getestet auf Linux. (klein bis mittel)
18. **Streamlit-Editor**: Die Architektur (Session-State-Cache, Fragments,
    keine Reruns pro Edit) ist im README als "do not regress" markiert. Es
    gibt keine Ende-zu-Ende-UI-Tests jenseits von `test_app_editor.py`. (mittel)

---

## 5. Mögliche Zuschnitte für eine Bachelorarbeit

Drei Varianten, jede mit klarer Forschungsfrage, vorhandenen Daten und
einer Metrik, die man am Ende berichten kann.

**A. Numerische Messwerte aus Handschrift (Punkte 1, 7, 10, 11)**
Forschungsfrage: Lässt sich die Erkennung handschriftlicher Zahlenwerte in
Tabellenzellen durch ein sequenzbasiertes Modell gegenüber dem
Einzelziffern-Ensemble verbessern? Daten: `files-3/` plus neu gelabelte
Seiten. Metrik: Genauigkeit Aile/Poids gegen Baseline 58 % / 43 %.
Werkzeug: `tools/evaluate_corpus.py`, `tools/compare_digit_backends.py`.

**B. Konfidenz und Review-Aufwand (Punkte 9, 10, 8)**
Forschungsfrage: Wie gut sagen die Konsens-Scores tatsächliche Fehler voraus,
und wie viel manuelle Prüfarbeit lässt sich bei gegebener Ziel-Genauigkeit
einsparen? Daten: bestehender Korpus reicht für einen Anfang. Metrik:
Kalibrierungskurve, Precision/Recall der ⚠-Markierung, gesparte Zellen pro
Seite. Das ist die Frage, die für die Beringungszentrale praktisch am meisten
zählt, weil der Engpass die menschliche Nachkorrektur ist.

**C. Reproduzierbarkeit und Evaluationsinfrastruktur (Punkte 7, 13, 15, 16)**
Eher ingenieurwissenschaftlich: Labeling-Leitfaden, erweiterter Korpus,
dokumentierte Trainingspipelines, automatisierte Evaluation in CI. Ergebnis
ist ein Benchmark, an dem spätere Arbeiten messen können. Weniger
Modellforschung, dafür hoher Nutzwert.

Punkt 12 (HTR-VT-Feintuning) ist als eigenständige Arbeit möglich, braucht
aber GPU-Zugang und deutlich mehr gelabelte Artenzellen als vorhanden.

---

## 6. Themen mit Forschungscharakter, einfach erklärt

Die Punkte in Abschnitt 4 sind Verbesserungen am bestehenden System. Die
folgenden Themen gehen einen Schritt weiter: Sie stellen eine Frage, auf die
es noch keine klare Antwort gibt, und lassen sich mit den vorhandenen Daten
sauber untersuchen. Jedes Thema ist so beschrieben, dass man es ohne
Vorwissen über DocDig versteht.

### 6.1 Dezimalzahlen: Wo ist das Komma?

**Das Problem.** Beim Gewicht schreibt ein Beringer "10,5", ein anderer
"10.5", ein dritter "105" und meint dasselbe. Das Komma ist oft nur ein
winziger Punkt, manchmal ein Strich, manchmal fehlt es ganz. Heute lesen
alle Modelle nur Ziffern, und ein Abgleich im Konsens-Modul behandelt
"105" und "10.5" als gleich. Das ist eine Krücke: Bei "1050" gegen "10.50"
oder "105" gegen "1.05" versagt sie.

**Die Forschungsfrage.** Kann ein Modell die Position des Dezimaltrenners
in handschriftlichen Zahlen zuverlässig erkennen, obwohl der Trenner selbst
oft kaum sichtbar ist? Zwei Wege sind denkbar:

- *Den Trenner sehen*: einen eigenen Detektor für Punkt und Komma
  trainieren, der auch sehr kleine Markierungen findet. YOLO kennt bisher
  nur die Klassen 0 bis 9, eine elfte Klasse "Trenner" wäre der direkte Weg.
- *Den Trenner erschliessen*: die Zahl als Ganzes lesen und aus dem Kontext
  ableiten, wo das Komma sein muss. Ein Vogel wiegt 10,5 g, nicht 105 g,
  und die Spalte kennt ihren plausiblen Wertebereich. Hier lernt das Modell
  aus der Verteilung echter Werte, nicht aus dem Bild.

**Warum das interessant ist.** Die zweite Variante ist ein schönes Beispiel
dafür, wie Weltwissen (Vogelgewichte) ein Bildproblem löst. Man kann beide
Wege gegeneinander messen: Wie viele Poids-Fehler entstehen durch falsche
Kommaposition, wie viele durch falsche Ziffern? Daten: die Poids-Spalte in
`files-3/`, ergänzt um Zellen, die man gezielt nach Trenner-Schreibweise
labelt.

### 6.2 Ein grosses Netz statt einer Kette von Modulen

**Das Problem.** DocDig ist eine Kette aus rund fünfzehn Schritten:
Seite drehen, Tabelle finden, Spalten und Zeilen schneiden, Zellen
entrauschen, Zellen lesen, Ergebnisse abstimmen. Jeder Schritt hat eigene
Regeln und eigene Fehler, und ein Fehler früh in der Kette (etwa eine
verschobene Zeilengrenze) zieht sich durch alle späteren Schritte.

**Die Forschungsfrage.** Kann ein einziges Modell, das die ganze Seite als
Bild bekommt und direkt die Tabelle als Text ausgibt, mit der Modulkette
mithalten oder sie schlagen? Solche Modelle gibt es seit einigen Jahren
(Stichworte für die Literaturrecherche: "Donut", "Nougat", "end-to-end
document understanding", "image-to-sequence"). Sie lernen Struktur,
Entrauschen und Lesen gemeinsam, statt dass ein Mensch jeden Schritt
einzeln entwirft.

**Was dagegen spricht und die Arbeit spannend macht.** Solche Modelle
brauchen viele Trainingsbeispiele, und DocDig hat sechs gelabelte Seiten.
Die eigentliche Frage ist also: Wie viele gelabelte Seiten braucht ein
End-to-End-Modell, um die handgebaute Kette einzuholen? Lässt sich der
Bedarf durch synthetische Seiten (siehe 6.4) oder durch Vortraining auf
ähnlichen Formularen senken? Und was verliert man: Die Kette kann zeigen,
*welche* Zelle unsicher ist. Ein grosses Netz muss das erst lernen.

**Kleinere Variante.** Statt der ganzen Kette nur zwei benachbarte Schritte
verschmelzen, zum Beispiel Entrauschen und Lesen: Heute wird jede Zelle
erst von einem Denoiser-Netz gesäubert und dann von einem Lesemodell
gelesen, das die Säuberung nie zu sehen bekommt. Trainiert man beide
gemeinsam, kann das Lesemodell dem Denoiser sagen, welche Striche wichtig
sind. Das ist mit den vorhandenen 152 Zell-Fixtures machbar und gibt eine
klare Vorher-Nachher-Zahl.

### 6.3 Wie sicher ist sich das System wirklich?

**Das Problem.** Jede Zelle bekommt einen Score, etwa 100 wenn drei Modelle
einig sind, 60 wenn alle etwas anderes lesen. Der Mensch im Editor prüft
die Zellen mit niedrigem Score. Aber niemand hat je gemessen, ob eine Zelle
mit Score 90 auch in 90 % der Fälle richtig ist. Wenn Score 90 in
Wirklichkeit nur 70 % bedeutet, rutschen Fehler durch. Wenn er 99 %
bedeutet, prüft der Mensch umsonst.

**Die Forschungsfrage.** Wie kalibriert sind die Scores, und lässt sich mit
einem gelernten Vertrauensmodell (das etwa Handschriftqualität, Tintenmenge
und Uneinigkeit der Modelle als Eingabe nimmt) die Menge der zu prüfenden
Zellen bei gleicher Endgenauigkeit halbieren? Das ist die Frage mit dem
grössten praktischen Nutzen: Der Engpass bei der Vogelwarte ist die
Arbeitszeit der Menschen, nicht die Rechenzeit.

### 6.4 Handschrift synthetisch erzeugen

**Das Problem.** Alle Modelle sind datenhungrig, und gelabelte Beringungs-
protokolle sind selten. Es gibt aber grosse Sammlungen einzelner
handschriftlicher Ziffern und Buchstaben (DIDA, MNIST, IAM).

**Die Forschungsfrage.** Kann man aus solchen Bausteinen künstliche
Tabellenzellen oder ganze Seiten zusammensetzen, die echt genug aussehen,
damit ein darauf trainiertes Modell auf den echten Scans besser wird? Zu
klären wäre: Wie realistisch müssen Linien, Papierkörnung, Tintenverlauf
und Zeilenversatz sein? Ab wann schadet synthetisches Material mehr als es
nützt? Messbar über die bestehende Evaluation auf `files-3/`.

### 6.5 Vom Korrigieren lernen

**Das Problem.** Im Editor korrigiert ein Mensch jeden Tag Dutzende Zellen.
Diese Korrekturen sind perfekte Trainingsdaten, verschwinden aber heute in
der CSV.

**Die Forschungsfrage.** Wie baut man eine Schleife, in der die
Korrekturen automatisch gesammelt, geprüft und zum Nachtrainieren genutzt
werden, und wie schnell wird das System dadurch besser? Stichworte:
"active learning", "human in the loop". Interessant ist auch die
Anpassung an einzelne Handschriften: Ein Beringer füllt oft hunderte
Seiten aus. Nach zehn korrigierten Seiten sollte das System seine
Handschrift kennen.

### Einordnung

| Thema | Datenbedarf | Rechenbedarf | Neuheitsgrad | Praktischer Nutzen |
|---|---|---|---|---|
| 6.1 Dezimaltrenner | gering | gering | mittel | hoch (Poids 43 %) |
| 6.2 End-to-End-Netz | hoch | hoch (GPU) | hoch | unklar, riskant |
| 6.2 Kleine Variante | gering | mittel | mittel | mittel |
| 6.3 Kalibrierung | gering | gering | mittel | sehr hoch |
| 6.4 Synthetische Daten | gering | mittel | mittel | mittel bis hoch |
| 6.5 Korrekturschleife | wächst mit Nutzung | gering | mittel | hoch, langfristig |

Für eine Bachelorarbeit mit begrenzter Zeit und ohne garantierten
GPU-Zugang sind 6.1 und 6.3 die sichersten Wahlen. 6.2 ist das ambitionierteste
Thema und sollte als Vergleichsstudie angelegt werden, nicht als Versprechen,
die Kette zu ersetzen.

---

## 7. Einstieg in fünf Schritten

1. Repo klonen, Python 3.12, `poetry install` (oder `pip install -r requirements.txt`).
2. TATR-Checkpoints nach `config/` legen (Link in `link_to_tatr_models.txt`),
   HTR-VT-Checkpoints nach `config/htr_vt/` (bei Dominik nachfragen).
3. Schnelle Testsuite laufen lassen:
   ```bash
   python -m pytest tests/
   ```
4. Evaluation gegen den Korpus starten, das ist der Massstab für alles:
   ```bash
   python tools/evaluate_corpus.py
   ```
5. App starten und eine Seite aus `files-3/scan_1972_sample.pdf` durchlaufen lassen:
   ```bash
   streamlit run src/app.py
   ```

Lesereihenfolge: `ReadMe.md` Abschnitte "Pipeline overview", "Numeric
ensemble", "Cell dictionary contract", dann `tests/README.md`, dann
`src/pipeline/pipeline.py` und `src/modules/numeric_consensus.py`.
