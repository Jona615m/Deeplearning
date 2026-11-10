# Assignment 2 med DermaMNIST

Åbn `DermaMNIST_CNN_Assignment_2.ipynb`. Den indeholder en enkel CNN i PyTorch,
træning, valideringsbaseret modelvalg og endelig test på DermaMNIST. Den er
kørt færdig, så de målte resultater og figurer kan læses uden genkørsel.

- `Forklaring.txt`: grundig dansk gennemgang af kode, metodevalg og begreber.
- `Rapport.md`: rapport med opgavens seks afsnit, målte resultater og referencer.
- `requirements.txt`: pakker til at køre notebooken.
- `results/`: checkpoint, tabeller, forudsigelser, konfiguration og figurer.
- `data/`: officielt datasæt, som er downloadet lokalt og ignoreres af Git.

Den allerede eksisterende BloodMNIST-notebook er en separat fil. Brug
DermaMNIST-notebooken til denne løsning.

## Kørsel i VS Code eller Jupyter

Brug Python 3.11 eller nyere. Fra denne mappe installeres pakkerne i det
Python-miljø, der skal bruges som notebook-kernel:

```bash
python -m pip install -r requirements.txt
```

Vælg samme miljø som kernel i notebookens øverste højre hjørne. I dette
repository findes allerede et miljø i `../.venv`, som den gemte kørsel brugte.
Vælg **Restart Kernel and Run All** for at køre hele forsøget fra en ren kernel.
Notebooken finder både sine mapper fra repositoryets rod og fra `Assignment_2`.
Første download kræver internet; efterfølgende kørsel kan bruge lokal data.

CPU er tilstrækkelig. Den medfølgende kørsel tog cirka seks minutter efter
import og opsætning; tiden varierer mellem maskiner. CUDA anvendes automatisk,
hvis det findes. Resultater kan variere lidt med hardware og pakkeversioner.
De præcise versioner og seed fra den gemte kørsel står i `results/metrics.json`.

## Kørsel i Google Colab

Upload notebooken, og aktivér installationslinjen i den første kodecelle,
hvis MedMNIST mangler. Kør derefter alle celler i rækkefølge. Data hentes til
Colab-miljøet; `results` oprettes dér. Download resultatmappen, før Colab-
sessionen afsluttes, hvis en ny kørsel skal bevares. Ekstra upload af data
eller de lokale rapportfiler er ikke nødvendig for at køre selve notebooken.

## Det gennemførte forsøg

Modellen har tre konvolutioner og 94.599 parametre. Klassevægtet cross-entropy,
spejlinger og dropout bruges under træning. Kun to learning rates afprøves.
Epoch og learning rate vælges efter macro-F1 på valideringsdata, før testen ses.

| Testmål | CNN | Majoritetsbaseline |
| --- | ---: | ---: |
| Accuracy | 0,6095 | 0,6688 |
| Balanced accuracy | 0,5972 | 0,1429 |
| Macro-F1 | 0,4808 | 0,1145 |
| Macro-AUC OVR | 0,9132 | Ikke beregnet |

CNN'en har lavere samlet accuracy og bedre genkendelse på tværs af klasserne.
Rapporten forklarer afvejningen og de konkrete fejl. De officielle DermaMNIST-
splits har dokumenteret overlap af læsioner; dette er beskrevet med kilde i
rapport og forklaring.

En ny kørsel overskriver notebook-output og filer i `results`. Kopiér dem
først, hvis den medfølgende kørsel skal bevares. `Rapport.md` og `Forklaring.txt`
beskriver den gemte kørsel og opdateres ikke automatisk ved nye forsøg.

## Resultatfiler

| Fil | Indhold |
| --- | --- |
| `metrics.json` | Konfiguration, versioner og alle centrale målinger |
| `best_model.pt` | Valgt checkpoint og konfiguration |
| `tuning_results.csv` | Sammenligning på valideringsdata |
| `history_lr_*.csv` | Trænings- og valideringsmål for hver epoch |
| `class_distribution.csv` | Klassefordeling for de tre splits |
| `architecture.csv` | Lag, outputdimensioner og parameterantal |
| `per_class_metrics.csv` | Precision, recall, specificitet, F1 og AUC pr. klasse |
| `test_predictions.csv` | Alle testlabels, forudsigelser og softmax-scores |
| `*.png` | Syv figurer til rapport og fremlæggelse |

Checkpointet kan indlæses med `torch.load(..., weights_only=True)`. Det
indeholder `model_state_dict` og `config`. Brug notebookens `SimpleCNN`,
normalisering fra config og samme labelrækkefølge ved genbrug. Der gemmes
ikke optimizer-state til genoptagelse af træning.

Data og billedkilder: [MedMNIST](https://medmnist.com/) og
[HAM10000](https://doi.org/10.1038/sdata.2018.161), CC BY-NC 4.0.
