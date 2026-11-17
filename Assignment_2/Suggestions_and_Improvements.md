# Feedback, Forbedringsforslag og Fejlrettelser til Assignment 2 (DermaMNIST)

> **Formål:** Dette dokument samler en udtømmende gennemgang af koden (`DermaMNIST_CNN_Assignment_2.ipynb`), rapporten (`Rapport.md`), noterne (`Forklaring.txt`) og resultatfilerne (`results/`) i forhold til SDU-kursets krav, opgavebeskrivelsen og Master Guiden.  
> **Bemærk:** Ingen eksisterende filer er blevet ændret eller overskrevet. Dette dokument fungerer som et selvstændigt tillæg med konkrete forbedringspunkter og svar på potentielle peer-review- og eksamensspørgsmål.

---

## Indholdsfortegnelse
1. [Overordnet Vurdering & Hovedkonklusion](#1-overordnet-vurdering--hovedkonklusion)
2. [Gennemgang af Kørsel og Testresultater](#2-gennemgang-af-kørsel-og-testresultater)
3. [Kritisk Gennemgang af Koden (`DermaMNIST_CNN_Assignment_2.ipynb`)](#3-kritisk-gennemgang-af-koden)
4. [Kritisk Gennemgang af Rapporten (`Rapport.md`)](#4-kritisk-gennemgang-af-rapporten)
5. [Kritisk Gennemgang af Noterne (`Forklaring.txt`)](#5-kritisk-gennemgang-af-noterne)
6. [Største Metodiske Svagheder & Konkrete Forbedringsforslag](#6-største-metodiske-svagheder--konkrete-forbedringsforslag)
7. [Forberedelse til 10-Minutters Fremlæggelse & Peer-Review Spørgsmål](#7-forberedelse-til-10-minutters-fremlæggelse--peer-review-spørgsmål)

---

## 1. Overordnet Vurdering & Hovedkonklusion

### Hvor godt ser det ud?
Kort sagt: **Ekstremt stærkt metodisk udgangspunkt.**  
I har ramt kernen i Jakob Holm Andersens mantra: *"Metode > Rå Resultater"*. 

**Hvad der fungerer exceptionelt godt:**
1. **Ingen data leakage:** Normalisering (mean/std) og klassevægte er strengt beregnet på `train`. Validering bruges til modelvalg/early stopping. Testsættet røres kun én eneste gang til sidst.
2. **Klasseubalance håndteres ærligt:** I falder ikke i fælden med at juble over 60,95 % accuracy. I viser direkte, at en dum majoritetsbaseline opnår 66,88 % accuracy, og forklarer *hvorfor* CNN'en alligevel er langt mere klinisk brugbar (balanced accuracy 59,72 % vs 14,29 %, macro-F1 0,4808 vs 0,1145).
3. **Akademisk ærlighed:** I citerer det nyeste studie af Abhishek et al. (2025) om læsionsoverlap mellem splits i DermaMNIST. Det viser et niveau langt over gennemsnittet for 5. semester.
4. **Fuldstændige sanity checks:** Både teoretisk start-tab $\ln(7) \approx 1,946$ og overfit-test på 28 billeder (100 % accuracy) er implementeret direkte efter pensum (Lecture 3, s. 70–71).

### Hvad mangler for at nå topkarakter (5/5 i FeedbackFruits)?
1. **Eksperimentelt design mangler en reel model-sammenligning:**  
   I sammenligner kun jeres `SimpleCNN` mod en triviel `MajorityBaseline` (altid gæt `nv`). For at score topkarakter i *Experimental Design* forventes typisk en sammenligning mellem:
   - `SimpleCNN (Scratch)` vs. `ResNet-18 (Pretrained/Transfer Learning)` ELLER
   - `SimpleCNN med klassevægte` vs. `SimpleCNN uden klassevægte` (ablation) ELLER
   - `SimpleCNN` vs. en flad lineær baseline (som i Opgave 1).
2. **Mangler forklarbarhed (XAI / Grad-CAM / t-SNE):**  
   Dette fremhæves i Master Guiden (afsnit 5.5) som "Discussion Gold".
3. **Klinisk dybde i Introduktion og Diskussion:**  
   Forklar hvad sygdommene reelt er (melanom = livstruende hudkræft, nevi = godartede modermærker) og diskuter **PPV vs. Prævalens** (Lecture 7, s. 59).

---

## 2. Gennemgang af Kørsel og Testresultater

Koden er blevet testet og verificeret mod `Assignment_2/data/dermamnist.npz` og checkpointet `results/best_model.pt`.

### Verificerede Metrikker på det officielle testsæt (N = 2.005)
| Metrik | Model (SimpleCNN) | Majoritetsbaseline | Kommentar |
|---|---|---|---|
| **Accuracy** | 0,6095 (1.222/2.005) | **0,6688** (1.341/2.005) | Baseline vinder på ren accuracy pga. 67 % `nv`! |
| **Balanced Accuracy** | **0,5972** | 0,1429 (1/7) | CNN finder mønstre i alle 7 klasser. |
| **Macro-F1** | **0,4808** | 0,1145 | CNN afbalancerer præcision og recall på tværs. |
| **Macro-AUC (OVR)** | **0,9132** | Ikke defineret | Viser stærk separations- og rangordningsevne. |
| **Cohen's Kappa** | **0,4157** | 0,0000 | "Moderat enighed" korrigeret for tilfældigt gæt. |
| **Vægtet test-tab** | 1,0874 | - | Numerisk stabilt. |

### Klasse-specifikke resultater der skal bemærkes
- **Melanom (`mel`):** Recall = **0,6771** (151 ud af 223 melanomer fanges), men Precision = **0,3186** (323 falske alarmer, hvoraf 274 er godartede `nv`).
  - *Klinisk tolkning:* Modellen er kalibreret mod høj sensitivitet over for kræft (godt i screening), men koster mange unødige biopsier (lav PPV).
- **Dermatofibrom (`df`):** Kun 23 testbilleder. Precision = **0,0965**, F1 = **0,1606**. Ekstremt følsom over for få fejl.
- **Vaskulære læsioner (`vasc`):** Kun 29 testbilleder. AUC = **0,9872**, Recall = **0,7931**. Karakteristisk rød farve gør den lettere at adskille trods få eksempler.

---

## 3. Kritisk Gennemgang af Koden

### 1. Robusthed ved automatisk download
- **Observation:** `DermaMNIST(root=str(DATA_DIR), download=True)` kaster en `RuntimeError` i `medmnist.dataset`, hvis `DATA_DIR` ikke allerede eksisterer fysisk på disken på forhånd, ELLER hvis Zenodo returnerer 403 Forbidden pga. bot-protection/proxy.
- **Løsning/Anbefaling:** Sørg altid for `DATA_DIR.mkdir(parents=True, exist_ok=True)` *før* kaldet til `DermaMNIST`. Hvis Zenodo blokerer automatisk download i scriptet, skal `dermamnist.npz` placeres manuelt i `Assignment_2/data/`.

### 2. Tabsaggregering i `run_epoch`
- **Observation:** I `run_epoch` defineres:
  ```python
  criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
  ...
  batch_loss_sum = criterion(logits, labels)
  batch_weight = class_weights[labels].sum()
  loss = batch_loss_sum / batch_weight
```
  Dette er matematisk korrekt for en vægtet gennemsnitlig batch-loss. Men det er præcis hvad PyTorchs indbyggede `reduction='mean'` gør som standard!
- **Konsekvens:** Koden fungerer 100 % korrekt, men unødigt kompliceret i batch-løkken.

### 3. `small_x` i Sanity Check (Celle 11)
- **Observation:** I celle 11 importeres `from PIL import Image` lokalt inde i cellen, og der konverteres frem og tilbage: `eval_transform(Image.fromarray(train_data.imgs[i]))`.
- **Anbefaling:** Flyt importen til Celle 2 (sammen med de andre imports) for renere kode-stil.

### 4. Flere CPU-tråde og GPU-understøttelse
- **Observation:** `torch.set_num_threads(min(4, torch.get_num_threads()))` og `num_workers=0`.
- **Vurdering:** Godt valg for stabilitet på Windows og CPU. Men hvis I vil køre eksperimenter på Google Colab med GPU (T4), kan I sætte `pin_memory=True` og `num_workers=2` for at fjerne I/O-flaskehalsen.

---

## 4. Kritisk Gennemgang af Rapporten (`Rapport.md`)

Rapporten følger opgavebeskrivelsens seks afsnit forbilledligt. Her er de vigtigste forbedringer til rapporten:

### Afsnit 1: Introduction
- **Hvad der mangler:**
  - En klarere klinisk motivation: Hvorfor er differentiering af hudlæsioner vigtig? Nævn malignitetsgraden af melanom (høj dødelighed ved sen opdagelse) over for basalcellekarcinom (lokalt invasiv, sjældent metastaserende) og godartede nevi.
  - Mangler bredere litteraturreferencer: Citer gerne **Esteva et al. (Nature, 2017)** (*Dermatologist-level classification of skin cancer with deep neural networks*) og **Raghu et al. (NeurIPS, 2019 — "Transfusion")** (*Understanding Transfer Learning for Medical Imaging*).

### Afsnit 2: Materials
- **Hvad der mangler:**
  - **Dermatoskopiske kendetegn (ABCD-reglen):** Asymmetri, Border irregularity, Color variation, Diameter. Forklar hvorfor $28 \times 28$ pixels er et massivt handicap: pigmentnetværk, forgrenede kar og fine sløringer forsvinder helt i pixeleringen.
  - **Billedartefakter:** Nævn de typiske artefakter i HAM10000: mørke vignet-hjørner, hår over læsionen, luftbobler i gel og gel-refleksioner.

### Afsnit 3: Methods
- **Hvad der mangler:**
  - **Hvorfor Global Average Pooling (GAP)?** Forklar at GAP (Lecture 4, s. 67) reducerer parametre i det fuldt forbundne lag fra $128 \times 7 \times 7 \times 7 = 43.911$ til blot $128 \times 7 = 896$ parametre. Det modvirker overfitting voldsomt.
  - **Hvorfor AdamW frem for SGD?** AdamW afkobler weight decay fra de adaptive gradientopdateringer (Lecture 3, s. 96, Loshchilov & Hutter, 2019).
  - **Domæne-specifik augmentation:** Nævn eksplicit, at vertikal og horisontal spejling er klinisk fuldt forsvarlige for dermatoskopi (en hudlæsion har ingen naturlig "op" eller "ned"), hvorimod vertikal vending ville være meningsløs for opretstående røntgenbilleder af brystkassen (PneumoniaMNIST).

### Afsnit 4: Results
- **Hvad der mangler:**
  - Tabellerne og figurerne er perfekte. Sørg for eksplicit at fremhæve forvekslingsmønstret i teksten: 274 `nv` forvekslet med `mel`.

### Afsnit 5: Discussion
- **Dette vil løfte diskussionen markant:**
  1. **Prævalens og PPV i klinisk screening (Lecture 7, s. 58–59):**
     $$\text{PPV} = \frac{\text{Sensitivitet} \times \text{Prævalens}}{\text{Sensitivitet} \times \text{Prævalens} + (1 - \text{Specificitet}) \times (1 - \text{Prævalens})}$$
     I DermaMNIST udgør melanom ca. 11 % af testdataene. I en almen praksis-screening er prævalensen af melanom ofte under 1 %. Selvom sensitiviteten (0,68) og specificiteten (0,82) forbliver uændrede, vil PPV styrtdykkede fra 32 % til under 4 %! Dette er en fremragende pointe, som viser dyb medicinsk AI-forståelse.
  2. **Hvorfor opnår MedMNIST v2 ResNet-18 benchmark 73,5 % accuracy, mens jeres CNN får 60,95 %?**
     - MedMNIST-benchmarket træner med **uvægtet** cross-entropy. En model der prioriterer `nv` højt får automatisk høj samlet accuracy på DermaMNIST.
     - ResNet-18 har residual-forbindelser ($F(x) + x$), 11 millioner parametre og dybere repræsentationer.
     - I valgte checkpoint ud fra **validerings-macro-F1**, hvilket bevidst ofrer accuracy for at redde minoritetsklasserne.

---

## 5. Kritisk Gennemgang af Noterne (`Forklaring.txt`)

`Forklaring.txt` er et forbilledligt grundigt dokument på over 800 linjer. Det dækker alle tænkelige detaljer.

**Små præciseringer/forslag til noterne:**
1. **Afsnit 11 (Klassevægte):** Det er værd at tilføje, at mens inverse frekvensvægte ($w_c = N / (K n_c)$) løfter recall på sjældne klasser, kan de også gøre gradienterne ustabile, hvis en ekstremt sjælden klasse (f.eks. `df` med kun 80 billeder) indeholder støj eller fejllabels. En "blødere" vægtning ($w_c \propto n_c^{-0.5}$) eller Focal Loss nævnes ofte i litteraturen som alternativ.
2. **Afsnit 24 (Litteratur):** Tilføj referencen til Raghu et al. (2019) vedrørende transfer learning i medicinsk billedbehandling.

---

## 6. Største Metodiske Svagheder & Konkrete Forbedringsforslag

Hvis I har tid og ønsker at gøre projektet helt skudsikkert til topkarakter, er her de tre vigtigste opgraderinger:

### Forslag 1: Tilføj en reel model-sammenligning (Vigtigst for karakteren!)
Lige nu har I kun ét neuralt netværk. Tilføj enten:
- **Ablation:** Træn samme `SimpleCNN` **uden klassevægte** (`class_weights=None`).
  - *Hypotese:* Uden klassevægte vil modellen opnå højere samlet accuracy (~68–72 %), men lavere macro-F1 (~0,35–0,40) og elendig recall på `df` og `mel`.  
  - *Effekt i rapporten:* I beviser eksperimentelt, at klassevægtene reelt er årsagen til jeres afvejning!
- **ELLER Baseline fra Assignment 1:** Kør en simpel logistisk regression / fladt lineært netværk på de udrullede pixels (2.352 værdier). Det viser empirisk, hvorfor konvolutioner og rumlig lokalitet er nødvendige.

### Forslag 2: Tilføj Grad-CAM eller Feature Embedding (t-SNE)
- Træk de 128 aktiveringer ud efter Global Average Pooling for alle testbilleder.
- Kør en 2D t-SNE eller PCA og farvelæg efter de sande klasser.
- *Hvad det viser:* I Lecture 1 (s. 88) lærte I, at t-SNE på rå pixels fejler, fordi baggrundsfarver dominerer. En t-SNE på CNN'ens lærte features viser, om netværket har formået at adskille sygdommene i et meningsfuldt latent rum.

---

## 7. Forberedelse til 10-Minutters Fremlæggelse & Peer-Review Spørgsmål

Til fremlæggelsen skal en anden gruppe forberede 2–3 spørgsmål til jer. Her er de 4 mest sandsynlige spørgsmål, I vil få, og hvordan I svarer skarpt:

### Spørgsmål 1: "Hvorfor har jeres CNN lavere accuracy (60,95 %) end en dum model, der bare gætter 'nv' hver gang (66,88 %)?"
> **Svar:**  
> "Fordi DermaMNIST er ekstremt ubalanceret — 67 % af billederne er melanocytære nevi (`nv`). Hvis en model udelukkende gætter `nv`, får den 66,88 % accuracy, men dens balanced accuracy er kun 14,3 % (1/7), og dens recall på melanom og hudkræft er 0 %. Vi har bevidst brugt klassevægtet tab og valideret efter macro-F1 for at tvinge modellen til at genkende alle syv sygdomme. Vores model opnår en balanced accuracy på knap 60 % og en macro-AUC på 0,913, hvilket beviser, at den adskiller sygdommene langt bedre end baselinen."

### Spørgsmål 2: "Hvorfor brugte I ikke en prætrænet ResNet-18 eller $224 \times 224$ billeder?"
> **Svar:**  
> "Vi valgte bevidst at træne en enkel, gennemskuelig CNN fra bunden på $28 \times 28$ for at have fuld kontrol over metodologien, undgå GPU-afhængighed og kunne redegøre for hvert eneste lag, parameterantal og tensorform (94.599 parametre og GAP mod overfitting). Desuden viser studiet af Raghu et al. (2019, 'Transfusion'), at overførsel af ImageNet-features til medicinske billeder ofte blot virker som god vægtskalering snarere end genbrug af højniveau-features. Vores formål var metodisk stringens frem for blind overdimensionering af modellen."

### Spørgsmål 3: "Jeres model finder 68 % af melanomerne, men precision er kun 32 %. Kan den bruges i klinikken?"
> **Svar:**  
> "I en klinisk triagesituation vil man ofte acceptere en lavere præcision for at sikre, at aggressive melanomer ikke overses (falske negativer er fatale). Dog er 32 % precision i et datasæt med 11 % melanom-prævalens en begrænsning. I en reel screeningspopulation, hvor prævalensen er under 1 %, ville PPV falde yderligere på grund af prævalens-paradokset (Lecture 7, s. 59). Modellen skal derfor ses som et benchmarkeksperiment og ikke som et færdigt diagnostisk værktøj."

### Spørgsmål 4: "I nævner en artikel fra 2025 om læsionsoverlap. Hvad betyder det for jeres testresultater?"
> **Svar:**  
> "Abhishek et al. (2025) påviste, at fordi HAM10000 indeholder flere fotos af samme læsion taget fra forskellige vinkler eller tidspunkter, er der lækage mellem de officielle train-, val- og test-splits i DermaMNIST. Det betyder, at vores testresultater (og alle andre publicerede DermaMNIST-benchmarks) kan have en optimistisk bias, fordi modellen kan have set den samme læsion under træning. Da NPZ-filen ikke indeholder patient-ID'er, har vi bevaret de officielle splits af hensyn til sammenlignelighed, men eksplicit diskuteret det som en validitetsbegrænsning i rapporten."
