# TTP-enhanced ransomware dataset

This folder holds the dataset used in RS-FEDRAD, already distributed over the ten federated clients exactly as in our experiments.

| File | Samples | Content |
|---|---:|---|
| `clients/client_01.csv.gz` ... `clients/client_10.csv.gz` | 3,015 each | 530 behavioural features and a `label` column |
| **Total** | **30,150** | 15,360 ransomware (50.9%), 14,790 benign (49.1%) |

Files are gzip-compressed CSV and are read directly by `pandas.read_csv`.

## Sources

The samples combine three ransomware collections and one benign collection (Table 2 of the paper).

| Reference | Source | Year | Type | Samples |
|---|---|---|---|---:|
| RSDT 1 | ISOT | 2020 | Ransomware | 668 |
| RSDT 2 | VirusShare | 2016 | Ransomware | 6,264 |
| RSDT 3 | SOREL-20M | 2020 | Ransomware | 8,444 |
| BDT | ISOT and Windows system files | 2019 to 2020 | Benign | 14,797 |

The behavioural traces of the ransomware and benign collections were also used by Gulmez et al. (2024). We extend them with ATT&CK technique annotations, as described below. The combined set was shuffled and split with stratification into ten equally sized clients, so every client holds the same class balance (about 51% ransomware).

## Feature layout

Each sample is one concatenated sequence of 530 positions recorded during sandboxed execution (Cuckoo Sandbox, with results cross-checked against ANY.RUN, ThreatGrid and Sysmon logs).

| Columns | Positions | Description | Distinct values | Empty positions |
|---|---|---|---:|---:|
| `ApiCall_0` ... `ApiCall_499` | 0 to 499 | Windows API calls in order of invocation | 272 | 32.9% |
| `Dll_0` ... `Dll_9` | 500 to 509 | loaded DLLs | 1,038 | 31.1% |
| `Mutex_0` ... `Mutex_9` | 510 to 519 | created or opened mutexes | 12,576 | 58.9% |
| `TTP_0` ... `TTP_9` | 520 to 529 | MITRE ATT&CK techniques mapped from the observed behaviour | 15 | 58.6% |
| `label` | | 1 = ransomware, 0 = benign | | |

Shorter traces leave trailing positions empty. DLL names are stored without the dot (`KERNEL32dll`).

## TTP enhancement

The `TTP_*` columns are the contribution of this dataset. Every sequence of API calls, DLL interactions and mutex operations was matched against ATT&CK technique profiles, as follows:

1. **Identification.** Behaviour is identified per feature type.
2. **Sequence alignment.** Related API, DLL and mutex events are aligned.
3. **Cosine similarity.** The concatenated behaviour vector is scored against known technique patterns.
4. **Rule matching.** The Jaccard index and a weighted confidence score are applied.
5. **Validation.** Mappings were refined with the ATT&CK Navigator and validated manually.

Thirteen techniques occur in the data:

| ID | Technique | ID | Technique |
|---|---|---|---|
| T1003 | OS Credential Dumping | T1083 | File and Directory Discovery |
| T1012 | Query Registry | T1098 | Account Manipulation |
| T1021 | Remote Services | T1105 | Ingress Tool Transfer |
| T1027 | Obfuscated Files or Information | T1106 | Native API |
| T1055 | Process Injection | T1134 | Access Token Manipulation |
| T1071 | Application Layer Protocol | T1073 | DLL Side-Loading |
| T1082 | System Information Discovery | | |

Two further values mark events without a technique match: `Unknown` for unresolved behaviour and `Benign` for behaviour mapped to legitimate activity.

## Preprocessing

Preprocessing happens locally on every client (`src/rsfedrad/encoding.py`):

1. **Imputation.** Empty positions are filled with the most frequent value of that position.
2. **Label encoding.** Each position is label-encoded.
3. **Scaling.** The codes are min-max scaled to [0, 1].
4. **Standardisation.** The scaled features are z-score standardised.
5. **Split.** Each client draws a stratified 80/20 train/test split (seed 42).

Because the vocabulary is fitted on the client itself, no client shares raw values or statistics with the server. The fitted encoder keeps every class, so an encoded value can always be traced back to the API call, DLL, mutex or technique it represents.

## Licence

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the RS-FEDRAD paper (see `CITATION.cff`) when you use it.
