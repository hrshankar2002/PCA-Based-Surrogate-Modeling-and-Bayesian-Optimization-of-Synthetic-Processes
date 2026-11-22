# BOSCH plasma-etching dataset

All files come from one Zenodo record:

> Sayyed, M. A., Seifert, T., Zieger, S., Schwarzenberg, S., Deshmukh, A., Haase, M., & Langer, J. *A Multi-Model Dataset for BOSCH Plasma-Etching: Optical Emission Spectra, Process Parameters, and Wafer Measurements for Data-Driven Plasma Modeling.* Zenodo, 2025. https://doi.org/10.5281/zenodo.17122442

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

`Real_Data_BOSCH_Experiment1.ipynb` downloads every file it needs from this record (section 1, "Data ingestion"). It checks each file against the MD5 published on Zenodo and downloads it again only if it is missing or the checksum does not match.

## Original files (unmodified)

| File | Where | Contents |
| --- | --- | --- |
| `Si_Oxide_etch_89_points.csv` | `data/bosch/` (in the repository) | One row per (wafer, measurement point); 88 wafers × 89 points. Columns used: `experiment_key` (date + wafer), `lot_number`, `wafer_number` (position in the lot, 1–10), `X`, `Y` (point position, µm from the wafer centre), `preox_thickness` (oxide before etching), `oxide_etch`, `si_etch` (etch depths). All thicknesses are in µm. |
| `Lot_status.xlsx` | `data/bosch/` (in the repository) | Date and conditioning of each lot. |
| `Process_data.nc`, `Dictionary_process.nc` | `data/bosch/raw/` (downloaded, git-ignored) | Machine log: up to 44 readings per wafer, 31 of them recorded for every wafer, 5 samples per second. Values are stored as codes into the dictionary file. |
| `Day_YYYY_MM_DD.nc` (10 files), `Dictionary_OES.nc` | `data/bosch/raw/` (downloaded, git-ignored) | Plasma optical emission spectra: 3,648 wavelengths, about 24 spectra per second, one group per wafer. About 7.4 GB. |

Conditioning labels (from the dataset Readme): the number is how many times the two-stage O₂ / O₂+SF₆ conditioning was repeated before the lot (1, 3 or 9); `C` = on the chuck, `Si` = on a blank Si wafer, `SiO2` = on a blank SiO₂ wafer. Each lot of up to 10 wafers was then etched without cleaning in between. The 3C - SiO2 run was repeated after a tool failure (lots 7, 8, 10); lot 7 lost its last 4 wafers. Wafer 7 of lot 1 and wafer 9 of lot 9 are not in the measurement file.

Per the dataset Readme, pre-etch oxide was measured at 15 points and post-etch interferometer failures were filled by inverse-distance interpolation; the 89-point measurements were taken in February 2025.

## Derived tables (written by the notebook)

Section 1 of the notebook rebuilds these from the original files on every run and writes them here for inspection; git ignores them (see `.gitignore`), so run the notebook (`make bosch`) to create them. No etch measurement is used to build the machine-reading or emission-line features.

- `lot_status.csv`: lot number, date and conditioning label, read from the first table of `Lot_status.xlsx`.
- `process_features.csv`: one row per wafer (96 wafers, including the 8 without etch measurements). Each machine reading is averaged over the etch steps (`|etch`, SF6 flow above 500) and over the protective steps (`|prot`, C4F8 flow above 250) while the plasma is on, plus the seconds spent in each (`etch_time_s`, `prot_time_s`, `plasma_time_s`). Only the 53 columns that differ between wafers are kept.
- `oes_lines.csv`: one row per wafer. The spectra are split into etch and protective steps by clustering them into the two plasma states. The wavelength axis is corrected by +1.21 nm, measured on nine known fluorine lines. Each feature is the height of an emission line above its local background, averaged over the step. Ten lines were looked for (Si 288, SiF 440, CO 483, C2 516, He 588, H 656, F 686, F 704, Ar 750, O 777 nm). Only those forming a clear peak (more than 5× the background ripple) are kept: `F686|etch`, `F704|etch`, `F686|prot`, `F704|prot`, `C2_516|prot`.

The spectrometer recording of wafer 2024-08-05_06 covers 399 s of plasma instead of about 575 s, but the machine log shows a normal run; its averages are kept.
