# WB-ML-2021
Authors: Konrad Komisarczyk, Kacper Grzymkowski, Jakub Fołtyn

## Folder structure

* `data/` - processed data in common format, aquired by running scripts from `process_raw.R`
  * `Northwell_US.csv` removed from repository, due to non-disclosure agreement.
* `raw_data/` - data we found, and is present in the repository as-is (except for renaming).
  * `Yan_reply_First_last_wtime.csv` removed from repository, due to non-disclosure agreement. Data from the supplement to the article (Barish et al. 2021), DOI: 10.1038/s42256-020-00254-2
  * `Outcomerea_FR.xlsx` data from the supplement to the article (Dupuis et al. 2021), DOI: 10.1038/s42256-020-00252-4
  * `St_Antonius_NL.xlsx` data from the supplement to the article (Quanjel et al. 2021), DOI: 10.1038/s42256-020-00253-3
  * `Tongji_375_CN.xlsx` training data from the supplement to the article (Yan et al. 2020), DOI: 10.1038/s42256-020-0180-7
  * `Tongji_110_CN.xlsx` validation data from the supplement to the article (Yan et al. 2020), DOI: 10.1038/s42256-020-0180-7
  

## Northwell data (Barish et al.)
Insert file `Yan_reply_First_last_wtime.csv` into raw_data, run the appropriate function from `process_raw.R`. Output should be present in
`data/Northwell_US.csv`.