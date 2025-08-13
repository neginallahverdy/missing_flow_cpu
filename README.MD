# Weather_Forecast
Negin Allahverdy code

## Exporting covariates

The `script/export_covariates.py` helper allows you to write the covariate
matrix (including the ID column) from the CSV files used by
`CustomWeatherDataset`.

Example:

```bash
python script/export_covariates.py \
    --csv_file_data data/meteo_training_X.csv \
    --csv_file_label data/meteo_training_Y.csv \
    --mask_file data/mask_training.csv \
    --output_file training_covariates.csv
```
