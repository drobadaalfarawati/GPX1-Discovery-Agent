# Data

Expected input file:

`GPX1_curated_for_RL.csv`

## Expected columns

* `PUBCHEM_CID`
* `PUBCHEM_EXT_DATASOURCE_SMILES`
* `PUBCHEM_ACTIVITY_OUTCOME`
* `label`
* `scaffold`

## Development dataset summary

* 9,304 compounds
* 155 actives
* 9,149 inactives
* 1.67% active rate
* 5,825 unique scaffolds
* No missing values
* No duplicate SMILES
* All SMILES RDKit-parsable

## Data availability

The raw CSV is intentionally not bundled with this repository.

Before public distribution of the curated dataset, the exact PubChem assay ID/source, activity definition, and curation procedure should be documented.

To run the modular notebooks locally, place the curated file at:

`data/GPX1_curated_for_RL.csv`

The interview demo notebook will also prompt for the CSV when run in Google Colab.
