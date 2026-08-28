# Data

Expected input file: `GPX1_curated_for_RL.csv`

Expected columns:
- `PUBCHEM_CID`
- `PUBCHEM_EXT_DATASOURCE_SMILES`
- `PUBCHEM_ACTIVITY_OUTCOME`
- `label`
- `scaffold`

Development dataset summary:
- 9,304 compounds
- 155 actives
- 9,149 inactives
- 5,825 unique scaffolds
- all SMILES RDKit-parsable

The raw CSV is intentionally not bundled. Before public release, document the exact PubChem assay ID/source, activity definition, and curation procedure.
