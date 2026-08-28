import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


def parse_smiles(smiles):
    """Parse SMILES and return valid molecules plus a valid-row mask."""
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    valid_mask = np.array([mol is not None for mol in mols], dtype=bool)
    valid_mols = [mol for mol in mols if mol is not None]
    return valid_mols, valid_mask


def add_rdkit_descriptors(df, mols):
    """Add a compact interpretable descriptor panel."""
    out = df.copy()
    out["MolWt"] = [Descriptors.MolWt(m) for m in mols]
    out["LogP"] = [Descriptors.MolLogP(m) for m in mols]
    out["TPSA"] = [rdMolDescriptors.CalcTPSA(m) for m in mols]
    out["HBD"] = [Lipinski.NumHDonors(m) for m in mols]
    out["HBA"] = [Lipinski.NumHAcceptors(m) for m in mols]
    out["RotBonds"] = [Lipinski.NumRotatableBonds(m) for m in mols]
    out["RingCount"] = [Lipinski.RingCount(m) for m in mols]
    out["FracCSP3"] = [rdMolDescriptors.CalcFractionCSP3(m) for m in mols]
    return out


def morgan_fingerprints(mols, radius=2, fp_size=2048):
    """Generate binary Morgan fingerprints."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size
    )
    X = np.zeros((len(mols), fp_size), dtype=np.uint8)
    for i, mol in enumerate(mols):
        X[i] = np.asarray(
            generator.GetFingerprintAsNumPy(mol),
            dtype=np.uint8,
        )
    return X
