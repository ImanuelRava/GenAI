import os

TEXT_CHUNK_SIZE = 12000
MAX_OUTPUT_TOKENS = 16384
MAX_RETRIES = 2
RETRY_DELAY = 3
EXTRACTION_TEMPERATURE = 0.0  
EXTRACTION_SEED = 42           
CHUNK_OVERLAP = 1000           
CACHE_DIR = "/tmp/chemextract_cache"  

os.makedirs(CACHE_DIR, exist_ok=True)

RGROUP_SMILES_REFERENCE = """
Common R-group substituent SMILES (memorise these and use them exactly):
  Me / Methyl          = C
  Et / Ethyl           = CC
  n-Pr / n-Propyl      = CCC
  i-Pr / Isopropyl     = CC(C)
  n-Bu / n-Butyl       = CCCC
  t-Bu / tert-Butyl    = C(C)(C)C
  c-Pr / Cyclopropyl   = C1CC1
  c-Bu / Cyclobutyl    = C1CCC1
  c-Pent / Cyclopentyl = C1CCCC1
  c-Hex / Cyclohexyl   = C1CCCCC1
  Ph / Phenyl          = c1ccccc1
  2-Cl-Ph / 2-Chlorophenyl           = c1ccccc1Cl  (ortho)
  3-Cl-Ph / 3-Chlorophenyl           = c1cc(Cl)ccc1  (meta)
  4-Cl-Ph / 4-Chlorophenyl           = c1ccc(Cl)cc1  (para)
  4-MeO-Ph / 4-Methoxyphenyl        = COc1ccc(cc1)
  4-Me-Ph / 4-Methylphenyl (p-Tolyl)= Cc1ccc(cc1)
  4-F-Ph / 4-Fluorophenyl            = Fc1ccc(cc1)
  4-Br-Ph / 4-Bromophenyl            = Brc1ccc(cc1)
  4-CF3-Ph / 4-Trifluoromethylphenyl = FC(F)(F)c1ccc(cc1)
  4-NO2-Ph / 4-Nitrophenyl           = O=[N+]([O-])c1ccc(cc1)
  4-CN-Ph / 4-Cyanophenyl            = N#Cc1ccc(cc1)
  3,4-(MeO)2-Ph / 3,4-Dimethoxyphenyl = COc1ccc(OC)c1
  2-Naph / 2-Naphthyl   = c1ccc2ccccc2c1
  1-Naph / 1-Naphthyl   = c1ccc2cccc(c12)
  2-Thienyl / Thiophen-2-yl = c1cccs1
  2-Furyl / Furan-2-yl     = c1ccoc1
  3-Pyridyl / Pyridin-3-yl = c1ccnc1  (attached at C3)
  4-Pyridyl / Pyridin-4-yl = c1ccncc1 (attached at C4)
  2-Pyridyl / Pyridin-2-yl = n1ccccc1 (attached at N, use n1ccccc1 or c1ccncc1)
  Bn / Benzyl            = Cc1ccccc1
  Allyl                 = C=CC
  Vinyl                  = C=C
  Propargyl             = C#CC
  CH2CF3                = C(F)(F)F
  OMe / Methoxy          = OC
  OEt / Ethoxy           = OCC
  OAc / Acetoxy          = OC(=O)C
  OCF3 / Trifluoromethoxy = OC(F)(F)F
  Ac / Acetyl            = CC(=O)  (note: attachment through carbonyl carbon = C(=O)C)
  COOMe / Methyl ester   = C(=O)OC
  COOEt / Ethyl ester    = C(=O)OCC
  CN / Cyano             = C#N
  NO2 / Nitro            = [N+](=O)[O-]
  CF3 / Trifluoromethyl  = C(F)(F)F
  OCF3 / Trifluoromethoxy = OC(F)(F)F
  NMe2 / Dimethylamino    = CN(C)
  NEt2 / Diethylamino     = CCN(CC)
  SEM                   = COCC[Si](C)(C)C
  TMS                   = [Si](C)(C)C
  TBS / TBDMS           = [Si](C)(C)C(C)(C)C
  Bpin                  = B1OC(C)(C)C(O1)C(C)C  (pinacol boronate ester)
  St / Styryl            = C=Cc1ccccc1
  H / Hydrogen            = [H]  (use null or omit, do not put "H" as SMILES)
  F / Fluorine            = F
  Cl / Chlorine           = Cl
  Br / Bromine            = Br
  I / Iodine              = I
  OH / Hydroxy            = O  (attachment: O)
  SH / Thiol              = S
  NH2 / Amino             = N
  NHBoc / tert-Butoxycarbonylamino = C(=O)NC(C)(C)C
  NAc / Acetamido         = NC(=O)C
  N3 / Azido              = N=[N+]=[N-]
  CHO / Aldehyde          = C=O  (attachment through C)
  COOH / Carboxylic acid  = C(=O)O
  SO2Me / Methylsulfonyl  = S(=O)(=O)C
  SO2NH2 / Sulfonamide    = S(=O)(=O)N
  PPh3 / Triphenylphosphine = P(c1ccccc1)(c1ccccc1)c1ccccc1
  PMe3 / Trimethylphosphine = P(C)(C)P
  Ar (aryl placeholder)   = READ FROM CONTEXT: must be resolved to a specific aryl group.
"""