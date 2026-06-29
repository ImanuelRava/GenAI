# NiCOBot Data Files — LFS Recovery Notice

## The Problem

The CSV files in this directory are **Git LFS pointer stubs**, not real data.
Each file is ~135 bytes and looks like this:

```
version https://git-lfs.github.com/spec/v1
oid sha256:1e33cb012da319875753691f7a974a36a63e1aea1c77a678567c4fab2d00c85d
size 36929841
```

This happened because the original repository used Git LFS to track these
files, but the LFS objects (the real ~37 MB of data behind each pointer)
were not included when the codebase was exported as a zip archive.

## Affected Files

All 12 CSV files in this directory are LFS pointer stubs:

| File | Real size |
|---|---|
| `data_all.csv` | ~? |
| `data_all_publish_info.csv` | ~? |
| `data_all_publish_info_with_descriptions.csv` | ~? |
| `data_all_publish_info_with_descriptions_JSON.csv` | ~? |
| `data_all_publish_info_with_descriptions_and_embeddings.csv` | ~37 MB |
| `data_all_publish_info_with_descriptions_json_and_embeddings.csv` | ~37 MB |
| `data_all_publish_info_with_iupac_names.csv` | ~? |
| `data_all_publish_info_with_iupac_names_C-O_strength.csv` | ~? |
| `base_all.csv` | ~? |
| `Interactive_Table.csv` | ~? |
| `Interactive_Table_Num.csv` | ~? |
| `schneider_50k_rxn_name.csv` | ~? |

The `.json` and `.xlsx` files in this directory are **real data** (not LFS
pointers) and work as-is.

## Impact

- `NiCOBotDatabase` (in `backend/modules/nicobot_database.py`) loads these
  files at startup via pandas. With LFS pointers in place, it will either:
  - Fail silently (returning empty DataFrames) for some loaders
  - Raise a `ParserError` when pandas tries to parse the LFS pointer text
    as CSV
- The `/api/database/*` endpoints will return empty or 500-error responses.
- The NiCOBot chatbot's RAG context retrieval (which calls
  `search_for_context()`) will return no results.

## How to Recover

### Option A — Pull from the original Git LFS server (preferred)

If you have access to the original Git repository:

```bash
# 1. Install Git LFS (one-time setup)
git lfs install

# 2. Clone the repo with LFS objects
git clone <original-repo-url>
cd GenAI

# 3. Explicitly pull LFS objects (in case clone skipped them)
git lfs pull

# 4. Verify the files are real data now (should be much larger than 135 bytes)
ls -la backend/nicobot_data/*.csv
```

### Option B — Restore from a backup copy

If you have a backup of the data files (e.g. on a shared drive, in
`backend/nicobot_data.zip`, or in another clone), simply copy them over
the LFS pointer stubs:

```bash
# Example: if you have a backup at /path/to/backup/
cp /path/to/backup/*.csv backend/nicobot_data/
cp /path/to/backup/*.xlsx backend/nicobot_data/
```

### Option C — Regenerate from source

If the data was originally derived from `Grand_Database_Fixed_Duplicated_Without_Ligand_T.xlsx`
(which IS present as real data), you can regenerate the derived CSVs by
re-running the data preparation scripts. (These scripts are not currently
in the repository — check the original data-engineering project.)

## Verifying Recovery

After recovery, every CSV file should be substantially larger than 200 bytes:

```bash
# All of these should show sizes > 1 KB (most should be > 1 MB):
ls -la backend/nicobot_data/*.csv

# Quick sanity check — should print "REAL DATA", not "LFS POINTER":
head -1 backend/nicobot_data/data_all.csv
```

## Preventing Future LFS-Pointer Commits

The root-level `.gitattributes` file now declares LFS tracking for
`*.csv`, `*.xlsx`, `*.json` in this directory. As long as Git LFS is
installed (`git lfs install`), any future large data file committed
to this directory will be properly LFS-tracked.

To verify LFS is set up correctly:

```bash
git lfs install        # one-time per machine
git lfs track          # show current LFS rules
git lfs ls-files       # list files currently tracked by LFS
```
