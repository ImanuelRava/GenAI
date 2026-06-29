"""
SMILES Extractor
================
Fetch SMILES strings for molecules listed in an Excel file by querying the
NCI Cactus Chemical Identifier Resolver.

Usage (CLI):
    python SMILES.py
    python SMILES.py --input path/to/input.xlsx --output path/to/output.xlsx
    python SMILES.py --column "Ligand Name" --delay 1.0

Usage (as a module):
    from SMILES import get_smiles, run_smiles_extraction
    smiles = get_smiles("aspirin")
    run_smiles_extraction("input.xlsx", "output.xlsx")
"""

import argparse
import os
import socket
import ssl
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

# =============================================================================
# Default configuration — overridable via CLI flags or environment variables.
# Defaults are relative to THIS script's location so the script is portable.
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = _SCRIPT_DIR / "Nitrogen_ligands.xlsx"
DEFAULT_OUTPUT_FILE = _SCRIPT_DIR / "output_with_smiles.xlsx"
DEFAULT_NAME_COLUMN = "Ligand Name"

# API settings
REQUEST_DELAY = 0.5
TIMEOUT = 30
MAX_RETRIES = 3

# SECURITY: SSL verification is now ENABLED by default.
# Set LIGAND_INSECURE_TLS=1 in the environment ONLY if you are behind a
# corporate TLS-intercepting proxy that uses a CA Python cannot verify, and
# you have no other way to trust it. Disabling TLS verification exposes the
# script to man-in-the-middle attacks.
_INSECURE_TLS = os.environ.get("LIGAND_INSECURE_TLS", "0") == "1"


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context. Defaults to verification ON."""
    if _INSECURE_TLS:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context()


def get_smiles(name: str, retry: int = 0) -> Optional[str]:
    """Fetch SMILES for a molecule name from the NCI Cactus resolver.

    Args:
        name: Molecule name (or any identifier Cactus understands).
        retry: Internal retry counter (do not pass directly).

    Returns:
        SMILES string, or None if not found / fetch failed after retries.
    """
    url = f"http://cactus.nci.nih.gov/chemical/structure/{quote(name)}/smiles"

    try:
        request = Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0')

        response = urlopen(request, timeout=TIMEOUT, context=_build_ssl_context())
        return response.read().decode('utf-8').strip()

    except HTTPError as e:
        if e.code == 404:
            return None  # Not found — expected for unknown names
        if e.code in [429, 503] and retry < MAX_RETRIES:
            time.sleep(2 ** retry)
            return get_smiles(name, retry + 1)
        return None

    except (URLError, ConnectionResetError):
        if retry < MAX_RETRIES:
            time.sleep(2 ** retry)
            return get_smiles(name, retry + 1)
        return None

    except (ssl.SSLError, socket.timeout, UnicodeDecodeError, OSError) as e:
        # SSL errors, socket timeouts, decode errors, and other OS-level
        # I/O errors not caught by URLError above.
        print(f"  [Warning] unexpected error fetching {name}: {e}")
        return None


def run_smiles_extraction(
    input_file: str,
    output_file: str,
    name_column: str = DEFAULT_NAME_COLUMN,
    delay: float = REQUEST_DELAY,
) -> None:
    """Read Excel, fetch SMILES for each row, save enriched Excel.

    Args:
        input_file: Path to input .xlsx file.
        output_file: Path to write enriched .xlsx file.
        name_column: Column header containing molecule names.
        delay: Seconds to wait between API calls (rate-limit politeness).
    """
    print("=" * 60)
    print("SMILES EXTRACTOR")
    print("=" * 60)
    print(f"Input:   {input_file}")
    print(f"Output:  {output_file}")
    print(f"Column:  {name_column}")
    if _INSECURE_TLS:
        print("WARNING: TLS verification is DISABLED (LIGAND_INSECURE_TLS=1).")
    print()

    # Load Excel file
    try:
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} molecules")
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_file}")
        return
    except (OSError, ValueError, KeyError) as e:
        # OSError: permission denied / disk error. ValueError: not an Excel
        # file / corrupt file. KeyError: empty DataFrame.
        print(f"ERROR: {e}")
        return

    # Check column exists
    if name_column not in df.columns:
        print(f"ERROR: Column '{name_column}' not found")
        print(f"Available columns: {list(df.columns)}")
        return

    # Add SMILES column
    df['SMILES'] = ''

    # Process each molecule
    success = 0
    failed = 0

    for idx, row in df.iterrows():
        name = row[name_column]
        print(f"[{idx + 1}/{len(df)}] {str(name)[:40]}", end=" ... ")

        smiles = get_smiles(name)

        if smiles:
            df.at[idx, 'SMILES'] = smiles
            success += 1
            print("OK")
        else:
            failed += 1
            print("FAILED")

        time.sleep(delay)

    # Save results
    df.to_excel(output_file, index=False)

    # Summary
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total:     {len(df)}")
    print(f"Success:   {success}")
    print(f"Failed:    {failed}")
    print(f"Saved to:  {output_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SMILES strings for molecules in an Excel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i',
        default=str(DEFAULT_INPUT_FILE),
        help='Input Excel file path',
    )
    parser.add_argument(
        '--output', '-o',
        default=str(DEFAULT_OUTPUT_FILE),
        help='Output Excel file path',
    )
    parser.add_argument(
        '--column', '-c',
        default=DEFAULT_NAME_COLUMN,
        help='Name of the column containing molecule names',
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=REQUEST_DELAY,
        help='Seconds to wait between API calls (rate-limit politeness)',
    )
    return parser.parse_args()


def main():
    """Backwards-compatible entry point — parses CLI args and runs extraction."""
    args = _parse_args()
    run_smiles_extraction(
        input_file=args.input,
        output_file=args.output,
        name_column=args.column,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
