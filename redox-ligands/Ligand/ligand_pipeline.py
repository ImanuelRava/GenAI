"""
Ligand Processing Pipeline
==========================
An integrated script that combines:
1. SMILES fetching from molecule names (SMILES.py)
2. SDF file fetching from molecule names (SDF.py)
3. SDF to GJF conversion for Gaussian calculations (Coordinate.py)

Input: Nitrogen_ligands.xlsx (or any Excel file with 'Ligand Name' column)
Output: 
  - Updated Excel file with SMILES
  - SDF files for each ligand
  - GJF files for Gaussian calculations

Author: Integrated from SMILES.py, SDF.py, and Coordinate.py
"""

import os
import pandas as pd
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import time
import socket
import ssl
from typing import Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration settings for the pipeline."""
    
    # Input/Output directories
    INPUT_FILE = r"C:\Users\DELL\Documents\PhD\GenAI\redox-ligands\Ligand\Nitrogen_ligands.xlsx"
    OUTPUT_DIR = r"C:\Users\DELL\Documents\PhD\GenAI\redox-ligands\Ligand\ligand_output"
    SDF_DIR = os.path.join(OUTPUT_DIR, "sdf_files")
    GJF_DIR = os.path.join(OUTPUT_DIR, "gjf_files")
    UPDATED_EXCEL = os.path.join(OUTPUT_DIR, "Nitrogen_ligands_updated.xlsx")
    
    # Column names (adjust if your Excel has different column names)
    NAME_COLUMN = "Ligand Name"
    SMILES_COLUMN = "SMILES"
    ID_COLUMN = "Index"
    CAS_COLUMN = "CAS ID"
    
    # API settings
    API_BASE_URL = "http://cactus.nci.nih.gov/chemical/structure"
    REQUEST_DELAY = 1.0
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0 
    TIMEOUT = 60 
    
    # GJF template settings
    GJF_MEMORY = "12GB"
    GJF_NPROCSHARED = "18"
    GJF_METHOD = "#opt=loose def2SVP pop=nbo bp86 em=gd3bj"


# =============================================================================
# API Functions
# =============================================================================

def fetch_from_cactus(identifier: str, format_type: str = "smiles", retry_count: int = 0) -> Optional[str]:
    """
    Fetch chemical data from NCI Cactus Chemical Identifier Resolver.
    
    Includes retry logic with exponential backoff for handling connection errors.
    
    Args:
        identifier: Molecule name, SMILES, or other identifier
        format_type: Output format ('smiles', 'sdf', 'inchi', etc.)
        retry_count: Current retry attempt number (internal use)
    
    Returns:
        Fetched data as string, or None if failed after all retries
    """
    url = f"{Config.API_BASE_URL}/{quote(identifier)}/{format_type}"
    
    try:
        # Create request with headers
        request = Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        request.add_header('Accept', '*/*')
        
        # Create SSL context that's more tolerant
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        response = urlopen(request, timeout=Config.TIMEOUT, context=ssl_context)
        data = response.read().decode('utf-8')
        return data
        
    except HTTPError as e:
        if e.code == 404:
            logger.debug(f"Not found: {identifier}")
            return None
        elif e.code in [429, 503]:
            # Rate limited or service unavailable
            if retry_count < Config.MAX_RETRIES:
                wait_time = Config.RETRY_DELAY * (2 ** retry_count)
                logger.warning(f"Rate limited ({e.code}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                return fetch_from_cactus(identifier, format_type, retry_count + 1)
        print(f"  [HTTP Error {e.code}] {identifier}")
        return None
        
    except (URLError, ConnectionResetError, BrokenPipeError, socket.timeout) as e:
        # Connection errors
        if retry_count < Config.MAX_RETRIES:
            wait_time = Config.RETRY_DELAY * (2 ** retry_count)
            logger.warning(f"Connection error for {identifier}, retry {retry_count + 1}/{Config.MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)
            return fetch_from_cactus(identifier, format_type, retry_count + 1)
        print(f"  [Connection Error] {identifier}: {str(e)}")
        return None
        
    except Exception as e:
        error_msg = str(e)
        if "forcibly closed" in error_msg.lower() or "connection reset" in error_msg.lower():
            if retry_count < Config.MAX_RETRIES:
                wait_time = Config.RETRY_DELAY * (2 ** retry_count)
                logger.warning(f"Connection forcibly closed for {identifier}, retry {retry_count + 1}/{Config.MAX_RETRIES} in {wait_time}s...")
                time.sleep(wait_time)
                return fetch_from_cactus(identifier, format_type, retry_count + 1)
        print(f"  [Error] {identifier}: {error_msg}")
        return None


def fetch_smiles(molecule_name: str) -> Optional[str]:
    """Fetch SMILES string for a molecule."""
    return fetch_from_cactus(molecule_name, "smiles")


def fetch_sdf(molecule_name: str) -> Optional[str]:
    """Fetch SDF data for a molecule."""
    return fetch_from_cactus(molecule_name, "sdf")


# =============================================================================
# SDF Processing Functions
# =============================================================================

def parse_sdf(sdf_content: str) -> List[List[Tuple[str, str, str, str]]]:
    """
    Parse SDF content and extract atom coordinates.
    
    Args:
        sdf_content: String content of an SDF file
    
    Returns:
        List of molecules, each containing list of (atom_type, x, y, z) tuples
    """
    molecules = sdf_content.split('$$$$')
    coords_list = []
    
    for molecule in molecules:
        lines = molecule.strip().splitlines()
        if len(lines) < 4:
            continue
        
        atom_info = []
        for line in lines[4:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                atom_type = parts[3]
                
                if atom_type.isalpha():
                    atom_info.append((atom_type, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"))
                else:
                    break
            except ValueError:
                break
        
        if atom_info:
            coords_list.append(atom_info)
    
    return coords_list


def write_gjf(output_path: str, molecule_name: str, coordinates: List[Tuple[str, str, str, str]]) -> bool:
    """
    Write coordinates to GJF file format for Gaussian.
    
    Args:
        output_path: Directory to save the GJF file
        molecule_name: Name for the molecule (used in filename and content)
        coordinates: List of (atom_type, x, y, z) tuples
    
    Returns:
        True if successful, False otherwise
    """
    try:
        gjf_file_path = os.path.join(output_path, f"{molecule_name}.gjf")
        
        gjf_content = f"""%chk={molecule_name}.chk
%mem={Config.GJF_MEMORY}
%nprocshared={Config.GJF_NPROCSHARED}
{Config.GJF_METHOD}

{molecule_name}

0 1
"""
        
        for atom in coordinates:
            atom_type, x, y, z = atom
            gjf_content += f"{atom_type:<2} {x:>10} {y:>10} {z:>10}\n"
        
        gjf_content += "\n"
        
        with open(gjf_file_path, 'w') as f:
            f.write(gjf_content)
        
        return True
    except Exception as e:
        print(f"  [Error writing GJF] {molecule_name}: {e}")
        return False


# =============================================================================
# File Operations
# =============================================================================

def save_sdf_file(output_dir: str, molecule_id: str, sdf_data: str) -> bool:
    """Save SDF data to a file."""
    try:
        if sdf_data:
            filename = os.path.join(output_dir, f"{molecule_id}.sdf")
            with open(filename, 'w') as f:
                f.write(sdf_data)
            return True
        return False
    except Exception as e:
        print(f"  [Error saving SDF] {molecule_id}: {e}")
        return False


def ensure_directories():
    """Create output directories if they don't exist."""
    for directory in [Config.OUTPUT_DIR, Config.SDF_DIR, Config.GJF_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"[Setup] Created directory: {directory}")


# =============================================================================
# Pipeline Functions
# =============================================================================

def process_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process SMILES column - fetch missing SMILES from molecule names.
    
    Args:
        df: DataFrame with molecule data
    
    Returns:
        Updated DataFrame with SMILES column populated
    """
    print("\n" + "="*60)
    print("STEP 1: Processing SMILES")
    print("="*60)
    
    # Initialize SMILES column if not exists
    if Config.SMILES_COLUMN not in df.columns:
        df[Config.SMILES_COLUMN] = ""
    
    missing_smiles = df[df[Config.SMILES_COLUMN].isna() | (df[Config.SMILES_COLUMN] == "")]
    total = len(missing_smiles)
    
    if total == 0:
        print("  All molecules already have SMILES. Skipping...")
        return df
    
    print(f"  Found {total} molecules missing SMILES. Fetching...")
    
    success_count = 0
    for idx, row in missing_smiles.iterrows():
        molecule_name = row[Config.NAME_COLUMN]
        print(f"  [{idx+1}/{len(df)}] Fetching SMILES for: {molecule_name}")
        
        smiles = fetch_smiles(molecule_name)
        if smiles:
            df.at[idx, Config.SMILES_COLUMN] = smiles.strip()
            success_count += 1
            print(f"    -> Success: {smiles.strip()[:50]}...")
        else:
            print(f"    -> Failed to fetch SMILES")
        
        time.sleep(Config.REQUEST_DELAY)
    
    print(f"\n  SMILES fetch complete: {success_count}/{total} successful")
    return df


def process_sdf_files(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch SDF files for all molecules.
    
    Uses multiple identifier strategies with fallback:
    1. SMILES (most reliable - works even for complex names)
    2. Molecule Name (fallback)
    
    Note: CAS ID is NOT used because it often returns HTTP 500 errors.
    
    Args:
        df: DataFrame with molecule data
    
    Returns:
        DataFrame with SDF status column added
    """
    print("\n" + "="*60)
    print("STEP 2: Fetching SDF Files")
    print("="*60)
    print("  Strategy: Try SMILES first, then Name as fallback")
    print("  (CAS ID skipped - often causes server errors)")
    
    df['SDF_Status'] = ""
    df['SDF_Identifier_Used'] = ""
    
    success_count = 0
    total = len(df)
    
    for idx, row in df.iterrows():
        molecule_name = row[Config.NAME_COLUMN]
        molecule_id = row.get(Config.ID_COLUMN, idx)
        smiles = row.get(Config.SMILES_COLUMN, "")
        
        print(f"  [{idx+1}/{total}] {molecule_name[:45]} (ID: {molecule_id})")
        
        sdf_data = None
        identifier_used = None
        
        # Strategy 1: Try SMILES first (most reliable)
        if pd.notna(smiles) and smiles:
            sdf_data = fetch_sdf(smiles)
            if sdf_data:
                identifier_used = "SMILES"
                print(f"    -> Found via SMILES")
        
        # Strategy 2: Fallback to molecule name
        if not sdf_data:
            sdf_data = fetch_sdf(molecule_name)
            if sdf_data:
                identifier_used = "Name"
                print(f"    -> Found via Name")
        
        # Save results
        if sdf_data:
            if save_sdf_file(Config.SDF_DIR, str(molecule_id), sdf_data):
                df.at[idx, 'SDF_Status'] = "Saved"
                df.at[idx, 'SDF_Identifier_Used'] = identifier_used
                success_count += 1
                print(f"    -> SDF saved as {molecule_id}.sdf")
            else:
                df.at[idx, 'SDF_Status'] = "Save Error"
        else:
            df.at[idx, 'SDF_Status'] = "Not Found"
            df.at[idx, 'SDF_Identifier_Used'] = "Failed"
            print(f"    -> Failed to fetch SDF (both SMILES and Name tried)")
        
        time.sleep(Config.REQUEST_DELAY)
    
    # Summary statistics
    smiles_count = (df['SDF_Identifier_Used'] == 'SMILES').sum()
    name_count = (df['SDF_Identifier_Used'] == 'Name').sum()
    
    print(f"\n  SDF fetch complete: {success_count}/{total} successful")
    print(f"    - Found via SMILES: {smiles_count}")
    print(f"    - Found via Name:   {name_count}")
    return df


def process_gjf_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SDF files to GJF format.
    
    Args:
        df: DataFrame with molecule data
    
    Returns:
        DataFrame with GJF status column added
    """
    print("\n" + "="*60)
    print("STEP 3: Converting SDF to GJF")
    print("="*60)
    
    df['GJF_Status'] = ""
    
    success_count = 0
    total = len(df)
    
    for idx, row in df.iterrows():
        molecule_id = str(row.get(Config.ID_COLUMN, idx))
        sdf_file = os.path.join(Config.SDF_DIR, f"{molecule_id}.sdf")
        
        print(f"  [{idx+1}/{total}] Processing: {molecule_id}")
        
        if not os.path.exists(sdf_file):
            print(f"    -> SDF file not found, skipping")
            df.at[idx, 'GJF_Status'] = "No SDF"
            continue
        
        try:
            with open(sdf_file, 'r') as f:
                sdf_content = f.read()
            
            coords_list = parse_sdf(sdf_content)
            
            if coords_list:
                # Process each molecule in the SDF (usually just one)
                for i, coords in enumerate(coords_list):
                    name_suffix = f"_{i+1}" if len(coords_list) > 1 else ""
                    gjf_name = f"{molecule_id}{name_suffix}"
                    
                    if write_gjf(Config.GJF_DIR, gjf_name, coords):
                        df.at[idx, 'GJF_Status'] = "Converted"
                        success_count += 1
                        print(f"    -> GJF saved as {gjf_name}.gjf")
                    else:
                        df.at[idx, 'GJF_Status'] = "Write Error"
            else:
                df.at[idx, 'GJF_Status'] = "Parse Error"
                print(f"    -> Failed to parse SDF")
        except Exception as e:
            df.at[idx, 'GJF_Status'] = f"Error: {str(e)[:20]}"
            print(f"    -> Error: {e}")
    
    print(f"\n  GJF conversion complete: {success_count}/{total} successful")
    return df


def save_updated_excel(df: pd.DataFrame):
    """Save the updated DataFrame to Excel."""
    print("\n" + "="*60)
    print("Saving Updated Excel File")
    print("="*60)
    
    try:
        df.to_excel(Config.UPDATED_EXCEL, index=False)
        print(f"  Saved to: {Config.UPDATED_EXCEL}")
    except Exception as e:
        print(f"  [Error] Failed to save Excel: {e}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(steps: list = None):
    """
    Run the complete ligand processing pipeline.
    
    Args:
        steps: List of steps to run. Options: ['smiles', 'sdf', 'gjf']
               If None, runs all steps.
    """
    print("\n" + "="*60)
    print("LIGAND PROCESSING PIPELINE")
    print("="*60)
    
    # Setup
    ensure_directories()
    
    # Load data
    print(f"\nLoading data from: {Config.INPUT_FILE}")
    try:
        df = pd.read_excel(Config.INPUT_FILE)
        print(f"  Loaded {len(df)} molecules")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  [Error] Failed to load Excel file: {e}")
        return
    
    # Default to all steps
    if steps is None:
        steps = ['smiles', 'sdf', 'gjf']
    
    # Run pipeline steps
    if 'smiles' in steps:
        df = process_smiles(df)
    
    if 'sdf' in steps:
        df = process_sdf_files(df)
    
    if 'gjf' in steps:
        df = process_gjf_conversion(df)
    
    # Save results
    save_updated_excel(df)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"  Total molecules processed: {len(df)}")
    if 'SDF_Status' in df.columns:
        print(f"  SDF files saved: {(df['SDF_Status'] == 'Saved').sum()}")
    if 'GJF_Status' in df.columns:
        print(f"  GJF files created: {(df['GJF_Status'] == 'Converted').sum()}")
    print(f"  Output directory: {Config.OUTPUT_DIR}")
    print("="*60 + "\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ligand Processing Pipeline")
    parser.add_argument('--steps', nargs='+', choices=['smiles', 'sdf', 'gjf'],
                        default=None, help="Steps to run (default: all)")
    parser.add_argument('--input', type=str, help="Input Excel file path")
    parser.add_argument('--output', type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Override config if command line args provided
    if args.input:
        Config.INPUT_FILE = args.input
    if args.output:
        Config.OUTPUT_DIR = args.output
        Config.SDF_DIR = os.path.join(Config.OUTPUT_DIR, "sdf_files")
        Config.GJF_DIR = os.path.join(Config.OUTPUT_DIR, "gjf_files")
        Config.UPDATED_EXCEL = os.path.join(Config.OUTPUT_DIR, "Nitrogen_ligands_updated.xlsx")
    
    # Run pipeline
    run_pipeline(steps=args.steps)
