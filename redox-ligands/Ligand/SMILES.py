import pandas as pd
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import ssl
import time

# =============================================================================
# CONFIGURATION - Edit these settings
# =============================================================================

INPUT_FILE = "Nitrogen_ligands.xlsx"      
OUTPUT_FILE = "output_with_smiles.xlsx"   
NAME_COLUMN = "Ligand Name"              

# API settings
REQUEST_DELAY = 0.5   
TIMEOUT = 30          
MAX_RETRIES = 3       

# =============================================================================
# SMILES EXTRACTION FUNCTION
# =============================================================================

def get_smiles(name, retry=0):
    url = f"http://cactus.nci.nih.gov/chemical/structure/{quote(name)}/smiles"
    
    try:
        request = Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0')
        
        # SSL context
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        response = urlopen(request, timeout=TIMEOUT, context=ctx)
        return response.read().decode('utf-8').strip()
        
    except HTTPError as e:
        if e.code == 404:
            return None  # Not found
        if e.code in [429, 503] and retry < MAX_RETRIES:
            time.sleep(2 ** retry)
            return get_smiles(name, retry + 1)
        return None
        
    except (URLError, ConnectionResetError) as e:
        if retry < MAX_RETRIES:
            time.sleep(2 ** retry)
            return get_smiles(name, retry + 1)
        return None
        
    except Exception:
        return None

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    print("=" * 60)
    print("SMILES EXTRACTOR")
    print("=" * 60)
    print(f"Input:   {INPUT_FILE}")
    print(f"Output:  {OUTPUT_FILE}")
    print(f"Column:  {NAME_COLUMN}")
    print()
    
    # Load Excel file
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"Loaded {len(df)} molecules")
    except FileNotFoundError:
        print(f"ERROR: File not found: {INPUT_FILE}")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Check column exists
    if NAME_COLUMN not in df.columns:
        print(f"ERROR: Column '{NAME_COLUMN}' not found")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Add SMILES column
    df['SMILES'] = ''
    
    # Process each molecule
    success = 0
    failed = 0
    
    for idx, row in df.iterrows():
        name = row[NAME_COLUMN]
        
        print(f"[{idx+1}/{len(df)}] {str(name)[:40]}", end=" ... ")
        
        smiles = get_smiles(name)
        
        if smiles:
            df.at[idx, 'SMILES'] = smiles
            success += 1
            print("OK")
        else:
            failed += 1
            print("FAILED")
        
        time.sleep(REQUEST_DELAY)
    
    # Save results
    df.to_excel(OUTPUT_FILE, index=False)
    
    # Summary
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total:     {len(df)}")
    print(f"Success:   {success}")
    print(f"Failed:    {failed}")
    print(f"Saved to:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
