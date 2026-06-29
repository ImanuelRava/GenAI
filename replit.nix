# Replit NixOS package dependencies.
# Docs: https://docs.replit.com/programming-ide/nix-config-files
#
# These packages are installed at the system level (not via pip) so that
# Python libraries with C extensions can find their shared libraries.
#
# After changing this file, Replit will prompt you to "Install packages" —
# click it (or run `nix-install` in the shell) to apply the changes.

{ pkgs }: {
  deps = [
    # RDKit — chemistry library used by /api/molecules, /api/reactions.
    # Installing it at the system level avoids the pip install issues
    # that sometimes happen with the rdkit wheel on Replit.
    pkgs.rdkit

    # PDF processing — pypdf, pdfplumber, and PyMuPDF (fitz) need these
    # system libraries for rendering and image extraction.
    pkgs.poppler_utils        # pdf2image needs pdftoppm
    pkgs.libjpeg_turbo        # Pillow JPEG support
    pkgs.zlib                 # PNG compression

    # lxml — used by some PDF/Excel libraries for XML parsing.
    pkgs.libxml2
    pkgs.libxslt

    # OpenSSL — needed for HTTPS calls to Crossref/OpenAlex/LLM providers.
    pkgs.openssl

    # Fontconfig + freetype — needed by matplotlib (used in chemistry
    # molecule/reaction rendering) and by some PDF renderers.
    pkgs.fontconfig
    pkgs.freetype

    # Build tools — some pip packages (lxml, Pillow) may need to compile
    # from source if no pre-built wheel is available for the Replit arch.
    pkgs.gcc
    pkgs.gnumake
    pkgs.pkg-config
  ];
}
