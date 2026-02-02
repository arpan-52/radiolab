"""
Command-line interface for RadioLab.

Provides CLI commands for all major features:
- radiolab-cube: Create spectral cubes
- radiolab-spectral: Fit spectral index/curvature
- radiolab-region: Extract spectra from regions
- radiolab-smooth: Smooth images to common resolution
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def parse_frequencies(freq_str: str) -> List[float]:
    """Parse comma-separated frequencies (supports MHz/GHz suffixes)."""
    frequencies = []
    for part in freq_str.split(','):
        part = part.strip()
        if part.lower().endswith('ghz'):
            freq = float(part[:-3]) * 1e9
        elif part.lower().endswith('mhz'):
            freq = float(part[:-3]) * 1e6
        elif part.lower().endswith('khz'):
            freq = float(part[:-3]) * 1e3
        else:
            freq = float(part)
        frequencies.append(freq)
    return frequencies


def make_cube_cli():
    """CLI for creating spectral cubes."""
    parser = argparse.ArgumentParser(
        prog='radiolab-cube',
        description='Create spectral cubes from multi-frequency images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect frequencies from FITS headers
  radiolab-cube "images/*.fits" -o cube.fits
  
  # With explicit frequencies (overrides auto-detection)
  radiolab-cube "images/*.fits" -f 1.4GHz,1.5GHz,1.6GHz -o cube.fits
  
  # With zoom to central 512x512 pixels
  radiolab-cube "images/*.fits" -o cube.fits --zoom 512
  
  # Smooth to specific beam (arcsec)
  radiolab-cube "images/*.fits" -o cube.fits --beam 10,10,0
        """
    )
    
    parser.add_argument(
        'images',
        help='Glob pattern for input FITS images (e.g., "images/*.fits")'
    )
    parser.add_argument(
        '-f', '--frequencies',
        help='Comma-separated frequencies (auto-detected from headers if not provided)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output FITS cube filename'
    )
    parser.add_argument(
        '--zoom',
        type=int,
        help='Crop to central NxN pixels'
    )
    parser.add_argument(
        '--beam',
        help='Target beam as bmaj,bmin,bpa in arcsec,arcsec,degrees'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    from .cube import make_cube
    from .beam import Beam
    
    # Parse frequencies
    frequencies = None
    if args.frequencies:
        frequencies = parse_frequencies(args.frequencies)
    
    # Parse beam
    target_beam = None
    if args.beam:
        parts = [float(x) for x in args.beam.split(',')]
        if len(parts) != 3:
            print("Error: --beam requires 3 values: bmaj,bmin,bpa", file=sys.stderr)
            sys.exit(1)
        # Convert arcsec to degrees
        target_beam = Beam(
            bmaj=parts[0] / 3600,
            bmin=parts[1] / 3600,
            bpa=parts[2]
        )
    
    try:
        cube, header, freqs = make_cube(
            args.images,
            frequencies=frequencies,
            zoom=args.zoom,
            target_beam=target_beam,
            output=args.output,
        )
        print(f"Created cube: {args.output}")
        print(f"Shape: {cube.shape}")
        print(f"Frequencies: {freqs/1e9} GHz")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def fit_spectral_cli():
    """CLI for spectral index/curvature fitting."""
    parser = argparse.ArgumentParser(
        prog='radiolab-spectral',
        description='Fit spectral index and curvature from images or cube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit spectral index only (order=1)
  radiolab-spectral cube.fits -o spectral --order 1
  
  # Fit with curvature (order=2)
  radiolab-spectral cube.fits -o spectral --order 2
  
  # From individual images (frequencies auto-detected)
  radiolab-spectral "images/*.fits" -o spectral
  
  # Custom RMS and sigma threshold
  radiolab-spectral cube.fits -o spectral --rms 1e-4 --sigma 5

Note: Frequencies and beams are automatically read from FITS headers. If images have
different beam sizes, they are smoothed to either --beam (if specified) or the
lowest resolution (largest beam) among the inputs.

Output files:
  {prefix}_spectral_index.fits      - Spectral index map
  {prefix}_spectral_index_error.fits - Error on spectral index
  {prefix}_curvature.fits           - Curvature map (if order>=2)
  {prefix}_curvature_error.fits     - Error on curvature
  {prefix}_chi2.fits                - Reduced chi-squared map
  {prefix}_mask.fits                - Fitted pixels mask
        """
    )
    
    parser.add_argument(
        'input',
        help='Input FITS cube or glob pattern for images'
    )
    parser.add_argument(
        '-f', '--frequencies',
        help='Comma-separated frequencies (auto-detected from headers if not provided)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output filename prefix'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=1,
        help='Polynomial order: 1=spectral index, 2=+curvature, etc. (default: 1)'
    )
    parser.add_argument(
        '--rms',
        type=float,
        help='RMS noise level (auto-computed if not provided)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=3.0,
        help='Sigma threshold for masking (default: 3.0)'
    )
    parser.add_argument(
        '--ref-freq',
        type=str,
        help='Reference frequency (e.g., 1.4GHz)'
    )
    parser.add_argument(
        '--beam',
        help='Target beam as bmaj,bmin,bpa in arcsec,arcsec,degrees'
    )
    
    args = parser.parse_args()
    
    from astropy.io import fits
    from .spectral import fit_spectral_index, save_spectral_fit
    from .beam import Beam
    from .io import freq_from_header
    
    # Determine if input is a cube or images
    input_path = Path(args.input)
    
    if input_path.suffix == '.fits' and '*' not in args.input:
        # Single FITS file - assume it's a cube
        print(f"Loading cube: {args.input}")
        with fits.open(args.input) as hdul:
            cube = hdul[0].data
            header = hdul[0].header
        
        # Extract frequencies from header
        nfreq = cube.shape[0]
        if args.frequencies:
            frequencies = np.array(parse_frequencies(args.frequencies))
        else:
            # Try to read from header
            frequencies = []
            for i in range(nfreq):
                key = f'FREQ{i:04d}'
                if key in header:
                    frequencies.append(header[key])
                else:
                    # Fallback to CRVAL3 + i*CDELT3
                    crval3 = header.get('CRVAL3', 1e9)
                    cdelt3 = header.get('CDELT3', 1e8)
                    frequencies.append(crval3 + i * cdelt3)
            frequencies = np.array(frequencies)
        
        print(f"Cube shape: {cube.shape}")
        print(f"Frequencies: {frequencies/1e9} GHz")
    else:
        # Glob pattern - load images (frequencies auto-detected)
        frequencies = None
        if args.frequencies:
            frequencies = np.array(parse_frequencies(args.frequencies))
        
        cube = args.input  # Pass pattern to fit_spectral_index
        header = None
    
    # Parse beam
    target_beam = None
    if args.beam:
        parts = [float(x) for x in args.beam.split(',')]
        target_beam = Beam(bmaj=parts[0]/3600, bmin=parts[1]/3600, bpa=parts[2])
    
    # Parse reference frequency
    ref_freq = None
    if args.ref_freq:
        ref_freq = parse_frequencies(args.ref_freq)[0]
    
    try:
        print(f"Fitting order-{args.order} polynomial...")
        result = fit_spectral_index(
            cube,
            frequencies,
            order=args.order,
            rms=args.rms,
            sigma=args.sigma,
            reference_freq=ref_freq,
            target_beam=target_beam,
        )
        
        print(f"Reference frequency: {result.reference_freq/1e9:.3f} GHz")
        print(f"RMS used: {result.rms_used:.3e}")
        print(f"Pixels fitted: {np.sum(result.mask)}")
        
        # Load header for WCS if we have one
        if isinstance(cube, str):
            import glob
            files = sorted(glob.glob(cube))
            if files:
                with fits.open(files[0]) as hdul:
                    header = hdul[0].header
        
        # Save results
        print(f"Saving results with prefix: {args.output}")
        created = save_spectral_fit(result, args.output, header)
        for f in created:
            print(f"  Created: {f}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def region_spectrum_cli():
    """CLI for region-based spectrum extraction."""
    parser = argparse.ArgumentParser(
        prog='radiolab-region',
        description='Extract spectra from regions (resolved or integrated)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Integrated spectrum from a DS9 region
  radiolab-region cube.fits source.reg -o spectrum.txt --mode integrated
  
  # Resolved spectral index within region
  radiolab-region cube.fits source.reg -o region_alpha --mode resolved --order 1
  
  # From images with frequencies
  radiolab-region "images/*.fits" source.reg -f 1.4GHz,1.5GHz -o spectrum.txt
        """
    )
    
    parser.add_argument(
        'input',
        help='Input FITS cube or glob pattern'
    )
    parser.add_argument(
        'region',
        help='DS9/CRTF region file'
    )
    parser.add_argument(
        '-f', '--frequencies',
        help='Comma-separated frequencies'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output filename or prefix'
    )
    parser.add_argument(
        '--mode',
        choices=['integrated', 'resolved'],
        default='integrated',
        help='Extraction mode (default: integrated)'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=1,
        help='Polynomial order for fitting (default: 1)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=3.0,
        help='Sigma threshold for resolved mode (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    from astropy.io import fits
    from .regions import fit_region_spectrum, load_region, region_to_mask
    from .spectral import save_spectral_fit
    
    # Load cube
    input_path = Path(args.input)
    
    if input_path.suffix == '.fits' and '*' not in args.input:
        with fits.open(args.input) as hdul:
            cube = hdul[0].data
            header = hdul[0].header
        
        nfreq = cube.shape[0]
        if args.frequencies:
            frequencies = np.array(parse_frequencies(args.frequencies))
        else:
            frequencies = []
            for i in range(nfreq):
                key = f'FREQ{i:04d}'
                if key in header:
                    frequencies.append(header[key])
                else:
                    crval3 = header.get('CRVAL3', 1e9)
                    cdelt3 = header.get('CDELT3', 1e8)
                    frequencies.append(crval3 + i * cdelt3)
            frequencies = np.array(frequencies)
    else:
        print("Error: For now, please provide a FITS cube", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.mode == 'integrated':
            coeffs, errors, chi2 = fit_region_spectrum(
                cube, frequencies,
                region_file=args.region,
                header=header,
                mode='integrated',
                order=args.order,
            )
            
            # Save as text file
            output_file = args.output if args.output.endswith('.txt') else f"{args.output}.txt"
            with open(output_file, 'w') as f:
                f.write(f"# RadioLab Region Spectrum Fit\n")
                f.write(f"# Region: {args.region}\n")
                f.write(f"# Order: {args.order}\n")
                f.write(f"# Chi2_reduced: {chi2:.4f}\n")
                f.write(f"#\n")
                f.write(f"# Coefficient  Value  Error\n")
                
                names = ['log_amplitude', 'spectral_index', 'curvature']
                names.extend([f'coef_{i}' for i in range(3, len(coeffs))])
                
                for i, (coef, err) in enumerate(zip(coeffs, errors)):
                    name = names[i] if i < len(names) else f'a{i}'
                    f.write(f"{name}  {coef:.6f}  {err:.6f}\n")
            
            print(f"Results saved to: {output_file}")
            print(f"Spectral index: {coeffs[1]:.3f} ± {errors[1]:.3f}")
            if args.order >= 2:
                print(f"Curvature: {coeffs[2]:.3f} ± {errors[2]:.3f}")
                
        else:  # resolved
            result = fit_region_spectrum(
                cube, frequencies,
                region_file=args.region,
                header=header,
                mode='resolved',
                order=args.order,
                sigma=args.sigma,
            )
            
            created = save_spectral_fit(result, args.output, header)
            print("Created files:")
            for f in created:
                print(f"  {f}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def smooth_cli():
    """CLI for smoothing images to common resolution."""
    parser = argparse.ArgumentParser(
        prog='radiolab-smooth',
        description='Smooth images to a common beam resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smooth single image to target beam
  radiolab-smooth image.fits -o smoothed.fits --beam 10,10,0
  
  # Smooth multiple images to lowest resolution (largest beam)
  radiolab-smooth "images/*.fits" -o smoothed/ --common
  
  # Smooth to specific beam
  radiolab-smooth "images/*.fits" -o smoothed/ --beam 15,12,45
        """
    )
    
    parser.add_argument(
        'input',
        help='Input FITS image(s) or glob pattern'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file or directory'
    )
    parser.add_argument(
        '--beam',
        help='Target beam as bmaj,bmin,bpa in arcsec,arcsec,degrees'
    )
    parser.add_argument(
        '--common',
        action='store_true',
        help='Auto-compute common beam (lowest resolution) from all inputs'
    )
    
    args = parser.parse_args()
    
    import glob
    from astropy.io import fits
    from .beam import Beam, get_beam, compute_common_beam, smooth_to_beam
    from .io import save_fits
    
    # Find input files
    if '*' in args.input:
        files = sorted(glob.glob(args.input))
    else:
        files = [args.input]
    
    if not files:
        print(f"Error: No files found matching {args.input}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(files)} file(s)")
    
    # Determine target beam
    if args.beam:
        parts = [float(x) for x in args.beam.split(',')]
        target_beam = Beam(bmaj=parts[0]/3600, bmin=parts[1]/3600, bpa=parts[2])
        print(f"Target beam: {target_beam}")
    elif args.common:
        # Load all beams
        beams = []
        for f in files:
            with fits.open(f) as hdul:
                beam = get_beam(hdul[0].header)
                if beam is None:
                    print(f"Warning: No beam in {f}", file=sys.stderr)
                else:
                    beams.append(beam)
                    print(f"  {Path(f).name}: {beam}")
        
        if not beams:
            print("Error: No beam information found", file=sys.stderr)
            sys.exit(1)
        
        target_beam = compute_common_beam(beams)
        print(f"Common beam (lowest resolution): {target_beam}")
    else:
        print("Error: Specify --beam or --common", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    if len(files) > 1:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process files
    for input_file in files:
        with fits.open(input_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        
        print(f"Smoothing: {input_file}")
        smoothed, new_header = smooth_to_beam(data, header, target_beam)
        
        if len(files) > 1:
            out_file = output_path / Path(input_file).name
        else:
            out_file = output_path
        
        save_fits(smoothed, new_header, out_file)
        print(f"  -> {out_file}")
    
    print("Done!")


def main():
    """Main entry point - show help for all commands."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                         RadioLab                              ║
║         Post-processing toolkit for radio astronomy           ║
╚═══════════════════════════════════════════════════════════════╝

Available commands:
  radiolab-cube      Create spectral cubes from multi-frequency images
  radiolab-spectral  Fit spectral index and curvature
  radiolab-region    Extract spectra from regions (resolved/integrated)
  radiolab-smooth    Smooth images to common resolution

Run any command with -h for detailed help, e.g.:
  radiolab-cube -h
  radiolab-spectral -h

Features:
  • Auto-detect frequencies and beams from FITS headers
  • Automatic resolution matching (smooth to lowest resolution)
  • Configurable polynomial order for spectral fitting
  • Full error propagation and χ² maps
  • DS9 region file support
""")


if __name__ == '__main__':
    main()
