#!/usr/bin/env python3
"""
Example script for using the Center Fit Quality Assessment module.

This script demonstrates how to use the center_fit_assessment module to
evaluate the quality of polynomial center fitting in SPHERE/IFS data.
"""

from pathlib import Path

# Add the spherical package to path if needed
# sys.path.insert(0, '/path/to/spherical/src')
from spherical.pipeline.center_fit_assessment import assess_center_fit_quality


def main():
    """
    Example usage of the center fit quality assessment.
    """
    # Path to the converted directory containing center fitting outputs
    converted_dir = "/path/to/converted/data"
    
    # Check if directory exists
    if not Path(converted_dir).exists():
        print(f"Error: Directory {converted_dir} does not exist.")
        print("Please update the converted_dir path to point to your data.")
        return
    
    # Run the complete assessment
    print("Running center fit quality assessment...")
    
    try:
        assessment, summary = assess_center_fit_quality(
            converted_dir=converted_dir,
            create_dashboard=True
        )
        
        # Print key results
        print("\\n" + "="*60)
        print("CENTER FIT QUALITY ASSESSMENT RESULTS")
        print("="*60)
        
        # Overall statistics
        stats = summary['overall_statistics']
        print("\\nOverall Statistics:")
        print(f"  Mean R²: {stats['mean_r_squared']:.4f}")
        print(f"  Median R²: {stats['median_r_squared']:.4f}")
        print(f"  Mean RMSE: {stats['mean_rmse_pixels']:.4f} pixels")
        print(f"  Success Rate (R² > 0.9): {stats['success_rate_r_squared_90']:.1%}")
        print(f"  Success Rate (RMSE < 0.2): {stats['success_rate_rmse_02']:.1%}")
        
        # Wavelength analysis
        wl_stats = summary['wavelength_analysis']
        print("\\nWavelength Analysis:")
        print(f"  Problematic channels: {wl_stats['n_problematic_channels']}")
        if wl_stats['problematic_wavelengths_nm']:
            print(f"  Problematic wavelengths: {wl_stats['problematic_wavelengths_nm'][:5]}...")
        
        # Temporal stability
        temp_stats = summary['temporal_stability']
        print("\\nTemporal Stability:")
        print(f"  R² stability (X): {temp_stats['r_squared_stability_x']:.4f}")
        print(f"  R² stability (Y): {temp_stats['r_squared_stability_y']:.4f}")
        
        # Recommendations
        if summary['recommendations']:
            print("\\nRecommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print("\\nNo specific recommendations - fitting appears adequate.")
        
        # File outputs
        output_dir = Path(converted_dir) / 'center_assessment'
        print(f"\\nOutput files saved to: {output_dir}")
        print("  - centering_assessment_report.json")
        print("  - centering_assessment_dashboard.pdf")
        
        print("\\nAssessment completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the center fitting pipeline has been run and output files exist.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # Example of accessing detailed results
    print("\\n" + "-"*40)
    print("Example: Accessing detailed metrics")
    print("-"*40)
    
    # Access the assessment object for detailed analysis
    r_squared = assessment.metrics['r_squared']
    print(f"Shape of R² array: {r_squared.shape}")
    print(f"Best frame R² (X): {np.nanmax(r_squared[:, 0]):.4f}")
    print(f"Worst frame R² (X): {np.nanmin(r_squared[:, 0]):.4f}")
    
    # Wavelength-specific analysis
    wavelength_stats = assessment.wavelength_stats
    print(f"Wavelength with highest X residual std: "
          f"{assessment.data['wavelengths'][np.nanargmax(wavelength_stats['std_residual'][:, 0])]:.1f} nm")


if __name__ == "__main__":
    # Import numpy for the example
    import numpy as np
    main()
