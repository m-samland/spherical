"""
Center Fit Quality Assessment Module

This module provides comprehensive assessment of polynomial center fitting quality
in SPHERE/IFS data reduction. It implements statistical analysis, residual evaluation,
and visualization tools to validate and improve the center fitting methodology.

Classes
-------
CenterFitAssessment
    Main class for performing quality assessment of polynomial center fits.

Functions
---------
assess_center_fit_quality
    Convenience function for complete assessment workflow.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.gridspec import GridSpec

from spherical.pipeline.logging_utils import optional_logger


class CenterFitAssessment:
    """
    Comprehensive assessment of polynomial center fitting quality.
    
    This class provides tools to evaluate how well polynomial models describe
    wavelength-dependent stellar center positions, identify systematic failures,
    and recommend improvements to the fitting methodology.
    
    Parameters
    ----------
    converted_dir : str or Path
        Directory containing the center fitting output files.
    logger : logging.Logger, optional
        Logger instance for structured logging.
    
    Attributes
    ----------
    data : dict
        Loaded center data and metadata.
    metrics : dict
        Calculated fit quality metrics.
    wavelength_stats : dict
        Wavelength-dependent residual statistics.
    stability_metrics : dict
        Temporal stability assessment results.
    order_assessment : dict
        Polynomial order evaluation results.
    """
    
    def __init__(self, converted_dir: Union[str, Path], logger=None):
        self.converted_dir = Path(converted_dir)
        self.logger = logger
        self.data = {}
        self.metrics = {}
        self.wavelength_stats = {}
        self.stability_metrics = {}
        self.order_assessment = {}
        
    def load_center_data(self) -> Dict:
        """
        Load raw and fitted center data with metadata.
        
        Returns
        -------
        dict
            Analysis data structure containing all loaded arrays and metadata.
            
        Raises
        ------
        FileNotFoundError
            If required input files are missing.
        """
        required_files = [
            'image_centers.fits',
            'image_centers_fitted.fits', 
            'image_centers_fitted_robust.fits',
            'wavelengths.fits'
        ]
        
        # Check for required files
        for filename in required_files:
            filepath = self.converted_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Required file not found: {filepath}")
        
        # Load data
        centers_raw = fits.getdata(self.converted_dir / 'image_centers.fits')
        centers_fitted = fits.getdata(self.converted_dir / 'image_centers_fitted.fits')
        centers_robust = fits.getdata(self.converted_dir / 'image_centers_fitted_robust.fits')
        wavelengths = fits.getdata(self.converted_dir / 'wavelengths.fits')
        
        # Ensure consistent shapes
        wavelengths = np.array(wavelengths).flatten()
        
        self.data = {
            'wavelengths': wavelengths,
            'centers_raw': centers_raw,
            'centers_fitted': centers_fitted,
            'centers_robust': centers_robust,
            'n_wavelengths': len(wavelengths),
            'n_frames': centers_raw.shape[1],
            'residuals_fitted': centers_raw - centers_fitted,
            'residuals_robust': centers_raw - centers_robust
        }
        
        if self.logger:
            self.logger.info(f"Loaded center data: {self.data['n_wavelengths']} wavelengths, "
                           f"{self.data['n_frames']} frames")
        
        return self.data
    
    def calculate_basic_metrics(self, use_robust: bool = True) -> Dict:
        """
        Calculate fundamental fit quality metrics.
        
        Parameters
        ----------
        use_robust : bool, optional
            Whether to use robust fits (default) or first-pass fits.
            
        Returns
        -------
        dict
            Dictionary containing R², RMSE, and residual statistics.
        """
        if not self.data:
            self.load_center_data()
        
        # Select which fit to analyze
        centers_fitted = self.data['centers_robust'] if use_robust else self.data['centers_fitted']
        residuals = self.data['residuals_robust'] if use_robust else self.data['residuals_fitted']
        
        n_frames = self.data['n_frames']
        
        # Initialize metric arrays
        r_squared = np.full((n_frames, 2), np.nan)
        rmse = np.full((n_frames, 2), np.nan)
        chi_squared_red = np.full((n_frames, 2), np.nan)
        
        for frame in range(n_frames):
            for coord in range(2):  # x, y coordinates
                observed = self.data['centers_raw'][:, frame, coord]
                predicted = centers_fitted[:, frame, coord]
                
                # Handle NaN values
                valid_mask = np.isfinite(observed) & np.isfinite(predicted)
                n_valid = np.sum(valid_mask)
                
                if n_valid > 5:  # Minimum points for meaningful fit
                    obs_valid = observed[valid_mask]
                    pred_valid = predicted[valid_mask]
                    
                    # R² calculation
                    ss_res = np.sum((obs_valid - pred_valid)**2)
                    ss_tot = np.sum((obs_valid - np.mean(obs_valid))**2)
                    r_squared[frame, coord] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    
                    # RMSE calculation
                    rmse[frame, coord] = np.sqrt(ss_res / n_valid)
                    
                    # Reduced chi-square (assuming unit weights)
                    dof = n_valid - (3 if coord == 1 else 2) - 1  # polynomial order + 1
                    chi_squared_red[frame, coord] = (ss_res / dof) if dof > 0 else np.nan
        
        self.metrics = {
            'r_squared': r_squared,
            'rmse': rmse,
            'chi_squared_red': chi_squared_red,
            'residuals': residuals,
            'use_robust': use_robust
        }
        
        if self.logger:
            mean_r2 = np.nanmean(r_squared)
            mean_rmse = np.nanmean(rmse)
            self.logger.info(f"Basic metrics calculated - Mean R²: {mean_r2:.3f}, "
                           f"Mean RMSE: {mean_rmse:.3f} pixels")
        
        return self.metrics
    
    def analyze_wavelength_dependence(self) -> Tuple[Dict, List[int]]:
        """
        Analyze residual patterns across wavelength channels.
        
        Returns
        -------
        tuple
            (wavelength_stats, problematic_channels) where wavelength_stats
            contains per-wavelength statistics and problematic_channels is
            a list of wavelength indices requiring attention.
        """
        if not self.metrics:
            self.calculate_basic_metrics()
        
        residuals = self.metrics['residuals']
        wavelengths = self.data['wavelengths']
        
        # Per-wavelength statistics
        self.wavelength_stats = {
            'mean_residual': np.nanmean(residuals, axis=1),
            'std_residual': np.nanstd(residuals, axis=1),
            'median_residual': np.nanmedian(residuals, axis=1),
            'mad_residual': np.nanmedian(np.abs(residuals - np.nanmedian(residuals, axis=1, keepdims=True)), axis=1)
        }
        
        # Identify problematic wavelengths
        problematic_channels = []
        std_threshold = np.nanpercentile(self.wavelength_stats['std_residual'], 90)  # Top 10% worst
        bias_threshold = 3 * np.nanmedian(self.wavelength_stats['mad_residual'])  # 3-MAD rule
        
        for wl_idx in range(len(wavelengths)):
            # High residual variance
            if (self.wavelength_stats['std_residual'][wl_idx, 0] > std_threshold or 
                self.wavelength_stats['std_residual'][wl_idx, 1] > std_threshold):
                problematic_channels.append(wl_idx)
            
            # Systematic bias
            if (np.abs(self.wavelength_stats['mean_residual'][wl_idx, 0]) > bias_threshold or 
                np.abs(self.wavelength_stats['mean_residual'][wl_idx, 1]) > bias_threshold):
                problematic_channels.append(wl_idx)
        
        problematic_channels = list(set(problematic_channels))
        
        if self.logger:
            self.logger.info(f"Wavelength analysis: {len(problematic_channels)} problematic channels identified")
        
        return self.wavelength_stats, problematic_channels
    
    def analyze_temporal_stability(self) -> Dict:
        """
        Assess stability of fit quality across frames.
        
        Returns
        -------
        dict
            Dictionary containing temporal stability metrics.
        """
        if not self.metrics:
            self.calculate_basic_metrics()
        
        r_squared = self.metrics['r_squared']
        rmse = self.metrics['rmse']
        frame_numbers = np.arange(self.data['n_frames'])
        
        # Calculate temporal trends
        stability_metrics = {}
        
        for coord_idx, coord_name in enumerate(['x', 'y']):
            valid_frames = np.isfinite(r_squared[:, coord_idx])
            
            if np.sum(valid_frames) > 5:
                # Linear trend in R²
                trend_coeff = np.polyfit(frame_numbers[valid_frames], 
                                       r_squared[valid_frames, coord_idx], 1)[0]
                stability_metrics[f'r_squared_trend_{coord_name}'] = trend_coeff
                
                # R² stability (standard deviation)
                stability_metrics[f'r_squared_stability_{coord_name}'] = np.std(r_squared[valid_frames, coord_idx])
                
                # RMSE trend
                valid_rmse = np.isfinite(rmse[:, coord_idx])
                if np.sum(valid_rmse) > 5:
                    rmse_trend = np.polyfit(frame_numbers[valid_rmse], 
                                          rmse[valid_rmse, coord_idx], 1)[0]
                    stability_metrics[f'rmse_trend_{coord_name}'] = rmse_trend
            else:
                stability_metrics[f'r_squared_trend_{coord_name}'] = np.nan
                stability_metrics[f'r_squared_stability_{coord_name}'] = np.nan
                stability_metrics[f'rmse_trend_{coord_name}'] = np.nan
        
        self.stability_metrics = stability_metrics
        
        if self.logger:
            x_stability = stability_metrics.get('r_squared_stability_x', np.nan)
            y_stability = stability_metrics.get('r_squared_stability_y', np.nan)
            self.logger.info(f"Temporal stability - X: {x_stability:.4f}, Y: {y_stability:.4f}")
        
        return self.stability_metrics
    
    def evaluate_polynomial_orders(self, max_order: int = 6) -> Dict:
        """
        Evaluate optimal polynomial orders using information criteria.
        
        Parameters
        ----------
        max_order : int, optional
            Maximum polynomial order to test (default: 6).
            
        Returns
        -------
        dict
            Dictionary containing order assessment results for both coordinates.
        """
        if not self.data:
            self.load_center_data()
        
        wavelengths = self.data['wavelengths']
        centers_raw = self.data['centers_raw']
        
        order_assessment = {
            'x_coordinate': {'orders': [], 'aic': [], 'bic': [], 'rmse': []},
            'y_coordinate': {'orders': [], 'aic': [], 'bic': [], 'rmse': []}
        }
        
        # Test different polynomial orders
        for order in range(1, max_order + 1):
            for coord_idx, coord_name in enumerate(['x_coordinate', 'y_coordinate']):
                aic_scores = []
                bic_scores = []
                rmse_scores = []
                
                for frame in range(self.data['n_frames']):
                    centers = centers_raw[:, frame, coord_idx]
                    valid_mask = np.isfinite(centers)
                    
                    if np.sum(valid_mask) > order + 1:
                        try:
                            # Fit polynomial
                            coeffs = np.polyfit(wavelengths[valid_mask], centers[valid_mask], order)
                            predicted = np.polyval(coeffs, wavelengths[valid_mask])
                            
                            # Calculate metrics
                            n = np.sum(valid_mask)
                            k = order + 1  # number of parameters
                            mse = np.mean((centers[valid_mask] - predicted)**2)
                            
                            if mse > 0:
                                # Information criteria
                                log_likelihood = -n/2 * np.log(2*np.pi*mse) - n/2
                                aic = -2 * log_likelihood + 2 * k
                                bic = -2 * log_likelihood + k * np.log(n)
                                rmse = np.sqrt(mse)
                                
                                aic_scores.append(aic)
                                bic_scores.append(bic)
                                rmse_scores.append(rmse)
                        except (np.linalg.LinAlgError, RuntimeWarning):
                            continue
                
                if aic_scores:
                    order_assessment[coord_name]['orders'].append(order)
                    order_assessment[coord_name]['aic'].append(np.mean(aic_scores))
                    order_assessment[coord_name]['bic'].append(np.mean(bic_scores))
                    order_assessment[coord_name]['rmse'].append(np.mean(rmse_scores))
        
        self.order_assessment = order_assessment
        
        if self.logger:
            # Find optimal orders
            for coord_name in ['x_coordinate', 'y_coordinate']:
                if order_assessment[coord_name]['bic']:
                    optimal_idx = np.argmin(order_assessment[coord_name]['bic'])
                    optimal_order = order_assessment[coord_name]['orders'][optimal_idx]
                    self.logger.info(f"Optimal order for {coord_name}: {optimal_order}")
        
        return self.order_assessment
    
    def create_assessment_dashboard(self, output_path: Optional[Union[str, Path]] = None):
        """
        Create comprehensive visualization dashboard.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Path to save the dashboard figure. If None, figure is not saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The created dashboard figure.
        """
        # Ensure all analyses are complete
        if not self.metrics:
            self.calculate_basic_metrics()
        if not self.wavelength_stats:
            self.analyze_wavelength_dependence()
        if not self.stability_metrics:
            self.analyze_temporal_stability()
        if not self.order_assessment:
            self.evaluate_polynomial_orders()
        
        # Set up the dashboard layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Fit Quality Overview (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        r_sq = self.metrics['r_squared']
        ax1.hist(r_sq[:, 0].flatten(), bins=20, alpha=0.7, label='X-coordinate', density=True)
        ax1.hist(r_sq[:, 1].flatten(), bins=20, alpha=0.7, label='Y-coordinate', density=True)
        ax1.set_xlabel('R² Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Fit Quality Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RMSE Distribution (Top Center-Left)
        ax2 = fig.add_subplot(gs[0, 1])
        rmse = self.metrics['rmse']
        ax2.hist(rmse[:, 0].flatten(), bins=20, alpha=0.7, label='X-coordinate', density=True)
        ax2.hist(rmse[:, 1].flatten(), bins=20, alpha=0.7, label='Y-coordinate', density=True)
        ax2.set_xlabel('RMSE (pixels)')
        ax2.set_ylabel('Density')
        ax2.set_title('RMSE Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Wavelength-Dependent Residuals (Top Center-Right)
        ax3 = fig.add_subplot(gs[0, 2])
        wavelengths = self.data['wavelengths']
        mean_res = self.wavelength_stats['mean_residual']
        std_res = self.wavelength_stats['std_residual']
        ax3.errorbar(wavelengths, mean_res[:, 0], yerr=std_res[:, 0], 
                    fmt='o-', label='X-coordinate', capsize=3, markersize=3)
        ax3.errorbar(wavelengths, mean_res[:, 1], yerr=std_res[:, 1], 
                    fmt='s-', label='Y-coordinate', capsize=3, markersize=3)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Mean Residual ± Std (pixels)')
        ax3.set_title('Wavelength-Dependent Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Temporal Stability (Top Right)
        ax4 = fig.add_subplot(gs[0, 3])
        frame_numbers = np.arange(self.data['n_frames'])
        ax4.plot(frame_numbers, r_sq[:, 0], 'o-', alpha=0.7, markersize=2, label='X R²')
        ax4.plot(frame_numbers, r_sq[:, 1], 's-', alpha=0.7, markersize=2, label='Y R²')
        ax4.set_xlabel('Frame Number')
        ax4.set_ylabel('R² Score')
        ax4.set_title('Temporal Stability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Polynomial Order Assessment (Second Row Left)
        ax5 = fig.add_subplot(gs[1, 0])
        for coord_name, coord_label in [('x_coordinate', 'X'), ('y_coordinate', 'Y')]:
            if self.order_assessment[coord_name]['orders']:
                ax5.plot(self.order_assessment[coord_name]['orders'], 
                        self.order_assessment[coord_name]['bic'], 
                        'o-', label=f'{coord_label}-coordinate')
        ax5.set_xlabel('Polynomial Order')
        ax5.set_ylabel('BIC Score')
        ax5.set_title('Polynomial Order Selection (BIC)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Sample Residual Patterns (Second Row Center-Left)
        ax6 = fig.add_subplot(gs[1, 1])
        residuals = self.metrics['residuals']
        # Show residuals for up to 10 frames
        n_sample = min(10, self.data['n_frames'])
        sample_frames = np.linspace(0, self.data['n_frames']-1, n_sample, dtype=int)
        
        for frame in sample_frames:
            residuals_x = residuals[:, frame, 0]
            valid_mask = np.isfinite(residuals_x)
            if np.sum(valid_mask) > 0:
                ax6.plot(wavelengths[valid_mask], residuals_x[valid_mask], 
                        'o-', alpha=0.6, markersize=3)
        ax6.set_xlabel('Wavelength (nm)')
        ax6.set_ylabel('Residual (pixels)')
        ax6.set_title('Sample Residual Patterns (X-coord)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Quality Score Evolution (Second Row Center-Right)
        ax7 = fig.add_subplot(gs[1, 2])
        quality_score = np.nanmean(r_sq, axis=1)
        ax7.plot(frame_numbers, quality_score, 'o-', markersize=3)
        ax7.set_xlabel('Frame Number')
        ax7.set_ylabel('Average R² Score')
        ax7.set_title('Overall Quality Evolution')
        ax7.grid(True, alpha=0.3)
        
        # 8. Performance Summary (Second Row Right)
        ax8 = fig.add_subplot(gs[1, 3])
        r_sq_flat = r_sq.flatten()
        r_sq_valid = r_sq_flat[np.isfinite(r_sq_flat)]
        
        performance_summary = {
            'Excellent\n(R²>0.95)': np.sum(r_sq_valid > 0.95),
            'Good\n(0.9-0.95)': np.sum((r_sq_valid > 0.9) & (r_sq_valid <= 0.95)),
            'Fair\n(0.8-0.9)': np.sum((r_sq_valid > 0.8) & (r_sq_valid <= 0.9)),
            'Poor\n(R²≤0.8)': np.sum(r_sq_valid <= 0.8)
        }
        
        categories = list(performance_summary.keys())
        values = list(performance_summary.values())
        colors = ['green', 'lightgreen', 'orange', 'red']
        
        # Create pie chart with better text formatting
        pie_result = ax8.pie(values, labels=categories, colors=colors, 
                           autopct='%1.1f%%', startangle=90,
                           textprops={'fontsize': 8, 'weight': 'bold'})
        # Handle variable return from pie() - we don't use the return values
        _ = pie_result
        ax8.set_title('Model Performance Summary', fontsize=10, weight='bold')
        
        # 9-12: Detailed Individual Frame Analysis (Third Row)
        sample_frames_detailed = [0, self.data['n_frames']//3, 2*self.data['n_frames']//3, -1]
        for i, frame_idx in enumerate(sample_frames_detailed):
            ax = fig.add_subplot(gs[2, i])
            
            # Plot raw data and fit
            centers_raw_x = self.data['centers_raw'][:, frame_idx, 0]
            centers_fitted_x = (self.data['centers_robust'] if self.metrics['use_robust'] 
                              else self.data['centers_fitted'])[:, frame_idx, 0]
            valid_mask = np.isfinite(centers_raw_x) & np.isfinite(centers_fitted_x)
            
            if np.sum(valid_mask) > 0:
                ax.plot(wavelengths[valid_mask], centers_raw_x[valid_mask], 'o', 
                       label='Raw data', markersize=4)
                ax.plot(wavelengths[valid_mask], centers_fitted_x[valid_mask], '-', 
                       label='Polynomial fit', linewidth=2)
                
                # Add residuals info
                r2_score = r_sq[frame_idx, 0] if frame_idx < len(r_sq) else np.nan
                ax.set_title(f'Frame {frame_idx}: R²={r2_score:.3f}')
            else:
                ax.set_title(f'Frame {frame_idx}: No valid data')
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('X Center (pixels)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Center Fit Quality Assessment Dashboard', fontsize=16, y=0.98)
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Dashboard saved to {output_path}")
        
        return fig
    
    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary statistics and recommendations.
        
        Returns
        -------
        dict
            Dictionary containing summary statistics and recommendations.
        """
        # Ensure all analyses are complete
        if not self.metrics:
            self.calculate_basic_metrics()
        if not self.wavelength_stats:
            _, problematic_channels = self.analyze_wavelength_dependence()
        else:
            # Get problematic channels
            std_threshold = np.nanpercentile(self.wavelength_stats['std_residual'], 90)
            bias_threshold = 3 * np.nanmedian(self.wavelength_stats['mad_residual'])
            problematic_channels = []
            for wl_idx in range(len(self.data['wavelengths'])):
                if (self.wavelength_stats['std_residual'][wl_idx, 0] > std_threshold or 
                    self.wavelength_stats['std_residual'][wl_idx, 1] > std_threshold or
                    np.abs(self.wavelength_stats['mean_residual'][wl_idx, 0]) > bias_threshold or 
                    np.abs(self.wavelength_stats['mean_residual'][wl_idx, 1]) > bias_threshold):
                    problematic_channels.append(wl_idx)
        
        if not self.stability_metrics:
            self.analyze_temporal_stability()
        if not self.order_assessment:
            self.evaluate_polynomial_orders()
        
        # Calculate summary statistics
        r_sq = self.metrics['r_squared']
        rmse = self.metrics['rmse']
        r_sq_valid = r_sq[np.isfinite(r_sq)]
        rmse_valid = rmse[np.isfinite(rmse)]
        
        summary = {
            'overall_statistics': {
                'mean_r_squared': float(np.nanmean(r_sq_valid)),
                'median_r_squared': float(np.nanmedian(r_sq_valid)),
                'std_r_squared': float(np.nanstd(r_sq_valid)),
                'mean_rmse_pixels': float(np.nanmean(rmse_valid)),
                'median_rmse_pixels': float(np.nanmedian(rmse_valid)),
                'success_rate_r_squared_90': float(np.sum(r_sq_valid > 0.9) / len(r_sq_valid)),
                'success_rate_rmse_02': float(np.sum(rmse_valid < 0.2) / len(rmse_valid))
            },
            'wavelength_analysis': {
                'n_problematic_channels': len(problematic_channels),
                'problematic_wavelengths_nm': [float(self.data['wavelengths'][i]) for i in problematic_channels],
                'worst_wavelength_x_nm': float(self.data['wavelengths'][np.nanargmax(self.wavelength_stats['std_residual'][:, 0])]),
                'worst_wavelength_y_nm': float(self.data['wavelengths'][np.nanargmax(self.wavelength_stats['std_residual'][:, 1])])
            },
            'temporal_stability': {
                'r_squared_stability_x': float(self.stability_metrics.get('r_squared_stability_x', np.nan)),
                'r_squared_stability_y': float(self.stability_metrics.get('r_squared_stability_y', np.nan)),
                'r_squared_trend_x': float(self.stability_metrics.get('r_squared_trend_x', np.nan)),
                'r_squared_trend_y': float(self.stability_metrics.get('r_squared_trend_y', np.nan))
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if summary['overall_statistics']['mean_r_squared'] < 0.9:
            summary['recommendations'].append("Consider higher polynomial orders or alternative models")
        
        if summary['overall_statistics']['mean_rmse_pixels'] > 0.2:
            summary['recommendations'].append("Investigate systematic instrumental effects")
        
        if len(problematic_channels) > 0.2 * self.data['n_wavelengths']:
            summary['recommendations'].append("Review wavelength-dependent patterns and excluded channels")
        
        if (summary['temporal_stability']['r_squared_stability_x'] > 0.1 or 
            summary['temporal_stability']['r_squared_stability_y'] > 0.1):
            summary['recommendations'].append("Poor temporal stability suggests observing condition effects")
        
        # Optimal polynomial orders
        for coord_name, coord_label in [('x_coordinate', 'x'), ('y_coordinate', 'y')]:
            if self.order_assessment[coord_name]['bic']:
                optimal_idx = np.argmin(self.order_assessment[coord_name]['bic'])
                optimal_order = self.order_assessment[coord_name]['orders'][optimal_idx]
                current_order = 2 if coord_name == 'x_coordinate' else 3
                if optimal_order != current_order:
                    summary['recommendations'].append(
                        f"Consider order {optimal_order} for {coord_label}-coordinate (currently {current_order})")
        
        return summary


@optional_logger
def assess_center_fit_quality(
    converted_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    create_dashboard: bool = True,
    logger=None
) -> Tuple[CenterFitAssessment, Dict]:
    """
    Convenience function for complete center fit quality assessment.
    
    Parameters
    ----------
    converted_dir : str or Path
        Directory containing the center fitting output files.
    output_dir : str or Path, optional
        Directory to save assessment outputs. If None, uses converted_dir/center_assessment.
    create_dashboard : bool, optional
        Whether to create and save the visualization dashboard (default: True).
    logger : logging.Logger, optional
        Logger instance for structured logging.
        
    Returns
    -------
    tuple
        (assessment_object, summary_report) where assessment_object is the
        CenterFitAssessment instance and summary_report is the summary dictionary.
        
    Examples
    --------
    >>> assessment, summary = assess_center_fit_quality('/path/to/converted')
    >>> print(f"Mean R²: {summary['overall_statistics']['mean_r_squared']:.3f}")
    """
    # Initialize assessment
    assessment = CenterFitAssessment(converted_dir, logger=logger)
    
    # Run complete analysis
    assessment.load_center_data()
    assessment.calculate_basic_metrics(use_robust=True)
    assessment.analyze_wavelength_dependence()
    assessment.analyze_temporal_stability()
    assessment.evaluate_polynomial_orders()
    
    # Generate summary
    summary = assessment.generate_summary_report()
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(converted_dir) / 'center_assessment'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Save summary report
    summary_path = output_dir / 'centering_assessment_report.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Create and save dashboard
    if create_dashboard:
        dashboard_path = output_dir / 'centering_assessment_dashboard.pdf'
        fig = assessment.create_assessment_dashboard(dashboard_path)
        plt.close(fig)
    
    if logger:
        logger.info(f"Assessment complete. Mean R²: {summary['overall_statistics']['mean_r_squared']:.3f}, "
                   f"Mean RMSE: {summary['overall_statistics']['mean_rmse_pixels']:.3f} pixels")
        logger.info(f"Results saved to {output_dir}")
    
    return assessment, summary
