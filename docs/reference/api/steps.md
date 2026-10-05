# Pipeline steps

One module per reduction step. You normally switch steps on and off through the
configuration rather than calling these directly.

```{eval-rst}
.. autosummary::
   :toctree: generated

   spherical.pipeline.steps.download_data
   spherical.pipeline.steps.wavelength_calibration
   spherical.pipeline.steps.extract_cubes
   spherical.pipeline.steps.bundle_output
   spherical.pipeline.steps.irdis_calibration
   spherical.pipeline.steps.irdis_preprocess
   spherical.pipeline.steps.frame_info
   spherical.pipeline.steps.cube_header_update
   spherical.pipeline.steps.find_star
   spherical.pipeline.steps.process_centers
   spherical.pipeline.steps.plot_center_evolution
   spherical.pipeline.steps.spot_photometry
   spherical.pipeline.steps.flux_psf_calibration
   spherical.pipeline.steps.spot_to_flux
   spherical.pipeline.steps.align_frames
```
