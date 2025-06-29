# 🔭 Logging Refactoring Instructions for Astronomical Pipeline Steps

This guide defines a uniform logging schema for all steps in the SPHERE/IFS data reduction pipeline.  
It is intended for AI agents or developers refactoring pipeline code to ensure consistent, testable, and multiprocessing-safe logging.

---

## 🧰 Logging Infrastructure Context

The pipeline uses a centralized logging setup provided via `logging_utils.py`, which includes:

```python
from spherical.pipeline.logging_utils import optional_logger
```

Every pipeline **step** must:

- Be decorated with `@optional_logger`
- Accept a `logger` argument (no default value, i.e. do **not** use `logger=None`)
- Use `logger` for all messages instead of `print()`

> **Static type checking tip:**
> 
> When using the `@optional_logger` decorator, always declare the `logger` argument without a default (i.e. as a required argument). This ensures static type checkers recognize that `logger` is always present and prevents false-positive errors about `logger` possibly being `None`.

**Static context fields** (`target`, `band`, `night`) are now injected automatically into all log records by the logger adapter. **You do not need to pass or log these fields manually in step functions.**

**Template example:**

```python
@optional_logger
def run_example_step(input_data, config, logger):
    logger.info("Starting example step", extra={"step": "example_step", "status": "started"})
    logger.debug(f"Input shape: {input_data.shape}, config: {config}")
    ...
    logger.info("Finished example step", extra={"step": "example_step", "status": "success"})
```

If no logger is provided, a silent `NullHandler` will be used, allowing isolated testing and reuse.

---

## 🧵 Multiprocessing Logging

The main pipeline uses a **`QueueHandler` + `QueueListener`** mechanism for multiprocessing-safe logging.  
You do **not** need to add any handlers inside step functions.

In multiprocessing worker functions:

- Use the logger passed in (either directly or by name via `logging.getLogger(logger_name)`)
- **Do not** attach `StreamHandler` or `FileHandler` inside worker processes

```python
def worker(..., logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.info("Worker started")
```

---

## ✅ Logging Guidelines for Each Pipeline Step

| Logging Level | When to Use | Example |
|---------------|-------------|---------|
| `info`        | At **start** and **end** of a step | `logger.info("Starting PSF calibration", extra={"step": "psf_calibration", "status": "started"})` |
| `debug`       | For input arguments, config values, and internal details | `logger.debug(f"Wavelength range: {lam_min}–{lam_max} µm")` |
| `warning`     | For unexpected but non-fatal issues | `logger.warning("Missing header keyword 'HIERARCH DEROT ANGLE'", extra={"step": "align_frames", "status": "failed"})` |
| `exception`   | For critical errors with traceback (inside `except`) | `logger.exception("Failed to write output cube", extra={"step": "bundle_output", "status": "failed"})` |

---

## 📦 Structured Logging at Step Completion

At the end of each pipeline step, you **must** log a structured summary using the `extra` argument to enable centralized aggregation and monitoring.

This structured entry should include:

- `step`: A short string identifying the step name (e.g. `"extract_psf"`)
- `status`: Either `"success"` or `"failed"`

**Example:**

```python
logger.info("Step finished", extra={
    "step": "bundle_output",
    "status": "success"
})
```

**If a step fails but recovers or exits early:**

```python
logger.info("Step failed gracefully", extra={
    "step": "align_frames",
    "status": "failed"
})
```

This log format ensures that reduction summaries can be aggregated across many targets.

---

## 🪐 Astronomy Context Tips

- Log missing FITS files or unexpected formats as `warning`.
- Log key metadata: target name, band, date, and wavelength range (these are now automatically included).
- Ensure shape and dtype of cube arrays are traceable via debug logs.
- Always use the `logger` passed in — never `print()` or new handlers.

---

You may now attach the relevant step code below to be refactored following these guidelines.
