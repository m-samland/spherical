# Writing docs

After reading this page you can write a page that fits the site and passes its
checks.

## Voice

- Write in the second person, present tense and active voice.
- Keep one idea per paragraph and use sentence case for headings.
- Do not use the em-dash (U+2014). Rephrase the sentence instead of swapping in
  another punctuation mark.
- Do not start a sentence with a clause that ends in a colon, and use semicolons
  sparingly.
- Avoid these words unless they are literally true: `seamless`, `powerful`,
  `effortless`, `comprehensive`, `cutting-edge`, `state-of-the-art`, `leverage`,
  `unlock`, `delve`, `simply`, `just`, `easily`, `intuitive`. `robust` is fine in
  its statistical sense.

`tests/docs/test_prose_rules.py` checks the em-dash and the word list on every
hand-written page and on the README. It ignores code blocks, inline code and link
targets. The other rules are checked in review.

## Page shape

- Open each page with what the reader can do after reading it.
- Getting started pages end with a "Next" link to the following page.
- Prefer a short page that does one job to a long page that does several.

## Facts and outputs

- Code samples are complete and can be copied and run as they stand.
- Outputs are pasted from a real run, never written by hand.
- Instrument facts link to a primary source, the instrument paper or the
  [ESO SPHERE User Manual](https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc.html).

## Callouts

Three callouts have fixed meanings.

`expected-result`
: What a correct run produces, so readers can check their own.

`instrument-background`
: SPHERE context a reader who knows the instrument can skip.

`common-mistake`
: An error readers are likely to make, and how to avoid it.

````markdown
```{common-mistake}
`*_parallactic_angles.fits` holds `DEROT ANGLE`, not the parallactic angle.
```
````

Use `note` and `warning` for everything else.

## Glossary and tabs

Link a glossary term on its first use on each page, with
``{term}`DEROT ANGLE` ``. Add new terms to the [glossary](../reference/glossary.md).

Instrument and installer choices use sphinx-design tabs whose choice carries over
between pages. Use these sync groups and keys.

| `:sync-group:` | `:sync:` keys |
|---|---|
| `instrument` | `ifs`, `irdis` |
| `installer` | `pip`, `pixi` |
| `install-scope` | `database`, `pipeline` |

## Generated reference

Directives that produce reStructuredText (`config-table`, `step-table`,
`autosummary`, `argparse`) must sit inside an `{eval-rst}` fence. In a MyST fence
the build stops with "must be written inside an {eval-rst} block".

The diagram directives produce HTML and go in ordinary MyST fences.
`step-diagram` draws the reduction steps from the step registry and takes no
options. `pipeline-map` and `sequence-strip` take one option per stage
(`:archive:`, `:database:`, `:reduction:`, `:trap:`, `:products:`), each naming the
page that stage links to. The build fails when a named page does not exist, and
`step-diagram` fails when a step is missing from `PHASES` in
`docs/_ext/step_diagram.py`.

- A new module goes on one of the [Python API](../reference/api/index.md) pages.
  `tests/docs/test_api_coverage.py` fails until it does.
- A new configuration field needs a `#:` comment above it.
  `tests/docs/test_config_docs.py` fails until it has one.
- A new pipeline step needs a summary in `step_registry.py`.
  `tests/docs/test_step_summaries.py` fails until it has one.

## Building

```bash
pixi run -e docs docs-clean && pixi run -e docs docs
```

Warnings are errors. An incremental build hides warnings from pages that did not
change, so clean first before you trust a build. While writing, run
`pixi run -e docs docs-live` for a preview that reloads on save. Without a network
connection, set `SPHINX_OFFLINE=1` to skip the intersphinx inventories.
