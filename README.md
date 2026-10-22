Brian Swingle - 10/17/2026: The purpose of this project is to understand the recent essay https://already-clean.goodfire.com/. This likely includes filling in a lot of background material, potentially reproducing claims in small examples, and producing explanatory material. The broad audience is interpretability amatuers with a physics background. I'd like to go slowly section by section through the essay, keeping things organized into separate sub-folders.


## Layout and build (added 2026-10-19)

- `source/essay.md` — plain-text copy of the essay, kept locally and not committed (it is the author's text). Fetch it with `curl -sL -o source/essay.md https://already-clean.goodfire.com/essay.md`.
- `00-preliminaries/`, `01-toy-models-of-superposition/`, then `02-…` to `11-…` — one folder per essay section, each an `index.qmd` (Quarto: markdown + executable Python).
- `common/` — shared Python (toy models, InceptionV1 helpers).
- `data/` (gitignored) — COCO val2017 images and cached activation scans.

Build the site (Quarto is in `~/.local/bin`, Python env is `~/.venvs/sci`):

```bash
export PATH=$HOME/.local/bin:$HOME/.venvs/sci/bin:$PATH
quarto preview        # live-reloading local server
quarto render         # static site in _site/
```

Executed cell results are frozen in `_freeze/`, so a plain `quarto render` does not re-run experiments unless a page changed.
