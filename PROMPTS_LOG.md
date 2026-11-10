# Prompts and decision log

This log records important assistant prompts and decisions as they occur. It
does not replace the required unedited conversation history.

## Milestone 0

- Scope: complete paper/data/documentation preparation only; wait for explicit
  approval before implementing the inference pipeline.
- Decision: use the existing Conda Python 3.13 environment and install only
  narrowly required dependencies.
- Decision: use exact versioned arXiv PDF URLs and verify each version through
  its arXiv abstract page before extraction.
- Decision: use PyMuPDF so each extracted passage retains its original PDF
  page number.
- Decision: keep the benchmark in a separate JSON artifact and do not run it
  through the system before review.
- Decision: prepare the LiveKit collector but do not bulk collect or manually
  grade issues yet.

Future entries must distinguish assistant suggestions from the candidate's
decisions and must record mistakes only when they are actually observed.
