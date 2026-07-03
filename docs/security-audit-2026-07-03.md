# Security / Input-Hardening Audit — 2026-07-03

Triggered by the tool becoming publicly deployable (Hugging Face Space) with
a file-upload feature. Scope: input sanitisation and denial-of-service
bounds for a Streamlit calculator that takes pasted text, uploaded files,
and numeric/date widgets. **Verdict: low inherent risk, now hardened at
each surface. No high-severity issues found.**

## Threat model

This is a stateless calculator. It has **no** database, no authentication,
no secrets, no user accounts, no outbound network calls, and it never writes
files or executes anything derived from user input (no `eval`/`exec`/
`pickle`/`subprocess`/`os.system`, no SQL, no shell). The nuclear data is a
bundled offline library. That removes most classic web-app vulnerability
classes outright. The realistic risks are (a) denial of service via
oversized/pathological input and (b) CSV injection in exported files.

## Surfaces, findings, mitigations

| # | Surface | Risk | Mitigation (this audit) |
|---|---------|------|--------------------------|
| 1 | Pasted text → `parse_paste` | Unbounded lines → CPU/memory DoS | `MAX_INPUT_LINES = 10,000` guard returns a single error instead of processing. Values already validated as finite, non-negative floats; NaN/Inf and unknown nuclides rejected. |
| 2 | File upload → pandas | Oversized file / XLSX zip bomb → memory DoS | Streamlit `maxUploadSize = 10 MB` (`.streamlit/config.toml`) as first line; `MAX_UPLOAD_BYTES = 8 MB` post-read and `MAX_ROWS × MAX_COLS = 100,000 × 256` shape caps in `app/ingest.py` as defence in depth. |
| 3 | `.xlsx` XML parsing (openpyxl) | XXE / billion-laughs (openpyxl uses stdlib ElementTree, no guard by default) | Added **`defusedxml`** to requirements; openpyxl auto-routes XML parsing through it when importable (verified: `openpyxl.xml.functions.fromstring` resolves to `defusedxml.common`). |
| 4 | CSV / "copy for Excel" exports | CSV injection — a cell starting with `= + - @` executing as a formula in Excel/Sheets (some result cells legitimately start with `+`/`-`, e.g. a `+5.00%` mismatch) | `_csv_safe()` prefixes a single quote to any string cell beginning with a formula trigger or control char, on every CSV download and copy-block. JSON export left as-is (not formula-interpreted by spreadsheets). Numeric cells untouched. |
| 5 | Numeric / date widgets | Out-of-range values | Bounded by `min_value`/`max_value` on every `number_input`; Monte Carlo trial counts capped (≤500k reverse, ≤200k age). Dates handled as calendar dates; reversed date pairs raise a clear error, never a negative interval into the engine. |
| 6 | Nuclide strings → `radioactivedecay` | Crafted input crashing the parser | Every nuclide parsed inside a `try/except` that turns any failure into a per-line error; the library performs no code execution. |

## Residual risks (accepted / documented)

- **Compute-time DoS within the caps.** A valid input near the line/trial
  caps with a long chain can still take a few seconds of CPU. Acceptable
  for a calculator; the caps bound it, and the HF Space is sandboxed with
  its own resource limits. Not worth a hard per-request timeout at this
  scale.
- **The `latin-1` decode fallback never fails**, so a binary file renamed
  `.csv` decodes to garbage rather than erroring — but it then produces
  ordinary per-line parse errors (no nuclide/number matches), not a crash
  or hang. Bounded by the size cap.
- **Deployment exposure.** The app binds localhost by default; the
  transient `--server.address 0.0.0.0` used once during local testing is
  not in any committed launch path. On HF Spaces, network exposure and
  isolation are the platform's responsibility.

## What was explicitly checked and found clean

- No `eval` / `exec` / `pickle` / `subprocess` / `os.system` / `open()`-for-
  write on user-derived data anywhere in `app/`.
- No f-string/format construction of shell/SQL from user input.
- Uploaded bytes are parsed in memory only; nothing is persisted to disk.
- Session state holds only computed results, no credentials.

## Tests added

- `test_parsing.py`: line-limit guard fires; at-limit input still parses.
- `test_ingest.py`: oversized-file and too-many-columns rejection.
- `test_ui_export_safety.py`: formula-leading cells neutralised, ordinary
  values (nuclide names, numbers) untouched.

Full suite after hardening: **159 passed.**
