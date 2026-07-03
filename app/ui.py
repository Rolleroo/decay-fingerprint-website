"""Streamlit front end. All physics/parsing/conversion logic lives in
parsing.py, conversions.py, engine.py, reverse.py, age_solve.py, plus the
ingest.py / dates.py helpers -- this module only wires widgets to those
modules and renders the results.

Three tabs share one interface style (paste OR upload a fingerprint, pick a
unit; give a time as an interval OR a pair of dates):

- Forward decay: single time point; table of the decayed composition.
- Reverse (Mode B): known age; reconstruct the t=0 composition with Monte
  Carlo uncertainty, conditioning/assumption flags, and the default-on
  forward-check overlay.
- Age (Mode A): known t=0 composition; solve for the age by weighted
  least squares against the forward engine (docs/mode-a-addendum.md).

Shared input conveniences (added 2026-07-03): CSV/XLSX upload alongside the
paste box (app.ingest), date-mode time entry (app.dates), a coverage-factor
selector for the uncertainty convention (1σ/2σ/95%), and CSV/JSON download
buttons on every results table.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.age_solve import age_readable, solve_age_from_entries
from app.conversions import UNIT_GROUPS, ValidationError, canonicalize, scale_to_display
from app.dates import DateError, date_from_age, forward_interval_seconds
from app.engine import (
    audit_conservation,
    filter_nuclides_by_half_life,
    half_life_readable,
    run_time_series,
)
from app.ingest import IngestError, paste_text_from_upload
from app.parsing import parse_paste
from app.reverse import atoms_to_display, reconstruct_from_entries

TIME_UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "hours": 3600.0,
    "days": 86400.0,
    "months": 86400.0 * 365.25 / 12,
    "years": 86400.0 * 365.25,
}

# Coverage factor of user-supplied uncertainties (see sigma_atoms_from_entries).
# Lab certificates quote at varying k; picking the right one keeps the MC
# intervals correctly scaled. The default seed uncertainty is always 1-sigma.
SIGMA_CONVENTIONS: dict[str, float] = {
    "1σ / standard uncertainty (68%)": 1.0,
    "2σ (95%)": 2.0,
    "95% confidence (k = 1.96)": 1.96,
    "3σ (99.7%)": 3.0,
}

UPLOAD_TYPES = ["csv", "tsv", "txt", "dat", "xlsx", "xlsm", "xls"]

PLACEHOLDER_PASTE = "Cs-137, 3.7e9\nCo-60, 1.2e8\nSr-90, 5.0e7\nMo-99, 2.0e6"
PLACEHOLDER_PASTE_REVERSE = "Cs-137, 3.7e9, 5%\nSr-90, 5.0e7 ± 2e6\nAm-241, 3.1e5"


def _input_text(
    key_prefix: str, label: str, placeholder: str, height: int, help_text: str | None = None
) -> str:
    """A paste box plus an optional CSV/XLSX uploader. When a file is present
    it is converted to the same paste text (via app.ingest) and used instead
    of the box, with the auto-detected column mapping shown so the choice is
    never silent. Falls back to the paste box on any read error."""
    paste_text = st.text_area(
        label, height=height, placeholder=placeholder, key=f"{key_prefix}_paste", help=help_text
    )
    uploaded = st.file_uploader(
        "…or upload a results table (CSV / XLSX)",
        type=UPLOAD_TYPES,
        key=f"{key_prefix}_file",
        help=(
            "Gamma-spec / spreadsheet exports: the nuclide, value, and optional "
            "uncertainty columns are auto-detected. Non-UTF-8 encodings, ';' "
            "delimiters, decimal commas, and metadata preamble rows are handled."
        ),
    )
    if uploaded is not None:
        try:
            text, mapping = paste_text_from_upload(uploaded.name, uploaded.getvalue())
        except IngestError as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")
            return paste_text
        unc = mapping.uncertainty or "—"
        pct = " (as %)" if mapping.uncertainty_is_percent else ""
        st.caption(
            f"📄 Using **{uploaded.name}** — detected columns: nuclide = **{mapping.nuclide}**, "
            f"value = **{mapping.value}**, uncertainty = **{unc}**{pct}. Remove the file to use "
            f"the paste box instead."
        )
        return text
    return paste_text


def _sigma_convention_selector(key_prefix: str) -> float:
    label = st.selectbox(
        "Uncertainties are quoted at",
        list(SIGMA_CONVENTIONS),
        index=0,
        key=f"{key_prefix}_sigma_conv",
        help=(
            "Coverage factor of the pasted/uploaded uncertainties. Lab certificates "
            "often quote 2σ or 95% (k ≈ 2); selecting the right one keeps the Monte "
            "Carlo intervals correctly scaled. The default seed uncertainty is always 1σ."
        ),
    )
    return SIGMA_CONVENTIONS[label]


# A cell whose text begins with one of these can be interpreted as a formula
# when a CSV is opened in Excel/Sheets (CSV injection). Some result cells
# legitimately start with '+'/'-' (e.g. a "+5.00%" mismatch), so neutralise
# on export by prefixing a single quote rather than dropping the character.
_CSV_INJECTION_LEADERS = ("=", "+", "-", "@", "\t", "\r", "\n")


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    def clean(x):
        if isinstance(x, str) and x[:1] in _CSV_INJECTION_LEADERS:
            return "'" + x
        return x

    return df.map(clean)


def _download_buttons(df: pd.DataFrame, basename: str, key_prefix: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV",
        _csv_safe(df).to_csv(index=False).encode("utf-8"),
        file_name=f"{basename}.csv",
        mime="text/csv",
        key=f"{key_prefix}_dl_csv",
    )
    c2.download_button(
        "Download JSON",
        df.to_json(orient="records", indent=2).encode("utf-8"),
        file_name=f"{basename}.json",
        mime="application/json",
        key=f"{key_prefix}_dl_json",
    )


def _interval_input(
    key_prefix: str,
    *,
    elapsed_label: str,
    reference_label: str,
    target_label: str,
    default_unit_index: int = 4,
) -> tuple[float | None, str, str | None]:
    """Render either an elapsed-interval input or a by-dates input and return
    (seconds, human label, error message). Professionals work from reference
    dates; the elapsed path is the original behaviour. Returns seconds=None
    with an error message when the two dates are the wrong way round."""
    mode = st.radio(
        "Specify time as", ["Elapsed interval", "By date"], horizontal=True, key=f"{key_prefix}_timemode"
    )
    if mode == "Elapsed interval":
        c1, c2 = st.columns(2)
        with c1:
            value = st.number_input(elapsed_label, min_value=0.0, value=1.0, key=f"{key_prefix}_tv")
        with c2:
            unit = st.selectbox(
                "Time unit", list(TIME_UNIT_TO_SECONDS), index=default_unit_index, key=f"{key_prefix}_tu"
            )
        return value * TIME_UNIT_TO_SECONDS[unit], f"{value:g} {unit}", None

    c1, c2 = st.columns(2)
    with c1:
        ref = st.date_input(
            reference_label, value=date.today() - timedelta(days=365), key=f"{key_prefix}_refdate", format="YYYY-MM-DD"
        )
    with c2:
        tgt = st.date_input(target_label, value=date.today(), key=f"{key_prefix}_tgtdate", format="YYYY-MM-DD")
    try:
        seconds = forward_interval_seconds(ref, tgt, what=target_label.lower())
    except DateError as exc:
        return None, "", str(exc)
    return seconds, f"{ref.isoformat()} → {tgt.isoformat()}", None


def _base_unit_and_series(kind: str, library_unit: str, result):
    """Map a CanonResult kind to the engine's base series + base unit string."""
    if kind in ("activity", "specific_activity"):
        return result.activities_bq, "Bq"
    if kind == "mass":
        return result.masses_g, "g"
    if kind == "amount":
        if library_unit == "num":
            return result.atoms, "num"
        return result.moles_mol, "mol"
    return result.fractions, "fraction"


def _quantity_column_label(canon) -> str:
    if canon.kind in ("activity", "specific_activity"):
        return f"Activity ({canon.display_unit})"
    if canon.kind == "mass":
        return f"Mass ({canon.display_unit})"
    if canon.kind == "amount":
        return f"Amount ({canon.display_unit})"
    return "Percent (%)" if canon.frac_as_percent else "Fraction"


def _results_table(canon, result, nuclides: list[str] | None = None) -> pd.DataFrame:
    """Build the fingerprint table from the *last* step of a TimeSeriesResult.

    The forward tab decays across ``[0.0, target]`` (a t=0 baseline plus the
    requested time) so the conservation audit has something to check; the
    displayed composition is always the last step. ``nuclides`` restricts
    which rows appear (e.g. after a half-life filter); defaults to every
    nuclide present (inputs and progeny).
    """
    base_series, base_unit = _base_unit_and_series(canon.kind, canon.library_unit, result)
    display_unit_for_scaling = "Bq" if canon.kind == "specific_activity" else canon.display_unit
    quantity_label = _quantity_column_label(canon)

    rows = []
    for nuclide in nuclides if nuclides is not None else result.nuclides:
        raw_value = base_series[nuclide][-1]
        value = raw_value if canon.kind.startswith("fraction") else scale_to_display(
            raw_value, base_unit, display_unit_for_scaling
        )
        rows.append(
            {
                "Nuclide": nuclide,
                "Half-life": half_life_readable(nuclide),
                quantity_label: value,
            }
        )

    df = pd.DataFrame(rows, columns=["Nuclide", "Half-life", quantity_label])
    return df.sort_values(quantity_label, ascending=False).reset_index(drop=True)


def _unit_picker(key_prefix: str) -> tuple[str, bool]:
    """The shared unit dropdown (same unit set and conversion module for
    forward and reverse, reverse spec Sec 4: one source of truth for units)."""
    group_names = [g for g, _ in UNIT_GROUPS]
    group_choice = st.selectbox("Unit group", group_names, key=f"{key_prefix}_group")
    units_in_group = dict(UNIT_GROUPS)[group_choice]
    unit_choice = st.selectbox("Unit", units_in_group, key=f"{key_prefix}_unit")

    frac_as_percent = False
    if group_choice == "Relative":
        frac_as_percent = (
            st.radio(
                "Enter values as",
                ["Fraction (0-1)", "Percent (0-100)"],
                horizontal=True,
                key=f"{key_prefix}_frac",
            )
            == "Percent (0-100)"
        )
    return unit_choice, frac_as_percent


def _show_parse_errors(parse_result) -> None:
    st.error("Fix the following before running:")
    for e in parse_result.errors:
        st.text(f"Line {e.line_no}: {e.raw!r} -- {e.message}")


def _forward_tab() -> None:
    st.subheader("Input")
    paste_text = _input_text(
        "fwd",
        "Paste 'nuclide, value' lines (one per line)",
        PLACEHOLDER_PASTE,
        height=200,
    )

    unit_choice, frac_as_percent = _unit_picker("fwd")

    target_time_s, time_label, time_error = _interval_input(
        "fwd",
        elapsed_label="Decay to",
        reference_label="Reference date (of the pasted values)",
        target_label="Decay to date",
    )

    run = st.button("Decay", type="primary", key="fwd_run")

    # st.button() only returns True on the single rerun triggered by the
    # click itself -- every subsequent rerun (e.g. from a later widget
    # interaction) sees run=False. Stash the computed result in
    # session_state so the table stays in place across those reruns.
    if run:
        if time_error:
            st.error(time_error)
        parse_result = parse_paste(paste_text)
        if parse_result.errors:
            _show_parse_errors(parse_result)

        if not parse_result.entries:
            if not parse_result.errors:
                st.info("Paste at least one 'nuclide, value' line.")
        elif time_error is None:
            try:
                canon = canonicalize(parse_result.entries, unit_choice, frac_as_percent=frac_as_percent)
            except ValidationError as exc:
                st.error(str(exc))
                canon = None

            if canon is not None:
                st.session_state["canon"] = canon
                st.session_state["target_time_s"] = target_time_s
                st.session_state["fwd_time_label"] = time_label
                # Include a t=0 baseline so the Layer-3 conservation audit can
                # check atom conservation; the table shows the last step.
                st.session_state["result"] = run_time_series(canon, [0.0, target_time_s])

    if "result" in st.session_state:
        canon = st.session_state["canon"]
        result = st.session_state["result"]

        st.subheader(f"Fingerprint after {st.session_state.get('fwd_time_label', '')}")

        apply_filter = st.checkbox("Filter out short-lived nuclides", value=False)
        filter_col, filter_unit_col = st.columns(2)
        with filter_col:
            filter_value = st.number_input("Filter out nuclides with half-life below", min_value=0.0, value=3.0)
        with filter_unit_col:
            filter_unit = st.selectbox(
                "Filter unit", list(TIME_UNIT_TO_SECONDS), index=3, key="filter_unit"
            )  # months

        if apply_filter:
            filter_threshold_s = filter_value * TIME_UNIT_TO_SECONDS[filter_unit]
            # include_stable=False: when the filter is actually engaged,
            # stable end-products are cut too, not just short-lived ones.
            kept_nuclides = filter_nuclides_by_half_life(
                result, filter_threshold_s, direction="above", include_stable=False
            )
        else:
            kept_nuclides = result.nuclides

        breaches = audit_conservation(result)
        if breaches:
            st.error(
                "**Conservation self-audit failed** — the result violates a law that "
                "must always hold, so it is likely wrong. Please report this input:\n\n"
                + "\n".join(f"- {b}" for b in breaches)
            )

        table = _results_table(canon, result, kept_nuclides)
        st.dataframe(table, use_container_width=True, hide_index=True)
        _download_buttons(table, "forward_fingerprint", "fwd")

        with st.expander("Copy table for Excel"):
            st.caption("Click the copy icon in the corner, then paste directly into a spreadsheet.")
            st.code(_csv_safe(table).to_csv(sep="\t", index=False), language=None)


CONDITIONING_BADGE = {"pass": "✅ pass", "marginal": "⚠️ marginal", "fail": "❌ fail"}


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:.4g}"


def _reverse_display_scale(canon, rows) -> float:
    """For the three fraction kinds the reconstruction is renormalized so
    the t=0 composition sums to 1 (or 100) -- the coherence rule of reverse
    spec Sec 4 output 6. Absolute kinds pass through unscaled."""
    if not canon.kind.startswith("fraction_"):
        return 1.0
    total = sum(
        atoms_to_display(r.nuclide, r.median_atoms, canon)
        for r in rows
        if math.isfinite(r.median_atoms) and r.median_atoms > 0
    )
    if total <= 0:
        return 1.0
    return (100.0 if canon.frac_as_percent else 1.0) / total


def _reverse_results_table(canon, result) -> pd.DataFrame:
    scale = _reverse_display_scale(canon, result.rows)
    quantity_label = f"t=0 {_quantity_column_label(canon)}"

    rows = []
    for r in result.rows:
        median = atoms_to_display(r.nuclide, r.median_atoms, canon) * scale
        lo = atoms_to_display(r.nuclide, r.lo_atoms, canon) * scale
        hi = atoms_to_display(r.nuclide, r.hi_atoms, canon) * scale

        if r.chain_tainted and r.conditioning != "fail":
            reliability = "❌ unreliable (chain member flagged)"
        elif r.conditioning == "fail":
            reliability = "❌ unreliable"
        else:
            reliability = "✅ ok"

        if r.assumed:
            assumption = f"⚠️ not measured: {r.assumption_note}"
        elif r.assumption_dependent:
            assumption = f"⚠️ depends on: {r.assumption_note}"
        else:
            assumption = ""

        conditioning = CONDITIONING_BADGE[r.conditioning]
        if r.conditioning_note:
            conditioning += f" ({r.conditioning_note})"

        rows.append(
            {
                "Nuclide": r.nuclide,
                "Half-life": half_life_readable(r.nuclide),
                quantity_label: _fmt(median),
                "95% interval": f"{_fmt(lo)} – {_fmt(hi)}",
                "Reach-back (half-lives)": f"{r.half_lives_back:.3g}",
                "Conditioning": conditioning,
                "Reliability": reliability,
                "Assumptions": assumption,
            }
        )
    return pd.DataFrame(rows)


def _reverse_forward_check_table(canon, result) -> pd.DataFrame:
    is_fraction = canon.kind.startswith("fraction_")
    measured_total = sum(row.measured_atoms for row in result.forward_check) or 1.0

    def display(nuclide: str, atoms: float) -> float:
        if is_fraction:
            share = atoms / measured_total
            return share * (100.0 if canon.frac_as_percent else 1.0)
        return atoms_to_display(nuclide, atoms, canon)

    unit_label = (
        ("Percent of total" if canon.frac_as_percent else "Fraction of total")
        if is_fraction
        else _quantity_column_label(canon)
    )
    rows = []
    for row in result.forward_check:
        rows.append(
            {
                "Nuclide": row.nuclide,
                f"Measured today ({unit_label})": _fmt(display(row.nuclide, row.measured_atoms)),
                f"Reconstructed t=0 → forward ({unit_label})": _fmt(display(row.nuclide, row.modeled_atoms)),
                "Mismatch": (
                    f"{row.rel_diff:+.2%}"
                    if math.isfinite(row.rel_diff)
                    else "∞ (measured 0, but the reconstruction regrows it)"
                ),
            }
        )
    return pd.DataFrame(rows)


def _reverse_tab() -> None:
    st.caption(
        "Mode B: given today's measured composition and a **known age**, reconstruct "
        "the composition at t=0. Forward-model based (never raw matrix inversion); "
        "every value is a Monte Carlo distribution with reliability flags, and the "
        "result is continuously self-checked by decaying it forward again."
    )

    st.subheader("Input")
    paste_text = _input_text(
        "rev",
        "Paste 'nuclide, value[, uncertainty]' lines (one per line)",
        PLACEHOLDER_PASTE_REVERSE,
        height=200,
        help_text=(
            "The uncertainty column is optional: absolute in the same unit "
            "('Cs-137, 3.7e9, 1e8' or '3.7e9 ± 1e8') or relative ('3.7e9, 5%'). "
            "Lines without one get the default from Advanced options."
        ),
    )

    unit_choice, frac_as_percent = _unit_picker("rev")

    age_s, age_label, age_error = _interval_input(
        "rev",
        elapsed_label="Known age",
        reference_label="Origin / manufacture date",
        target_label="Measurement date",
    )

    with st.expander("Advanced (Monte Carlo settings)"):
        n_trials = int(
            st.number_input(
                "Monte Carlo trials",
                min_value=1_000,
                max_value=500_000,
                value=100_000,
                step=1_000,
                help="10⁵–10⁶ trials is the DQPB-recommended range; 10⁵ runs in seconds.",
            )
        )
        default_sigma_pct = st.number_input(
            "Default measurement uncertainty (%) for lines without one",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            help="An assumed seed value so MC runs out of the box — override per line with real measurement precision.",
        )
        coverage_k = _sigma_convention_selector("rev")

    closed_system = st.checkbox(
        "I understand these results assume a **closed system**: nothing was added to or "
        "removed from the sample over the stated age except by radioactive decay. "
        "The tool cannot detect open-system behaviour.",
        key="rev_closed",
    )

    run = st.button("Reconstruct t=0", type="primary", disabled=not closed_system, key="rev_run")
    if not closed_system:
        st.caption("Acknowledge the closed-system assumption to enable reconstruction.")

    if run:
        parse_result = parse_paste(paste_text)
        if parse_result.errors:
            _show_parse_errors(parse_result)

        if age_error:
            st.error(age_error)
        elif age_s is None or age_s <= 0:
            st.error("The known age must be greater than zero.")
        elif not parse_result.entries:
            if not parse_result.errors:
                st.info("Paste at least one 'nuclide, value' line.")
        else:
            try:
                canon = canonicalize(parse_result.entries, unit_choice, frac_as_percent=frac_as_percent)
            except ValidationError as exc:
                st.error(str(exc))
                canon = None

            if canon is not None:
                with st.spinner("Back-solving with Monte Carlo uncertainty..."):
                    rev_result = reconstruct_from_entries(
                        parse_result.entries,
                        canon,
                        age_s,
                        default_rel_sigma=default_sigma_pct / 100.0,
                        n_trials=n_trials,
                        coverage_k=coverage_k,
                    )
                st.session_state["rev_canon"] = canon
                st.session_state["rev_result"] = rev_result
                st.session_state["rev_age_label"] = age_label

    if "rev_result" in st.session_state:
        canon = st.session_state["rev_canon"]
        result = st.session_state["rev_result"]

        st.subheader(f"Reconstructed composition — reach-back {st.session_state['rev_age_label']}")

        # Forward-check verdict first: it overrides everything else.
        if result.forward_check_ok:
            st.success(
                "Forward check passed: decaying the reconstructed t=0 composition "
                "forward by the known age reproduces the measured values."
            )
        else:
            st.error(
                "Forward check FAILED: the reconstruction does not reproduce the "
                "measured input when decayed forward. Treat every value below as "
                "suspect regardless of its individual flags."
            )

        for w in result.warnings:
            if "conditional on closed-system" in w:
                st.caption(f"ℹ️ {w}")
            elif "Negative reconstructed" in w or "Forward check failed" in w:
                st.error(w)
            else:
                st.warning(w)

        table = _reverse_results_table(canon, result)
        if table.empty:
            st.info("Nothing to reconstruct (no radioactive nuclides in the input).")
            return
        st.dataframe(table, use_container_width=True, hide_index=True)
        _download_buttons(table, "reconstructed_t0", "rev")
        st.caption(
            (
                f"{result.n_trials:,} Monte Carlo trials; values are medians with 95% intervals."
                if result.n_trials
                else "All uncertainties are zero — deterministic back-solve, no MC spread."
            )
            + " 'Conditioning' is numerical stability at this reach-back; 'Assumptions' is a "
            "separate axis — a green conditioning flag never implies the value is uniquely determined."
        )

        if result.excluded_stable:
            st.caption(
                f"Excluded stable nuclides (carry no timing information and cannot be "
                f"un-grown): {', '.join(result.excluded_stable)}"
            )
        if result.pruned:
            st.caption(
                f"Pruned intermediates (half-life negligible against the reach-back age; "
                f"decay *through* them is still modelled): {', '.join(result.pruned)}"
            )

        with st.expander("Forward-check overlay (always computed)", expanded=not result.forward_check_ok):
            st.caption(
                "Reconstructed t=0 → decayed forward by the known age → compared against "
                "what you measured. Mismatch means the result is suspect."
            )
            st.dataframe(_reverse_forward_check_table(canon, result), use_container_width=True, hide_index=True)

        with st.expander("Copy table for Excel"):
            st.caption("Click the copy icon in the corner, then paste directly into a spreadsheet.")
            st.code(_csv_safe(table).to_csv(sep="\t", index=False), language=None)


def _age_results_table(canon_today, result) -> pd.DataFrame:
    unit_label = _quantity_column_label(canon_today)
    rows = []
    for r in result.residuals:
        rows.append(
            {
                "Nuclide": r.nuclide,
                "Half-life": half_life_readable(r.nuclide),
                f"Measured today ({unit_label})": _fmt(atoms_to_display(r.nuclide, r.measured_atoms, canon_today)),
                f"t=0 → forward at solved age ({unit_label})": _fmt(atoms_to_display(r.nuclide, r.modeled_atoms, canon_today)),
                "Mismatch": (
                    f"{r.mismatch_rel:+.2%} ({r.mismatch_sigma:+.1f}σ)"
                    if math.isfinite(r.mismatch_rel)
                    else "∞ (measured 0)"
                ),
                "Constrains the age": "✅ yes" if r.informative else "— no",
            }
        )
    return pd.DataFrame(rows)


def _age_tab() -> None:
    st.caption(
        "Mode A: given the composition the sample **started** with and what it "
        "measures **today**, solve for its age. A weighted least-squares fit of one "
        "scalar against the trusted forward engine — with Monte Carlo uncertainty, "
        "resolvability gates, and ambiguity detection. Absolute units only "
        "(see docs/mode-a-addendum.md)."
    )

    st.subheader("Known composition at t=0")
    t0_text = _input_text(
        "age_t0",
        "Paste 'nuclide, value[, uncertainty]' lines — lines without an uncertainty are treated as exact",
        "Pu-241, 1.0e15\nAm-241, 2.0e13",
        height=150,
    )
    t0_unit, _ = _unit_picker("age_t0")

    st.subheader("Measured composition today")
    today_text = _input_text(
        "age_today",
        "Paste 'nuclide, value[, uncertainty]' lines — lines without one get the default uncertainty",
        "Pu-241, 2.9e14, 3%\nAm-241, 6.8e14 ± 3e13",
        height=150,
    )
    today_unit, _ = _unit_picker("age_today")

    measurement_date = None
    if st.checkbox(
        "I have a measurement date (show the implied origin/production date)", key="age_use_date"
    ):
        measurement_date = st.date_input(
            "Measurement date", value=date.today(), key="age_meas_date", format="YYYY-MM-DD"
        )

    with st.expander("Advanced (Monte Carlo settings)"):
        n_trials = int(
            st.number_input(
                "Monte Carlo trials",
                min_value=1_000,
                max_value=200_000,
                value=20_000,
                step=1_000,
                key="age_trials",
                help="Each trial re-solves the age, so the default is lower than reverse mode's.",
            )
        )
        default_sigma_pct = st.number_input(
            "Default measurement uncertainty (%) for today-lines without one",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            key="age_sigma",
        )
        coverage_k = _sigma_convention_selector("age")

    closed_system = st.checkbox(
        "I understand the solved age assumes a **closed system** and that the t=0 "
        "composition is complete for every measured chain.",
        key="age_closed",
    )
    run = st.button("Solve for age", type="primary", disabled=not closed_system, key="age_run")
    if not closed_system:
        st.caption("Acknowledge the assumptions to enable the solve.")

    if run:
        parse_t0 = parse_paste(t0_text)
        parse_today = parse_paste(today_text)
        ok = True
        for label, parsed in [("t=0", parse_t0), ("today", parse_today)]:
            if parsed.errors:
                st.error(f"Fix the following in the {label} paste:")
                for e in parsed.errors:
                    st.text(f"Line {e.line_no}: {e.raw!r} -- {e.message}")
                ok = False
            elif not parsed.entries:
                st.info(f"Paste at least one 'nuclide, value' line for {label}.")
                ok = False

        if ok:
            try:
                canon_t0 = canonicalize(parse_t0.entries, t0_unit)
                canon_today = canonicalize(parse_today.entries, today_unit)
                with st.spinner("Fitting the age with Monte Carlo uncertainty..."):
                    result = solve_age_from_entries(
                        parse_t0.entries,
                        canon_t0,
                        parse_today.entries,
                        canon_today,
                        default_rel_sigma=default_sigma_pct / 100.0,
                        n_trials=n_trials,
                        coverage_k=coverage_k,
                    )
            except ValidationError as exc:
                st.error(str(exc))
            else:
                st.session_state["age_result"] = result
                st.session_state["age_canon_today"] = canon_today
                st.session_state["age_measurement_date"] = (
                    measurement_date.isoformat() if measurement_date is not None else None
                )

    if "age_result" in st.session_state:
        result = st.session_state["age_result"]
        canon_today = st.session_state["age_canon_today"]

        st.subheader("Solved age")
        if not result.resolvable:
            st.error(
                "**Not resolvable.** The measurements do not pin an age down "
                "(see the warnings below). No age is certified."
            )
        if math.isfinite(result.age_s_median):
            col_med, col_lo, col_hi = st.columns(3)
            col_med.metric("Median age", age_readable(result.age_s_median))
            col_lo.metric("95% low", age_readable(result.age_s_lo))
            col_hi.metric("95% high", age_readable(result.age_s_hi))
            st.caption(
                f"Central weighted-least-squares age: {age_readable(result.age_s)}. "
                + (
                    f"{result.n_trials:,} Monte Carlo trials."
                    if result.n_trials
                    else "All uncertainties zero — deterministic fit, no MC spread."
                )
                + f" Fit quality chi²/dof = {result.chi2_per_dof:.3g}."
            )
            meas_iso = st.session_state.get("age_measurement_date")
            if meas_iso and result.resolvable and math.isfinite(result.age_s_median):
                meas = date.fromisoformat(meas_iso)
                origin = date_from_age(meas, result.age_s_median)
                lo = date_from_age(meas, result.age_s_hi)  # older age -> earlier date
                hi = date_from_age(meas, result.age_s_lo)
                st.info(
                    f"**Implied origin date:** {origin.isoformat()} "
                    f"(95% range {lo.isoformat()} – {hi.isoformat()}), "
                    f"measured on {meas.isoformat()}."
                )
        if result.ambiguous_ages_s:
            st.warning(
                "**Ambiguous:** these ages fit comparably — "
                + ", ".join(age_readable(t) for t in result.ambiguous_ages_s)
            )

        for w in result.warnings:
            if "conditional on closed-system" in w:
                st.caption(f"ℹ️ {w}")
            elif "Ambiguous age" in w:
                continue  # already shown as the headline warning above
            elif "Not resolvable" in w or "Poor fit" in w:
                st.error(w)
            else:
                st.warning(w)

        st.markdown("**Forward check at the solved age** (known t=0 decayed forward, vs measured):")
        age_table = _age_results_table(canon_today, result)
        st.dataframe(age_table, use_container_width=True, hide_index=True)
        _download_buttons(age_table, "age_forward_check", "age")


def main() -> None:
    st.set_page_config(page_title="Decay Fingerprint Tool", layout="wide")
    st.title("Decay Fingerprint Tool")
    st.caption(
        "Built on the published, validated `radioactivedecay` library (ICRP-107 data) -- "
        "the library is the trusted calculation engine; this app is a thin wrapper around it. "
        "Forward decay, plus reverse reconstruction of the original composition when the age is known."
    )
    st.warning(
        "**Scope boundary:** this tool does not model spontaneous-fission ingrowth "
        "or neutron-induced activation. Results are decay-only. Do not use for "
        "reactor-burnup fingerprints without understanding this.",
        icon="⚠️",
    )

    tab_forward, tab_reverse, tab_age = st.tabs(
        [
            "Forward decay",
            "Reverse — reconstruct t=0 (known age)",
            "Age — solve for age (known t=0)",
        ]
    )
    with tab_forward:
        _forward_tab()
    with tab_reverse:
        _reverse_tab()
    with tab_age:
        _age_tab()

    st.divider()
    with st.expander("How to use this tool"):
        st.markdown(
            "**Forward decay** — what does this fingerprint look like after some time?\n\n"
            "1. **Paste your fingerprint** — one `nuclide, value` per line "
            "(e.g. `Cs-137, 3.7e9`). Comma, tab, or space all work as the "
            "separator, so pasting straight out of a spreadsheet is fine.\n"
            "2. **Pick the unit** the values are in — Activity, Specific "
            "activity, Mass, Amount, or a Relative fraction/percent. For "
            "Relative units, also choose whether you're entering a fraction "
            "(0-1) or a percent (0-100).\n"
            "3. **Set 'Decay to'** — how far forward in time to run the decay, "
            "and the time unit.\n"
            "4. **Click Decay.** The table shows every nuclide present at "
            "that time, including progeny that grew in (not just what you "
            "pasted), sorted by quantity.\n"
            "5. **Filter out short-lived nuclides** (optional) — tick the "
            "checkbox to hide anything (including stable end-products) "
            "with a half-life below the threshold you set.\n\n"
            "**Reverse (Mode B)** — what did this sample look like when it was made, "
            "given that I know how old it is?\n\n"
            "1. **Paste today's measured fingerprint** the same way, optionally adding "
            "a third column with each line's measurement uncertainty (`, 5%` or `± 1e8`).\n"
            "2. **Pick the unit** — the same unit list as forward.\n"
            "3. **Enter the known age** of the sample.\n"
            "4. **Acknowledge the closed-system assumption** and click Reconstruct.\n"
            "5. **Read the flags, not just the numbers.** Every nuclide is shown — "
            "none are withheld — but 'Conditioning' tells you whether the back-solve "
            "is numerically trustworthy at this reach-back, 'Assumptions' tells you "
            "what the value depends on that wasn't measured, and if any member of a "
            "decay chain fails, the whole chain is flagged. The forward-check overlay "
            "re-decays the answer to verify it reproduces your input.\n\n"
            "**Age (Mode A)** — how old is this sample, given I know what it started as?\n\n"
            "1. **Paste the known t=0 composition** (e.g. as manufactured/certified) — "
            "lines are treated as exact unless you add an uncertainty column.\n"
            "2. **Paste today's measured composition** — lines without an uncertainty "
            "get the configurable default. Absolute units only (no fractions).\n"
            "3. **Acknowledge the assumptions** and click Solve for age.\n"
            "4. **Read the verdict**: median age with a 95% interval, a residual "
            "table showing how well each nuclide fits at that age (and whether it "
            "actually constrains the age), plus loud flags when the age is not "
            "resolvable, when two ages fit equally well, or when the inputs are "
            "inconsistent with closed-system decay.\n\n"
            "**Copy table for Excel** — open that section and click the copy icon to "
            "grab the results as tab-separated text ready to paste into a spreadsheet.\n\n"
            "**Shared conveniences (all tabs):**\n"
            "- **Upload instead of paste** — drop a CSV or XLSX results table "
            "(e.g. a gamma-spec export); the nuclide, value, and uncertainty columns "
            "are auto-detected and shown. Non-UTF-8 encodings, `;` delimiters, decimal "
            "commas, and metadata preamble rows are handled.\n"
            "- **By date** — instead of an elapsed interval, give a reference date and a "
            "target/measurement date; the interval is computed for you. On the Age tab, "
            "supplying a measurement date converts the solved age into an implied "
            "origin/production date.\n"
            "- **Uncertainty convention** — tell the tool whether pasted/uploaded "
            "uncertainties are 1σ, 2σ, or 95% (k ≈ 2) so the Monte Carlo intervals are "
            "scaled correctly (Advanced options, reverse and age tabs).\n"
            "- **Download** — every results table has CSV and JSON download buttons.\n\n"
            "**Units** — activity (Bq…TBq, dpm, Ci…pCi), specific activity (Bq/g, Bq/kg), "
            "mass (g…t incl. µg), amount (mol, atoms), and relative fraction/percent."
        )

    st.markdown(
        "**Data & library credits** — this tool is a thin interface over "
        "[`radioactivedecay`](https://github.com/radioactivedecay/radioactivedecay), "
        "created by **Alex Malins and Thom Lemoine**, with Ian Cullen and other "
        "contributors (MIT License, © 2020–2024 Japan Atomic Energy Agency & "
        "contributors). It performs all decay-chain physics and unit conversions; "
        "this app adds none of its own.\n\n"
        "**Reverse-mode method** — the reverse mode's design (forward-model "
        "back-solve, Monte Carlo uncertainty propagation, the analytical-"
        "resolvability gate applied before MC, and short-lived-intermediate "
        "pruning) transfers the pattern published in **DQPB**:\n"
        "> Timothy Pollard, Jon Woodhead, John Hellstrom, John Engel, Roger Powell "
        "& Russell Drysdale, *DQPB: software for calculating disequilibrium U–Pb "
        "ages*. Geochronology, **5**, 181–196 (2023). DOI: "
        "[10.5194/gchron-5-181-2023](https://doi.org/10.5194/gchron-5-181-2023).\n\n"
        "DQPB's underlying package [`pysoplot`](https://pypi.org/project/pysoplot/) "
        "(MIT License) was studied as the reference implementation; the pattern is "
        "transferred, no code is copied. The governing decay-chain equations are "
        "**Bateman's** (H. Bateman, *Solution of a system of differential equations "
        "occurring in the theory of radioactive transformations*, Proc. Cambridge "
        "Philos. Soc., **15**, 423–427, 1910), which the validation suite also "
        "evaluates directly as an independent cross-check of the engine.\n\n"
        "If you use results from this tool in research, please cite the library's "
        "own paper, per their request:\n"
        "> Alex Malins & Thom Lemoine, *radioactivedecay: A Python package for "
        "radioactive decay calculations*. Journal of Open Source Software, **7** (71), "
        "3318 (2022). DOI: [10.21105/joss.03318](https://doi.org/10.21105/joss.03318).\n\n"
        "Nuclear data:\n"
        "- Decay data: ICRP, 2008. *Nuclear Decay Data for Dosimetric Calculations*, "
        "ICRP Publication 107, Ann. ICRP 38(3). © 2008 A. Endo and K.F. Eckerman.\n"
        "- Atomic mass data: W.J. Huang et al. 2021, *AME2020 (I)*, Chinese Phys. C "
        "45 030002; Meng Wang et al. 2021, *AME2020 (II)*, Chinese Phys. C 45 030003; "
        "F.G. Kondev et al. 2021, *NUBASE2020*, Chinese Phys. C 45 030001. Source: "
        "[AMDC](https://www-nds.iaea.org/amdc/)."
    )


if __name__ == "__main__":
    main()
