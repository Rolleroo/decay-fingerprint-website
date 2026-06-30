"""Streamlit front end. All physics/parsing/conversion logic lives in
parsing.py, conversions.py, and engine.py (spec Sec 7) -- this module only
wires widgets to those modules and renders the result.

Single time point only for now: paste a fingerprint, pick a target time,
get a table of the decayed composition at that moment (inputs and
in-grown progeny). No plotting yet -- that's deliberately deferred.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.conversions import UNIT_GROUPS, ValidationError, canonicalize, scale_to_display
from app.engine import filter_nuclides_by_half_life, half_life_readable, run_time_series
from app.parsing import parse_paste

TIME_UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "hours": 3600.0,
    "days": 86400.0,
    "months": 86400.0 * 365.25 / 12,
    "years": 86400.0 * 365.25,
}

PLACEHOLDER_PASTE = "Cs-137, 3.7e9\nCo-60, 1.2e8\nSr-90, 5.0e7\nMo-99, 2.0e6"


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
    """Build the single-time fingerprint table from a one-point TimeSeriesResult.

    ``nuclides`` restricts which rows appear (e.g. after a half-life
    filter); defaults to every nuclide present (inputs and progeny).
    """
    base_series, base_unit = _base_unit_and_series(canon.kind, canon.library_unit, result)
    display_unit_for_scaling = "Bq" if canon.kind == "specific_activity" else canon.display_unit
    quantity_label = _quantity_column_label(canon)

    rows = []
    for nuclide in nuclides if nuclides is not None else result.nuclides:
        raw_value = base_series[nuclide][0]
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


def main() -> None:
    st.set_page_config(page_title="Decay Fingerprint Tool", layout="centered")
    st.title("Decay Fingerprint Tool")
    st.caption(
        "Forward decay only. Built on the published, validated `radioactivedecay` "
        "library (ICRP-107 data) -- the library is the trusted calculation engine; "
        "this app is a thin wrapper around it."
    )
    st.warning(
        "**Scope boundary:** this tool does not model spontaneous-fission ingrowth "
        "or neutron-induced activation. Results are decay-only. Do not use for "
        "reactor-burnup fingerprints without understanding this.",
        icon="⚠️",
    )

    st.subheader("Input")
    paste_text = st.text_area(
        "Paste 'nuclide, value' lines (one per line)",
        height=200,
        placeholder=PLACEHOLDER_PASTE,
    )

    group_names = [g for g, _ in UNIT_GROUPS]
    group_choice = st.selectbox("Unit group", group_names)
    units_in_group = dict(UNIT_GROUPS)[group_choice]
    unit_choice = st.selectbox("Unit", units_in_group)

    frac_as_percent = False
    if group_choice == "Relative":
        frac_as_percent = (
            st.radio("Enter values as", ["Fraction (0-1)", "Percent (0-100)"], horizontal=True)
            == "Percent (0-100)"
        )

    time_col, unit_col = st.columns(2)
    with time_col:
        time_value = st.number_input("Decay to", min_value=0.0, value=1.0)
    with unit_col:
        time_unit = st.selectbox("Time unit", list(TIME_UNIT_TO_SECONDS), index=4)  # years

    run = st.button("Decay", type="primary")

    # st.button() only returns True on the single rerun triggered by the
    # click itself -- every subsequent rerun (e.g. from a later widget
    # interaction) sees run=False. Stash the computed result in
    # session_state so the table stays in place across those reruns.
    if run:
        parse_result = parse_paste(paste_text)
        if parse_result.errors:
            st.error("Fix the following before decaying:")
            for e in parse_result.errors:
                st.text(f"Line {e.line_no}: {e.raw!r} -- {e.message}")

        if not parse_result.entries:
            if not parse_result.errors:
                st.info("Paste at least one 'nuclide, value' line.")
        else:
            try:
                canon = canonicalize(parse_result.entries, unit_choice, frac_as_percent=frac_as_percent)
            except ValidationError as exc:
                st.error(str(exc))
                canon = None

            if canon is not None:
                target_time_s = time_value * TIME_UNIT_TO_SECONDS[time_unit]
                st.session_state["canon"] = canon
                st.session_state["target_time_s"] = target_time_s
                st.session_state["result"] = run_time_series(canon, [target_time_s])

    if "result" in st.session_state:
        canon = st.session_state["canon"]
        result = st.session_state["result"]

        st.subheader(f"Fingerprint after {time_value:g} {time_unit}")

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

        table = _results_table(canon, result, kept_nuclides)
        st.dataframe(table, use_container_width=True, hide_index=True)

        with st.expander("Copy table for Excel"):
            st.caption("Click the copy icon in the corner, then paste directly into a spreadsheet.")
            st.code(table.to_csv(sep="\t", index=False), language=None)

    st.divider()
    with st.expander("How to use this tool"):
        st.markdown(
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
            "with a half-life below the threshold you set, handy for "
            "decluttering a table full of nuclides that have already "
            "decayed away to nothing.\n"
            "6. **Copy table for Excel** — open that section and click the "
            "copy icon to grab the results as tab-separated text ready to "
            "paste into a spreadsheet."
        )

    st.markdown(
        "**Data & library credits** — this tool is a thin interface over "
        "[`radioactivedecay`](https://github.com/radioactivedecay/radioactivedecay), "
        "created by **Alex Malins and Thom Lemoine**, with Ian Cullen and other "
        "contributors (MIT License, © 2020–2024 Japan Atomic Energy Agency & "
        "contributors). It performs all decay-chain physics and unit conversions; "
        "this app adds none of its own.\n\n"
        "If you use results from this tool in research, please cite the project's "
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
