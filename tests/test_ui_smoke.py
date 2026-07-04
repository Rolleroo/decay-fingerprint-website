"""End-to-end smoke tests of both tabs through Streamlit's AppTest harness.

Not physics validation (that lives in the other test modules) -- these
confirm the widgets are wired to the modules correctly: a paste goes in,
a table comes out, and the reverse tab's gating (closed-system checkbox)
behaves. Also the regression guard that adding the reverse tab did not
break the forward tool.
"""

from datetime import date

from streamlit.testing.v1 import AppTest

APP = "app/ui.py"


def make_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


def test_forward_tab_still_works_end_to_end():
    at = make_app()
    at.run()
    assert not at.exception

    at.text_area(key="fwd_paste").set_value("Cs-137, 1000\nCo-60, 500")
    at.button(key="fwd_run").click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # results table rendered


def test_reverse_button_is_disabled_until_closed_system_acknowledged():
    at = make_app()
    at.run()
    assert at.button(key="rev_run").disabled

    at.checkbox(key="rev_closed").check()
    at.run()
    assert not at.button(key="rev_run").disabled


def test_reverse_tab_reconstructs_and_shows_forward_check():
    at = make_app()
    at.run()
    at.text_area(key="rev_paste").set_value("Cs-137, 1000, 5%\nSr-90, 400, 3%")
    at.checkbox(key="rev_closed").check()
    at.run()
    at.button(key="rev_run").click()
    at.run()
    assert not at.exception
    # Results table + forward-check table rendered, and the forward check
    # passed for a healthy 1-year reach-back.
    assert len(at.dataframe) >= 2
    assert any("Self-check passed" in s.value for s in at.success)


def test_age_tab_solves_and_shows_metrics():
    at = make_app()
    at.run()
    at.text_area(key="age_t0_paste").set_value("Pu-241, 1.0e15\nAm-241, 2.0e13")
    # Pu-241/Am-241 after ~25 years (atoms), from the validated round-trip case.
    at.text_area(key="age_today_paste").set_value("Pu-241, 2.98e14\nAm-241, 6.95e14")
    at.checkbox(key="age_closed").check()
    at.run()
    at.button(key="age_run").click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # residual/forward-check table rendered
    assert any("Median age" in m.label for m in at.metric)


def test_forward_tab_date_mode_computes_interval():
    at = make_app()
    at.run()
    at.text_area(key="fwd_paste").set_value("Cs-137, 1000")
    at.radio(key="fwd_timemode").set_value("By date")
    at.run()
    at.date_input(key="fwd_refdate").set_value(date(2020, 1, 1))
    at.date_input(key="fwd_tgtdate").set_value(date(2026, 1, 1))
    at.run()
    at.button(key="fwd_run").click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert any("2020-01-01" in s.value for s in at.subheader)


def test_reverse_tab_sigma_convention_selector_runs():
    at = make_app()
    at.run()
    at.text_area(key="rev_paste").set_value("Cs-137, 1000, 10%\nSr-90, 400, 6%")
    at.checkbox(key="rev_closed").check()
    at.run()
    at.selectbox(key="rev_sigma_conv").set_value("2σ (95%)")
    at.run()
    at.button(key="rev_run").click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1


def test_age_tab_shows_implied_origin_date():
    at = make_app()
    at.run()
    at.text_area(key="age_t0_paste").set_value("Pu-241, 1.0e15\nAm-241, 2.0e13")
    at.text_area(key="age_today_paste").set_value("Pu-241, 2.98e14\nAm-241, 6.95e14")
    at.checkbox(key="age_use_date").check()
    at.run()
    at.date_input(key="age_meas_date").set_value(date(2026, 7, 3))
    at.checkbox(key="age_closed").check()
    at.run()
    at.button(key="age_run").click()
    at.run()
    assert not at.exception
    assert any("Implied origin date" in i.value for i in at.info)


def test_compatibility_button_disabled_until_closed_system_acknowledged():
    at = make_app()
    at.run()
    assert at.button(key="cmp_run").disabled

    at.checkbox(key="cmp_closed").check()
    at.run()
    assert not at.button(key="cmp_run").disabled


def test_compatibility_tab_scores_and_shows_verdict():
    at = make_app()
    at.run()
    at.text_area(key="cmp_t0_paste").set_value("Cs-137, 1.0e15\nSr-90, 4.0e14")
    at.text_area(key="cmp_today_paste").set_value("Cs-137, 5.0e14, 3%\nSr-90, 2.1e14, 3%")
    at.checkbox(key="cmp_closed").check()
    at.run()
    at.button(key="cmp_run").click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # per-nuclide detail table rendered
    # a verdict box (compatible / in-tension / not-compatible) was shown
    assert at.success or at.warning or at.error


def test_reverse_tab_surfaces_parse_errors():
    at = make_app()
    at.run()
    at.text_area(key="rev_paste").set_value("NotANuclide, 12")
    at.checkbox(key="rev_closed").check()
    at.run()
    at.button(key="rev_run").click()
    at.run()
    assert not at.exception
    assert any("Fix the following" in e.value for e in at.error)
