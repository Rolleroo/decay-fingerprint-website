from app.parsing import parse_paste


def test_three_spellings_resolve_identically():
    result = parse_paste("Cs-137, 1\nCs137, 1\n137Cs, 1")
    # all three are the same nuclide -> duplicate detection should fire,
    # proving they normalized to the same canonical name.
    assert not result.entries
    assert len(result.errors) == 3
    assert all("Duplicate" in e.message for e in result.errors)


def test_metastable_preserved_across_spellings():
    for spelling in ["Tc-99m", "Tc99m", "99mTc"]:
        result = parse_paste(f"{spelling}, 5")
        assert result.ok, result.errors
        assert result.entries[0].nuclide == "Tc-99m"


def test_ground_state_not_confused_with_metastable():
    result = parse_paste("Tc-99, 5\nTc-99m, 5")
    assert result.ok
    names = {e.nuclide for e in result.entries}
    assert names == {"Tc-99", "Tc-99m"}


def test_casing_and_whitespace_tolerance():
    result = parse_paste("  CS-137  ,   10  ")
    assert result.ok
    assert result.entries[0].nuclide == "Cs-137"
    assert result.entries[0].value == 10.0


def test_unknown_nuclide_is_flagged_not_guessed():
    result = parse_paste("Xx-999, 1")
    assert not result.ok
    assert result.errors[0].line_no == 1
    assert "Unrecognized" in result.errors[0].message


def test_one_bad_line_does_not_break_the_rest():
    result = parse_paste("Cs-137, 10\nXx-999, 1\nCo-60, 5")
    assert len(result.entries) == 2
    assert {e.nuclide for e in result.entries} == {"Cs-137", "Co-60"}
    assert len(result.errors) == 1
    assert result.errors[0].line_no == 2


def test_duplicate_nuclide_is_flagged_as_error():
    result = parse_paste("Cs-137, 10\nCs-137, 5")
    assert not result.ok
    assert len(result.errors) == 2
    assert not result.entries


def test_duplicate_does_not_suppress_unrelated_errors():
    result = parse_paste("Cs-137, 10\nCs-137, 5\nXx-999, 1\nCo-60, 7")
    assert {e.nuclide for e in result.entries} == {"Co-60"}
    messages = [e.message for e in result.errors]
    assert any("Duplicate" in m for m in messages)
    assert any("Unrecognized" in m for m in messages)


def test_blank_lines_are_skipped():
    result = parse_paste("Cs-137, 10\n\n\nCo-60, 5\n")
    assert result.ok
    assert len(result.entries) == 2


def test_empty_input_handled_gracefully():
    result = parse_paste("")
    assert result.ok
    assert result.entries == []
    assert result.errors == []


def test_single_line_input():
    result = parse_paste("Cs-137, 10")
    assert result.ok
    assert len(result.entries) == 1


def test_line_with_no_separator_at_all_is_an_error():
    result = parse_paste("Cs-13710")
    assert not result.ok
    assert "format" in result.errors[0].message.lower()


def test_tab_separated_line_parses_like_excel_paste():
    # Copying two adjacent spreadsheet cells and pasting lands as
    # tab-separated, not comma-separated.
    result = parse_paste("Cs-137\t10\nCo-60\t5")
    assert result.ok, result.errors
    assert {(e.nuclide, e.value) for e in result.entries} == {("Cs-137", 10.0), ("Co-60", 5.0)}


def test_space_separated_line_parses():
    result = parse_paste("Cs-137 10\nCo-60 5")
    assert result.ok, result.errors
    assert {(e.nuclide, e.value) for e in result.entries} == {("Cs-137", 10.0), ("Co-60", 5.0)}


def test_multiple_spaces_between_nuclide_and_value_parses():
    result = parse_paste("Cs-137      10")
    assert result.ok, result.errors
    assert result.entries[0].nuclide == "Cs-137"
    assert result.entries[0].value == 10.0


def test_comma_wins_over_tab_and_space_if_present():
    result = parse_paste("Cs-137, 10\t")
    assert result.ok, result.errors
    assert result.entries[0].nuclide == "Cs-137"
    assert result.entries[0].value == 10.0


def test_tab_wins_over_space_if_no_comma():
    result = parse_paste("Cs-137\t10 extra")
    # tab is the separator, so "10 extra" is the (invalid) value token --
    # confirms tab beats space rather than space splitting first.
    assert not result.ok
    assert "not a number" in result.errors[0].message.lower()


def test_negative_value_rejected():
    result = parse_paste("Cs-137, -5")
    assert not result.ok
    assert "negative" in result.errors[0].message.lower()


def test_zero_value_allowed():
    result = parse_paste("Cs-137, 0")
    assert result.ok
    assert result.entries[0].value == 0.0


def test_non_numeric_value_rejected():
    result = parse_paste("Cs-137, abc")
    assert not result.ok
    assert "not a number" in result.errors[0].message.lower()


def test_nan_and_inf_rejected():
    result = parse_paste("Cs-137, nan\nCo-60, inf")
    assert not result.ok
    assert len(result.errors) == 2
