"""CSV-injection safety of the download / copy exports (security audit
2026-07-03). A results cell that begins with '=', '+', '-', '@' (or a
control char) can execute as a formula when the CSV is opened in a
spreadsheet; the export helper must neutralise those while leaving ordinary
values (nuclide names, numbers) untouched.
"""

import pandas as pd

from app.ui import _csv_safe


def test_formula_leading_cells_are_neutralised():
    df = pd.DataFrame(
        {
            "Nuclide": ["Cs-137", "Co-60"],
            "Mismatch": ["+5.00%", "-3.00%"],
            "Formula": ["=SUM(A1:A9)", "@cmd"],
        }
    )
    safe = _csv_safe(df)
    assert safe.loc[0, "Mismatch"] == "'+5.00%"
    assert safe.loc[1, "Mismatch"] == "'-3.00%"
    assert safe.loc[0, "Formula"] == "'=SUM(A1:A9)"
    assert safe.loc[1, "Formula"] == "'@cmd"


def test_ordinary_values_untouched():
    df = pd.DataFrame({"Nuclide": ["Cs-137"], "Activity": ["3.7e9"], "Num": [3.7e9]})
    safe = _csv_safe(df)
    assert safe.loc[0, "Nuclide"] == "Cs-137"  # starts with a letter
    assert safe.loc[0, "Activity"] == "3.7e9"  # starts with a digit
    assert safe.loc[0, "Num"] == 3.7e9  # numeric cell, unchanged
