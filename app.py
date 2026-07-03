"""Entry point for `streamlit run app.py`.

Hugging Face Spaces (and Streamlit Community Cloud) look for an `app.py`
at the repository root by default. All the real code lives in the `app/`
package; this is a thin shim so the same repo deploys unchanged. Run
locally the same way: `streamlit run app.py`.
"""

from app.ui import main

main()
