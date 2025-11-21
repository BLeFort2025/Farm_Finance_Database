# OFA Canada Farm Finance Dashboard

A Streamlit dashboard and Python pipeline to fetch, validate, transform, and visualize **Statistics Canada** farm-finance datasets at **Canada, province/territory, and census division (CD)** levels.

## Quick start (VS Code)

1. **Install Python 3.11** and open this folder in **Visual Studio Code**.
2. Create & activate a virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux**:
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline once to fetch data:
   ```bash
   python scripts/run_pipeline.py
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Automated weekly updates (GitHub Actions)

- The workflow at `.github/workflows/update.yml` runs **every Monday 06:00 UTC**.
- It fetches updated StatCan tables, computes derived metrics, and commits changed CSVs to `data/latest/`.
- Ensure repository **Actions** → **General** → **Workflow permissions** is set to **Read and write permissions** for the bot to push changes.

## Data layout

```
data/
  latest/      # current CSVs for dashboard
  archive/     # dated snapshots on change
config/
  tables.yml   # authoritative list of StatCan tables (from masterlist)
scripts/
  fetch_statcan.py  # download + change-detect
  transform.py      # standardize + derived metrics (NOI, margin, direct sales share)
  validate.py       # schema checks (pandera)
  run_pipeline.py   # orchestrator
app/
  streamlit_app.py  # Streamlit UI
```

## Attribution

Source: Statistics Canada, various tables (see `config/tables.yml`). Data may be subject to revision.
