This project is for running Receptive Field mapping algorithm on 2P data
Do not modify any code outside this project
We load our data using standard ibl loader ../ONE/ 
Our data is produced by the converter with documentation existing as docstring inside ../OneConverter/exporters/ which is adherent to IBL One standard which has further documentation at "../OneConverter/IBL documentation/"


## Setup from scratch on another machine

### Repo dependencies

All repos sit side by side in one parent folder (paths in this file and in the code are relative to it):

```
FANCiRepos/
├── Average_RF_mapping/        this project: git clone https://github.com/FANCilab/RF_mappers.git Average_RF_mapping
├── ONE/                       needed to run: git clone https://github.com/int-brain-lab/ONE  (tested at 636d76f, ONE-api 3.6.1)
└── OneConverter/              needed to run: git clone https://github.com/FANCilab/OneConverter.git  (tested at 21907e6)
```

ONE and OneConverter are installed as editable packages (`pip install -e`), so edits or `git pull` in those checkouts take effect without reinstalling. OneConverter imports `one`, so install ONE first.

### Python dependencies

Python >= 3.10 (ONE's minimum); tested with Python 3.14.5. Direct imports of this project, with tested versions:

| Package | Tested | Used by |
|---|---|---|
| ONE-api (from ../ONE) | 3.6.1 | loading data, ALF paths, re-indexing (brings pandas, pyarrow, iblutil) |
| OneConverter (from ../OneConverter) | 0.1.0 | conversion, `update_dataset_types`, `explain_dataset` |
| numpy | 2.3.5 | everywhere |
| scipy | 1.17.1 | mapAverageRF.py, utils.py |
| statsmodels | 0.15.0 | utils.py (neuropil LFR fit) |
| pylops | 2.8.0 | utils.py (L1 fit) |
| pyproximal | 0.13.0 | utils.py (L1 fit, FISTA) |
| sparse | 0.18.0 | single_session_plots.py, OneConverter |
| matplotlib | 3.10.9 | plotting modules, notebook |
| pycolormap-2d | 1.1.7 | plotting modules (import name `pycolormap_2d`) |
| ipykernel | 7.4.0 | running the notebook |

### Install with conda (preferred when conda is available)

From an Anaconda Prompt (or any shell where `conda` works), in the parent folder `FANCiRepos/`:

```
conda create -n rfmap -c conda-forge python=3.14 -y
conda activate rfmap
python -m pip install -e ./ONE
python -m pip install -e ./OneConverter
python -m pip install numpy scipy statsmodels pylops pyproximal sparse matplotlib pycolormap-2d ipykernel
python -m ipykernel install --user --name rfmap --display-name "Python (rfmap)"
```

Packages go through pip inside the conda env (pycolormap-2d, pylops and pyproximal are pip packages). In VS Code / Jupyter pick the `rfmap` interpreter or kernel.

### Install without conda

Same steps with a venv, from `FANCiRepos/`:

```
python -m venv rfmap-venv
rfmap-venv\Scripts\Activate.ps1        # Windows PowerShell; Linux/macOS: source rfmap-venv/bin/activate
python -m pip install -e ./ONE
python -m pip install -e ./OneConverter
python -m pip install numpy scipy statsmodels pylops pyproximal sparse matplotlib pycolormap-2d ipykernel
python -m ipykernel install --user --name rfmap --display-name "Python (rfmap)"
```

If Windows blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `rfmap-venv\Scripts\activate.bat` from cmd.

### Check and run

From inside `Average_RF_mapping/` (the project's modules `utils`, `mapAverageRF`, `single_session_plots` and `grouped_session_plots` are imported from the current folder, not installed):

```
python -c "import one.api, OneConverter, mapAverageRF, single_session_plots, grouped_session_plots; print('imports ok')"
python main.py                 # convert NM023/20260428/1 into ./ExampleSession and map its RFs
```

or open `RF_mapping_tutorial.ipynb` with the rfmap kernel (writes to ./ExampleRoot and ./ExamplePlots).

- Conversion reads raw and Suite2p data from the lab NAS (`//10.233.25.135/FANCiNAS1/Data` and `/Processed`), so the machine must be on the lab network with that share accessible. Without it, copy an already converted ONE root (e.g. `ExampleRoot/`) and skip the conversion cells/lines; mapping and plotting only need the ONE root.
- The notebook hard-codes the RF revisions `2026-10-27` and `2026-10-27a` (the dates it was first run). A fresh run writes revisions named after the run date, so update those strings.
- Use `n_shifts=5` for test runs; 500 for results.