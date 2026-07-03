# RRIN Project — File-by-File Edit Guide

This document lists **every single file** in this project folder and tells
you exactly what to do with it: whether to leave it alone, or exactly what
to type and where. You do not need to understand programming to follow
this — just match the text exactly.

This guide assumes you are using a **Mac**.

---

## 🚀 If You Are Training ONLY on Kaggle (Most Common Case)

If you never plan to run `python main.py` on your own Mac — only on
Kaggle's website — here is exactly which parts of this guide apply to you:

| Section | Do you need it? |
|---|---|
| Part 1, Step 1 — create `.env` | 🔵 Optional — only useful if you ever want to run things locally on your Mac. Not needed for Kaggle-only training |
| Part 1, Step 2 — fill in Kaggle credentials in `.env` | 🔵 Optional — same as above. Your Kaggle username/key are **not** needed as Kaggle Secrets. `kagglehub` authenticates automatically |
| Part 1, Step 3 — edit `config.yaml` dataset paths | ❌ **Skip this entirely.** The Kaggle notebook builds its own separate config automatically — explained in Part 4 |
| Part 2 — optional edits | ❌ Skip unless you want non-default behaviour |
| Part 3 — the AI engine files | — Nothing to do, just leave them alone as always |
| Part 4 — the Kaggle notebook | ✅ **This is your main destination.** Read this part fully |
| "Final Checklist" at the bottom | ❌ Skip — that checklist is for local Mac runs only |

In short: skip straight to **Part 4**.

---

## ⚠️ FIRST — About Your `.env` File Being "Invisible"

If you copied `.env.example` to `.env` and now can't see it in Finder —
**this is correct and expected.** It is not broken or lost.

On macOS, any file whose name starts with a dot (`.`) is a hidden system
file. Finder hides these by default. Your `.env` file is sitting right
there in your project folder; Finder is just choosing not to show it.

**To see it in Finder right now:**
Open the folder in Finder, then press **⌘ Cmd + Shift + .** (period).
Hidden files will fade into view. Press the same combo again to hide them.

**To edit it without ever touching Finder (recommended):**
Open the Terminal app, go to your project folder, and type:
```
nano .env
```
This opens the file in a simple text editor inside the Terminal window.
Use the arrow keys to move around, type to edit, then press **Control + O**
then **Enter** to save, then **Control + X** to exit.

**Or, to open it in TextEdit directly:**
```
open -e .env
```
This bypasses Finder entirely and opens the hidden file in TextEdit.

You will use one of these two methods (`nano` or `open -e`) every time
this guide tells you to edit a file whose name starts with a dot.

---

## How To Read This Guide

Every file below is listed with one of these three labels:

| Label | Meaning |
|---|---|
| 🟢 **NO EDIT — leave alone** | This is part of the AI program itself. Never open or change it. |
| 🟡 **EDIT REQUIRED** | You must open this and change something before running the project. |
| 🔵 **OPTIONAL EDIT** | Only change this if you want to customise behaviour. Works fine without editing. |

For every file marked 🟡 or 🔵, you will see four things:
1. **FIND THIS** — the exact text currently in the file
2. **REPLACE WITH** — exactly what to type instead
3. **HOW TO DO IT** — the precise steps on a Mac
4. **HOW TO CHECK IT WORKED** — a way to confirm the change saved correctly

---

## Quick Reference Table (all files at a glance)

| # | File Path | Status |
|---|---|---|
| 1 | `.env.example` | 🟢 Template only — see step below to copy it |
| 2 | `.env` (you create this) | 🟡 EDIT REQUIRED |
| 3 | `config.yaml` | 🟡 EDIT REQUIRED |
| 4 | `requirements.txt` | 🟢 NO EDIT |
| 5 | `setup.sh` | 🟢 NO EDIT (just run it) |
| 6 | `main.py` | 🔵 OPTIONAL EDIT |
| 7 | `README.md` | 🟢 NO EDIT (this file) |
| 8 | `src/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 9 | `src/config.py` | 🟢 NO EDIT |
| 10 | `src/database.py` | 🟢 NO EDIT |
| 11 | `src/dataset_parsers.py` | 🔵 OPTIONAL EDIT (only if adding a new dataset type) |
| 12 | `src/quality_scoring.py` | 🟢 NO EDIT |
| 13 | `src/degradation.py` | 🟢 NO EDIT |
| 14 | `src/splits.py` | 🟢 NO EDIT |
| 15 | `src/datasets.py` | 🟢 NO EDIT |
| 16 | `src/models/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 17 | `src/models/generator.py` | 🟢 NO EDIT |
| 18 | `src/models/discriminator.py` | 🟢 NO EDIT |
| 19 | `src/models/losses.py` | 🟢 NO EDIT |
| 20 | `src/training/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 21 | `src/training/train.py` | 🟢 NO EDIT |
| 22 | `src/training/checkpoints.py` | 🟢 NO EDIT |
| 23 | `src/training/domain_adaptation.py` | 🟢 NO EDIT |
| 24 | `src/evaluation/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 25 | `src/evaluation/metrics.py` | 🟢 NO EDIT |
| 26 | `src/inference/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 27 | `src/inference/restore.py` | 🟢 NO EDIT |
| 28 | `src/utils/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 29 | `src/utils/image_utils.py` | 🟢 NO EDIT |
| 30 | `src/utils/logging_utils.py` | 🟢 NO EDIT |
| 31 | `api/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 32 | `api/app.py` | 🔵 OPTIONAL EDIT (only for production security) |
| 33 | `api/schemas.py` | 🟢 NO EDIT |
| 34 | `api/routes/__init__.py` | 🟢 NO EDIT (empty marker file) |
| 35 | `api/routes/data.py` | 🟢 NO EDIT |
| 36 | `api/routes/training.py` | 🟢 NO EDIT |
| 37 | `api/routes/inference.py` | 🟢 NO EDIT |
| 38 | `scripts/kaggle_setup.py` | 🟢 NO EDIT |
| 39 | `scripts/run_inference.py` | 🟢 NO EDIT |
| 40 | `notebooks/kaggle_training.ipynb` | 🟢 NO EDIT — ready to run; only needs Kaggle Secrets set up on the website (see Part 4) |

Auto-created empty folders you will also see appear (never create or edit
these yourself — the program makes them automatically the first time it runs):
`metadata/`, `checkpoints/`, `logs/`, `data/`, `output/`

---

# PART 1 — Files You MUST Edit (do these in order)

## 1. Create your `.env` file

📂 **File:** `.env.example` → you will create `.env`
🟡 **Status: EDIT REQUIRED**

This file does not get edited directly — you copy it first, then edit the copy.

**HOW TO DO IT:**
1. Open the **Terminal** app on your Mac (press ⌘+Space, type "Terminal", press Enter)
2. Type this command to go into your project folder (adjust the path to wherever you unzipped it):
   ```
   cd ~/Downloads/rrin_project
   ```
3. Type this exact command and press Enter:
   ```
   cp .env.example .env
   ```
4. Nothing will print — that means it worked. You now have a new file called `.env`.

**HOW TO CHECK IT WORKED:**
Type `ls -la` and press Enter. You should see a line that says `.env` in the list
(along with `.env.example` — both now exist side by side).

---

## 2. Edit your new `.env` file — add your Kaggle credentials

📂 **File:** `.env`
🟡 **Status: EDIT REQUIRED**

**Before you do this step, get your Kaggle key:**
1. Go to `kaggle.com` in your browser and sign in (create a free account if needed)
2. Click your profile picture (top right corner) → click **Settings**
3. Scroll down to the section called **API**
4. Click the button **Create New Token**
5. A file named `kaggle.json` will download to your Downloads folder
6. Open that file (double-click it, or drag it into TextEdit). It looks like this:
   ```
   {"username":"johnsmith123","key":"a1b2c3d4e5f6g7h8i9j0"}
   ```
   The text between quotes after `"username":` and `"key":` are the two
   values you need.

**NOW EDIT THE FILE:**

Open Terminal, make sure you're in the project folder, then type:
```
nano .env
```

You will see the file open inside Terminal. Find this line:
```
KAGGLE_USERNAME=PASTE_YOUR_KAGGLE_USERNAME_HERE
```
Use your arrow keys to move your cursor right after the `=` sign. Delete
the text `PASTE_YOUR_KAGGLE_USERNAME_HERE` (use Delete/Backspace) and type
your real username instead, so it looks like:
```
KAGGLE_USERNAME=johnsmith123
```

Then find this line just below it:
```
KAGGLE_KEY=PASTE_YOUR_KAGGLE_API_KEY_HERE
```
Replace it the same way with your real key:
```
KAGGLE_KEY=a1b2c3d4e5f6g7h8i9j0
```

Leave every other line in this file exactly as it is.

**HOW TO SAVE AND EXIT:**
Press **Control + O** (the letter O, not zero), then press **Enter** to confirm
the filename, then press **Control + X** to close the editor.

**HOW TO CHECK IT WORKED:**
Type this command to print the file back out:
```
cat .env
```
Look at the `KAGGLE_USERNAME=` and `KAGGLE_KEY=` lines — confirm they now
show YOUR real username and key, not the word "PASTE_YOUR...".

**Note on `HUGGINGFACE_TOKEN`:** Leave this line as-is for now
(`HUGGINGFACE_TOKEN=PASTE_YOUR_HF_TOKEN_HERE`) unless you intend to upload
your finished model to Hugging Face for hosting — that process is covered
in the separate `rrin_deploy` project, not here.

---

## 3. Edit `config.yaml` — tell the program where your datasets are

📂 **File:** `config.yaml`
🟡 **Status: EDIT REQUIRED**

This is the single most important file to edit. It tells the program
where to find your downloaded image datasets on your computer.

**HOW TO OPEN IT:**
```
nano config.yaml
```
(or, if you prefer a graphical editor: `open -e config.yaml` opens it in TextEdit,
or drag the file into VS Code if you have it installed)

**FIND THIS SECTION** near the top of the file:
```yaml
dataset_paths:
  eyepacs:    "data/eyepacs"          # EyePACS Kaggle DR dataset folder
  aptos:      "data/aptos"            # APTOS 2019 folder
  idrid:      "data/idrid"            # IDRiD folder
  messidor2:  "data/messidor2"        # MESSIDOR-2 (NEVER touched during training)
  rfmid:      "data/rfmid"            # RFMiD folder
  odir:       "data/odir"             # ODIR-5K folder
  stare:      "data/stare"            # STARE vessel dataset
  drive:      "data/drive"            # DRIVE vessel dataset
```

**WHAT TO REPLACE EACH WITH:**

For every dataset line, the text inside the quotation marks `" "` must
become the REAL folder path on your Mac where that dataset's images live.

If you have **not** downloaded a particular dataset, change its value to
an empty pair of quotes `""` so the program skips it instead of erroring.

Example — if you ran `python scripts/kaggle_setup.py` and it downloaded
everything into a folder called `data` inside your project (the default),
you likely don't need to change anything for those lines — they
already point to `data/eyepacs`, `data/aptos`, etc.

But if you instead downloaded datasets to, say, your Desktop, you would
change it to a full path, for example:
```yaml
dataset_paths:
  eyepacs:    "/Users/yourname/Desktop/datasets/eyepacs"
  aptos:      "/Users/yourname/Desktop/datasets/aptos"
  idrid:      ""
  messidor2:  ""
  rfmid:      "/Users/yourname/Desktop/datasets/rfmid"
  odir:       ""
  stare:      ""
  drive:      ""
```
(The example above leaves `idrid`, `messidor2`, `odir`, `stare`, and
`drive` empty because they were not downloaded — that's perfectly fine,
the program just trains on whatever datasets you do provide.)

**To find the exact full path of a folder on your Mac:**
Open Finder, navigate to the dataset folder, right-click it, hold the
**Option** key, and the menu will change to show **"Copy [foldername] as Pathname"**.
Click that, then paste it between the quotation marks in `config.yaml`.

**OPTIONAL — settings you may also want to change in this same file:**

If your Mac does not have a dedicated NVIDIA graphics card (most Macs
don't — Apple Silicon and Intel Macs use different chips that PyTorch
treats as CPU-only for this project), training will be extremely slow
locally. Find this line:
```yaml
batch_size:          4      # How many images per training step  (reduce to 2 if GPU runs out of memory)
```
You can lower it to `2` to reduce memory use, but this will not fix
slow CPU speed — for that, use the free Kaggle GPU instead (see Part 4
below). This is why we recommend training on Kaggle rather than your own Mac.

**HOW TO SAVE AND EXIT (if using nano):**
Press **Control + O**, then **Enter**, then **Control + X**.

**HOW TO CHECK IT WORKED:**
Type:
```
cat config.yaml | head -25
```
Read through the `dataset_paths:` section that prints and confirm every
path matches a real folder that exists. You can double-check a path is
real by typing (replace with your actual path):
```
ls "/Users/yourname/Desktop/datasets/eyepacs"
```
If this prints a list of image files, the path is correct. If it says
"No such file or directory", the path is wrong — fix it and save again.

---

# PART 2 — Files You CAN Edit, But Don't Have To (Optional)

## 4. `main.py` — change default training behaviour

📂 **File:** `main.py`
🔵 **Status: OPTIONAL EDIT**

You normally control this file using command flags when you run it
(for example `python main.py --batch-size 2`), so you do **not** need
to open or edit this file at all for normal use.

Only open this file if you want to change the *default* values so you
don't have to type flags every time. For example, to make 100 epochs
the default instead of 200, you would find this line:
```python
parser.add_argument("--epochs",        type=int,   default=NUM_EPOCHS,
                    help=f"Total training epochs (default: {NUM_EPOCHS})")
```
and change `default=NUM_EPOCHS` to `default=100`. This is genuinely
optional — skip this entire section if you're fine using flags.

**HOW TO CHECK A CHANGE WORKED:** Run `python main.py --help` and read
the printed default value next to `--epochs`.

---

## 5. `api/app.py` — restrict API access (only for public deployment)

📂 **File:** `api/app.py`
🔵 **Status: OPTIONAL EDIT — only relevant if you expose this API publicly**

If you are only running this on your own Mac for your own use, skip
this file entirely. It only matters if you deploy this specific API
(not the separate `rrin_deploy` project) to a public server.

**FIND THIS:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # In production, replace with your specific domain
```
**REPLACE WITH** (example — replace the URL with your actual frontend's address):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
```

---

## 6. `src/dataset_parsers.py` — only if adding a brand-new dataset type

📂 **File:** `src/dataset_parsers.py`
🔵 **Status: OPTIONAL EDIT — advanced, skip unless you have a dataset not already supported**

The project already understands EyePACS, APTOS, IDRiD, MESSIDOR-2,
RFMiD, ODIR, STARE, and DRIVE. If you are only using these eight
datasets, do not touch this file. This is mentioned only for completeness.

---

# PART 3 — Files You Should NEVER Edit (The AI Engine Itself)

Everything below is the actual program code that makes the AI work —
the neural network, the training loop, the math behind image
restoration. You never need to open these. They work automatically once
Parts 1 and 2 above are done. They are listed here only so you know
they exist and can safely ignore them.

**Empty marker files** (these are intentionally blank — they just tell
Python "this folder is part of the program"):
`src/__init__.py`, `src/models/__init__.py`, `src/training/__init__.py`,
`src/evaluation/__init__.py`, `src/inference/__init__.py`,
`src/utils/__init__.py`, `api/__init__.py`, `api/routes/__init__.py`

**Core data pipeline:**
- `src/config.py` — reads `config.yaml` and makes its values available to the rest of the program
- `src/database.py` — keeps a running record of every image and its quality score
- `src/quality_scoring.py` — automatically scores image sharpness, brightness evenness, and reflections
- `src/degradation.py` — artificially adds blur, glare, and noise to clean images during training
- `src/splits.py` — divides images into training/validation/test groups by patient, to keep results honest
- `src/datasets.py` — feeds batches of images into the model during training

**The AI model itself:**
- `src/models/generator.py` — the actual restoration network (a U-Net with attention)
- `src/models/discriminator.py` — the "critic" network used during training to judge realism
- `src/models/losses.py` — the mathematical formulas that measure how good a restoration is

**Training machinery:**
- `src/training/train.py` — runs the training and validation loops
- `src/training/checkpoints.py` — saves and loads model progress to disk
- `src/training/domain_adaptation.py` — an advanced fine-tuning stage for real-world images

**Testing and using the trained model:**
- `src/evaluation/metrics.py` — calculates final accuracy scores after training
- `src/inference/restore.py` — loads a trained model and restores a new image
- `src/utils/image_utils.py` — helper functions for loading/saving/resizing images
- `src/utils/logging_utils.py` — writes progress messages to the screen and to log files

**The web API:**
- `api/schemas.py` — defines the shape of data sent to/from the API
- `api/routes/data.py`, `api/routes/training.py`, `api/routes/inference.py` — the actual API endpoints

**Command-line helper scripts:**
- `scripts/kaggle_setup.py` — automatically downloads datasets using your `.env` Kaggle credentials
- `scripts/run_inference.py` — restores an image from the command line without starting the API

---

# PART 4 — The Kaggle Notebook (Recommended — This Is How You Should Train)

📂 **File:** `notebooks/kaggle_training.ipynb`
🟢 **Status: NO EDIT NEEDED — every cell is already written and ready to run**

This is the recommended way to train, especially if you do not want to
store any image datasets on your own Mac at all. The notebook fetches
every dataset directly through Kaggle's own API (a library called
`kagglehub`) the moment it runs — it does not rely on you manually
searching for and attaching datasets through Kaggle's website interface,
and it does not require typing a single folder path anywhere.

Concretely, this means: you never touch `config.yaml` on your Mac for
this workflow. The notebook builds its own separate internal config
file (`config_kaggle.yaml`) automatically, every time it runs, by
calling `kagglehub.dataset_download(...)` and
`kagglehub.competition_download(...)` and writing whatever real paths
those calls return directly into that file. `config.yaml` in this
project folder only matters if you later decide to also train locally
on your own Mac — for a Kaggle-only workflow you can ignore it entirely.

**What you DO need to do — all of it happens on Kaggle's website, not
by editing any file on your Mac:**

1. Accept the two competition rules (EyePACS, APTOS) in your browser, once
2. Upload this project as a Kaggle Dataset named `rrin-code` (your code only — this is the one thing the Kaggle API cannot fetch on its own, since it is private to you)
3. Optionally add two Hugging Face secrets inside the notebook editor (`HUGGINGFACE_TOKEN` and `HF_MODEL_REPO`) if you want the trained model auto-uploaded to Hugging Face at the end — you do **not** add Kaggle username/key secrets, because `kagglehub` authenticates using your Kaggle login session automatically
4. Toggle the GPU and Internet access on in the notebook's Session options
5. Click **Run All**

The complete click-by-click walkthrough for every one of those five
items — including exactly where to click on Kaggle's website — lives in
**`DEPLOYMENT_GUIDE.md` inside the `rrin_deploy` folder, Stage 1**.

**What the notebook does automatically once you click Run All:**

| Cell | What it does |
|---|---|
| CELL 1 | Installs `kagglehub` and the other required packages |
| CELL 2 | Copies your `rrin-code` upload into the working directory, checks the GPU is active |
| CELL 3 | Loads `HUGGINGFACE_TOKEN` and `HF_MODEL_REPO` from Kaggle Secrets if you added them — prints a notice and continues if you did not. No Kaggle credentials needed here. |
| CELL 4 | Calls `kagglehub.dataset_download(...)` for RFMiD and ODIR-5K, and `kagglehub.competition_download(...)` for APTOS and EyePACS — then writes the four real returned paths straight into `config_kaggle.yaml` for you |
| CELL 5 | Ingests every image into the metadata database, scores image quality, splits into train/val/test |
| CELL 6 | Trains the model (12–20 hours) |
| CELL 7 | Evaluates the trained model on the held-out test set |
| CELL 8 | Saves a sample before/after restoration image so you can see how it is performing |
| CELL 9 | Uploads `best.pt` to Hugging Face automatically — or, if you skipped the optional Hugging Face Secrets, prints a short message and does nothing instead of failing |

**A note on dataset folder structures:** you might wonder which exact
subfolder inside a downloaded dataset (e.g. `train_images/` inside
APTOS, or `train/` inside EyePACS) needs to be referenced. You never
need to figure this out — CELL 4 points at the top-level folder
`kagglehub` returns, and the program searches every subfolder inside it
automatically, no matter how deeply nested, to find every image file.
There is no folder name for you to choose or guess.

**HOW TO CHECK IT WORKED:** After CELL 4 finishes, its output prints
four lines like `-> /root/.cache/kagglehub/datasets/.../versions/3`,
one per dataset. As long as none of the four lines show an error, the
data is ready and CELL 5 can proceed.

---

# Final Checklist Before You Run Anything

Go through this list in order. Do not skip ahead.

- [ ] `.env` file exists (check with `ls -la` — see Step 1)
- [ ] `.env` contains your real Kaggle username and key, not placeholder text (check with `cat .env` — see Step 2)
- [ ] `config.yaml` dataset paths point to real folders on your Mac, or are set to `""` if not downloaded (see Step 3)
- [ ] You ran `./setup.sh` once to install all required packages (see below)
- [ ] You ran `python scripts/kaggle_setup.py` once to download datasets (or you are using the Kaggle notebook instead, which downloads them for you automatically)

**Running setup for the first time (only needed once):**
```
cd ~/Downloads/rrin_project
chmod +x setup.sh
./setup.sh
```

**Starting training on your own Mac (slow without a dedicated GPU):**
```
source .venv/bin/activate
python main.py
```

**Recommended instead — training for free on Kaggle (fast GPU):**
See `DEPLOYMENT_GUIDE.md` inside the `rrin_deploy` folder, Stage 1,
for the complete step-by-step walkthrough of running this on Kaggle.

---

# Troubleshooting

| Problem | What it means | Fix |
|---|---|---|
| `cat .env` shows "No such file or directory" | You skipped Step 1 | Run `cp .env.example .env` again |
| Program says "Kaggle credentials not found" | Step 2 wasn't saved correctly | Re-open with `nano .env`, check both KAGGLE lines have real values, save again |
| Program says "No images in the database" | A path in `config.yaml` is wrong or empty | Re-check Step 3 — use `ls "your/path/here"` to confirm the folder is real |
| `nano` opens but you can't find a line | nano shows the whole file at once | Use arrow keys to scroll; the line numbers in this guide match the file top-to-bottom |
| Can't see `.env` in Finder | This is normal Mac behaviour | Press ⌘+Shift+. in Finder, or just keep using Terminal/`nano` |
| `chmod: setup.sh: No such file or directory` | You're in the wrong folder | Type `pwd` to see where you are, then `cd` to the correct project folder |
