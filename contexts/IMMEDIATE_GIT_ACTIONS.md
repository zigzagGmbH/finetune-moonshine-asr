# IMMEDIATE ACTIONS — Git, Fork & Context Handoff

**Situation:**
- Worked code (all changes) lives on **ziggie** at `~/Projects/finetune-moonshine-asr/`
- You cloned Pierre's repo, created a branch, made changes — but can't push to his remote
- The Mac has an original clone (untouched) at `/Users/saurabhdatta/Documents/Projects/finetune-moonshine-asr/`
- The next Claude agent can access the Mac filesystem but NOT ziggie directly
- You (Saurabh) are the bridge — running commands on ziggie via SSH, results visible to agent on Mac

---

## STEP 1: Fork Pierre's repo on GitHub (do this on your Mac browser)

1. Go to: https://github.com/pierre-cheneau/finetune-moonshine-asr
2. Click **Fork** → fork to your **dattazigzag** org (or `dattasaurabh82` personal — your choice)
3. Note the new URL, e.g.: `git@github-dattazigzag:dattazigzag/finetune-moonshine-asr.git`

---

## STEP 2: On ziggie — set up the fork remote and push

```fish
cd ~/Projects/finetune-moonshine-asr

# Check current state
git status
git branch
git remote -v
# (should show origin → pierre-cheneau/finetune-moonshine-asr)

# Add your fork as a new remote
# Use your SSH alias (github-dattazigzag or github-dattasaurabh82)
git remote rename origin upstream
git remote add origin git@github-dattazigzag:dattazigzag/finetune-moonshine-asr.git

# Verify
git remote -v
# origin    → your fork
# upstream  → Pierre's original

# Stage and commit all changes
git add -A
git status  # review what's being committed

git commit -m "feat: German (DE) fine-tuning pipeline

- Added prepare_german_dataset.py for MLS German data prep
- Added mls_cv_german_no_curriculum.yaml config for RTX 5090
- Patched train.py for bf16 support and safe config key access
- Pinned datasets<4.0, transformers<4.50 for compatibility
- First model: 36.7% WER on MLS German test set (10k steps)
- See contexts/moonshine_de_context.md for full details"

# Push to your fork
git push -u origin your-branch-name
# (replace 'your-branch-name' with whatever branch you're on)
```

---

## STEP 3: Copy context doc INTO the repo on ziggie (before committing)

```fish
# Create contexts dir in the repo
mkdir -p ~/Projects/finetune-moonshine-asr/contexts/

# Copy the context doc (you downloaded it from Claude to your Mac)
# Option A: scp from Mac to ziggie
# On your Mac:
scp ~/Downloads/moonshine_de_context.md zigzagadmin@192.168.178.160:~/Projects/finetune-moonshine-asr/contexts/

# Option B: or just create it directly on ziggie by pasting
# (the file content was already provided by Claude)
```

Then amend the commit:
```fish
cd ~/Projects/finetune-moonshine-asr
git add contexts/moonshine_de_context.md
git commit --amend --no-edit
git push -f origin your-branch-name
```

---

## STEP 4: Clone the fork to your Mac (replace the old Pierre clone)

```fish
# On your Mac
cd ~/Documents/Projects

# Rename the old Pierre clone
mv finetune-moonshine-asr finetune-moonshine-asr-pierre-original

# Clone your fork
git clone git@github-dattazigzag:dattazigzag/finetune-moonshine-asr.git
cd finetune-moonshine-asr

# Checkout your working branch
git checkout your-branch-name
```

Now the next agent can read everything via the Mac filesystem at:
`/Users/saurabhdatta/Documents/Projects/finetune-moonshine-asr/`

Including `contexts/moonshine_de_context.md` which has ALL the details.

---

## STEP 5: What to tell the next agent

Start a new chat with something like:

> I'm working on fine-tuning Moonshine ASR for German. Full context is at:
> `/Users/saurabhdatta/Documents/Projects/finetune-moonshine-asr/contexts/moonshine_de_context.md`
>
> Read that file first — it has everything: what we did, all gotchas, pinned deps,
> server details, file locations, and detailed next steps.
>
> The model is trained and saved on our server (ziggie) at `/data/results-moonshine-de/final/`.
> I need help with [specific next step from the context doc].
>
> I'll be your bridge to ziggie — I run commands there via SSH and paste results back.

---

## KEY DETAIL FOR NEXT AGENT

The next agent can ONLY see the Mac filesystem. It CANNOT SSH into ziggie.
The workflow is:
1. Agent reads code/configs/context from Mac filesystem
2. Agent tells Saurabh what commands to run on ziggie
3. Saurabh runs them, pastes output back
4. Agent analyzes and gives next steps

This is the same pattern used for all ziggie work (documented in ziggie_setup_assistance).

---

## FILES THE NEXT AGENT NEEDS TO READ FIRST

1. `contexts/moonshine_de_context.md` — **THE** context doc (everything: gotchas, commands, roadmap)
2. `configs/mls_cv_german_no_curriculum.yaml` — training config
3. `scripts/prepare_german_dataset.py` — data prep script
4. `train.py` — training script (with our bf16 patches)

All of these will be in the Mac clone after Step 4 above.
