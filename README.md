# ImmunoVLM

ImmunoVLM aligns histopathology patches, spatial gene expression, and clinical subtype language in a shared 1024-dimensional space. Its training objective combines distance-dependent image–gene contrast, topology distribution matching, spatial-neighborhood augmentation, and image–language alignment for lupus nephritis, inflammatory bowel disease, rheumatoid arthritis, and mixed inflammatory tissue cohorts.

## Environment

The reported environment uses Python 3.10.12, PyTorch 2.1.0, CUDA 12.1, Transformers 4.36.0, Scanpy 1.9.6, scikit-learn 1.3.2, and R 4.3.2 with clusterProfiler 4.12.0.

```bash
conda env create -f environment.yml
conda activate immunovlm
pip install -e .
```

The container path uses the same PyTorch and CUDA versions.

```bash
docker build -t immunovlm:0.1.0 .
```

## Data

Verified source links are collected in `datasets.txt`.

| Cohort | Source | Access and license | Role |
|---|---|---|---|
| AMP-LN and AMP-RA | ImmPort SDY997–SDY999 | Registered ImmPort access; repository terms apply | Primary training and evaluation |
| GCA-IBD | Gut Cell Atlas | Public atlas; constituent study terms apply | Primary training and evaluation |
| HEST-AI | HEST-1k | CC BY-NC-SA 4.0 | Primary training and evaluation |
| Childhood LN | Public CosMx cohort referenced by the study | External evaluation only | External validation |
| Broad Crohn cohort | Controlled source referenced by the study | Controlled access | External validation |

HEST-1k exceeds 2 TB in full. Query only the kidney, colon, and joint profiles needed for HEST-AI. The repository does not redistribute tissue images, molecular measurements, or participant-level metadata. Each processed manifest records patient, section, platform, spatial provenance, paths, tissue fraction, and subtype. Run preparation separately for each authorized source and retain its resulting manifest digest with the experiment outputs.

Expected processed records use one image patch and one 500-HVG vector per tissue spot. Images are RGB PNG files at 224 × 224 pixels. Expression vectors are one-dimensional float32 NPY files. Coordinates are stored in the manifest in platform-native spatial units. Spots with less than 50% tissue or fewer than 200 detected genes are excluded.

## Preparation

The preprocessing modules expose library-size normalization, log transformation, Seurat-v3-style binned variance selection, Macenko stain normalization, Visium patch extraction, spot quality control, and reference-based nonnegative deconvolution. Build the final CSV manifest with these fields:

```text
sample_id,patient_id,section_id,disease,subtype,image_path,expression_path,coordinate_x,coordinate_y,tissue_fraction,platform,native_spatial
```

The subtype vocabulary is fixed to `LN_I` through `LN_VI`, `IBD_CD`, `IBD_UC`, `IBD_IC`, `RA_LM`, `RA_DM`, and `RA_PI`. Patient identifiers must be study-local, de-identified values. Do not place names, contact details, medical record numbers, dates of birth, or free clinical notes in manifests.

## Training

Launch the primary study with four distributed workers:

```bash
torchrun --standalone --nproc_per_node=4 -m immunovlm.commands.train \
  --study settings/studies/main.yaml \
  --manifest data/manifests/primary.csv \
  --data-root data \
  --output outputs/primary
```

Run each of five patient-level folds and each of the ten reported seeds by overriding `fold` and `seed`:

```bash
torchrun --standalone --nproc_per_node=4 -m immunovlm.commands.train \
  --study settings/studies/main.yaml \
  --manifest data/manifests/primary.csv \
  --data-root data \
  --output outputs/fold_1_seed_123 \
  fold=1 seed=123
```

The study configuration preserves the reported values: 100 epochs, AdamW, learning rate 5e-4, weight decay 0.01, cosine decay, FP16 dynamic scaling, gradient norm 1.0, and 512 spot pairs per device. The vision backbone is frozen for contrastive pretraining. Disease-specific supervised training should unfreeze it.

The manuscript states both 512 pairs per GPU on four GPUs with four accumulation steps and an effective batch of 2,048. Those values are arithmetically inconsistent: the product is 8,192. The configuration preserves every stated primitive parameter and does not silently rewrite one of them. For an effective batch of 2,048, set `optimizer.gradient_accumulation=1`; document that override with the run output.

Available study files cover removal of topology loss, fixed temperature, removal of graph augmentation, removal of language alignment, image-only input, gene-count sensitivity, and batch-size sensitivity.

## Evaluation

The evaluation command consumes an NPZ file containing `labels` and `probabilities` arrays:

```bash
immunovlm-evaluate \
  --predictions outputs/primary/predictions.npz \
  --output outputs/primary/metrics.json \
  --resamples 2000
```

It reports macro AUROC with a stratified bootstrap interval, macro F1, accuracy, one-versus-rest macro specificity, Cohen’s kappa, expected calibration error with ten bins, maximum calibration error, and multiclass Brier score. Additional modules implement paired DeLong comparison, Holm–Bonferroni correction, Cochran’s Q, I², decision curves, category-free NRI, Moran’s I, and spatial-embedding correlation.

The primary reported targets are mean AUROC 0.741 for AMP-LN, 0.749 for GCA-IBD, 0.801 for HEST-AI, and 0.762 for AMP-RA across ten seeds. External targets are 0.698 for childhood LN and 0.723 for the Crohn cohort. Run-to-run acceptance should use the paper’s bootstrap intervals and the identical patient-level folds; individual runs are not expected to equal the ten-seed mean.

## Compute

The reported system has four NVIDIA A100 80GB GPUs, two AMD EPYC 7763 64-core processors, and 512 GB RAM. Contrastive pretraining takes about 18 hours. Downstream classification adds about two hours per disease. A full five-fold, ten-seed run requires substantial scheduled cluster time and dataset-dependent storage beyond the model artifacts. Deterministic CUDA execution is enabled with the cuBLAS workspace configuration used in the study.

## Project layout

`code/immunovlm/corpora` contains ingestion and preprocessing. `code/immunovlm/encoders` contains the three modality branches. `code/immunovlm/objectives` contains equations 2–6. `code/immunovlm/optimization` contains distributed training, schedules, and atomic state persistence. `code/immunovlm/assessment` contains the statistical protocol. `settings/cohorts` records cohort provenance, while `settings/studies` records the primary and sensitivity runs.

## Privacy and scope

Only de-identified public or properly authorized controlled-access records belong in this pipeline. Training logs use study-local identifiers and aggregate losses. The code performs retrospective computational analysis and is not a medical device. Outputs require clinical and pathology review before any translational use.
