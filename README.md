# CYP3A4 structure prediction by Boltz-2 consensus

Code and method report for the OpenADMET CYP Structure Prediction Track, participant `asidsal11`.

20 blinded CYP3A4 ligand complexes were predicted with Boltz-2 (open-source `boltz` 2.2.1, no templates, no restraints). Up to 80 samples per ligand were generated, filtered for correct stereochemistry and PoseBusters validity, clustered, and the representative of the heaviest cluster was submitted.

## What is distinctive

- **Uncertainty-driven sampling.** Sampling was done in two rounds. After round 1 (30 samples per ligand), the weight of the winning cluster against the runner-up was used as an uncertainty readout, and extra seeds were spent only on the 9 ligands whose pools were ambiguous or had lost many samples to the stereochemistry filter. The same readout is reported per ligand in `results/selection_report_v7.csv`, and three ligands remain flagged as unresolved.
- **Stereochemistry is filtered before ranking.** Boltz returned the wrong stereoisomer for a large share of samples for some ligands (59 of 80 for OCNT-2312966), so each sample is checked against the official SMILES and discarded if it does not match.
- **Heme supplied as a chemical component.** Passing heme as the component HEM, not as a SMILES string, gave iron to ligand-nitrogen distances of 2.0 to 2.3 angstrom and Cys442 sulfur to iron distances of 2.28 to 2.40 angstrom, against about 2.8 and 3.0 angstrom with SMILES heme in earlier runs.
- **Consensus over confidence.** For two ligands the voted pose is more than 5 angstrom from the single highest-confidence sample, because most samples agree on a different pose.

## Files

| File | Content |
|-|-|
| `cyp3a4_boltz_consensus.py` | The full pipeline as Colab blocks: setup, sampling, adaptive second round, selection, packaging |
| `results/selection_report_v7.csv` | Per-ligand sample counts, cluster weights and the analog tie-break log |
| `LICENSE` | MIT |

## Reproducing

1. Open a Google Colab notebook with a GPU runtime and paste the blocks of `cyp3a4_boltz_consensus.py` into separate cells, or upload the file and run it block by block.
2. Run blocks 1 to 3 (round 1), then block 5 once to inspect the round-1 cluster statistics, then block 4 for the extra seeds, then blocks 5 and 6 for the final selection and the zip.
3. Block 4 lists the ligands that received extra seeds. These were chosen from the round-1 statistics and are hard-coded here as the record of what was run.
4. Boltz was run with the `no_kernels` option because the default GPU kernels failed on the Blackwell GPU used. Sampling is seeded, but GPU results are not guaranteed to be bit-identical across hardware.

## Method report

### 1. Summary

Each of the 20 complexes was predicted with open-source Boltz-2 (no templates, no restraints). Between 30 and 80 samples were generated per ligand across several seeds, filtered for correct ligand stereochemistry and PoseBusters validity, and clustered by symmetry-corrected ligand RMSD in a common pocket frame. The submitted pose is a representative of the most populated, confidence-weighted cluster. The heme was supplied as the chemical component HEM, not as a SMILES string. No proprietary data were used and no public PDB structures were used as templates.

This report describes a method, not a validated improvement. The reference structures are blinded, so nothing here has been scored against ground truth at the time of writing.

### 2. Inputs

- Protein: human CYP3A4, UniProt P08684, full length, 503 residues, one chain (chain A).
- Ligand: the official SMILES for each of the 20 IDs from the challenge dataset, including stereochemistry where specified (chain B).
- Cofactor: heme as the chemical component HEM (chain C).
- MSA: computed once with the ColabFold MSA server (Boltz option use_msa_server) for the protein chain and reused for all 20 ligands and all seeds.
- No templates, no pocket or contact constraints, no affinity prediction.

### 3. Sampling

| Item | Value |
|-|-|
| Model | Boltz-2, open-source package boltz 2.2.1 |
| Samples per run | diffusion_samples 10 |
| Recycling steps | 3 |
| Sampling steps | 200 |
| Output format | pdb |
| Kernel option | no_kernels (the default GPU kernels failed on this GPU) |
| Seeds | 1, 2, 3 for all 20 ligands (30 samples each) |
| Extra seeds | 4 to 8 for 9 ligands (80 samples each): OCNT-2395342, 2395328, 2313429, 2395460, 2395457, 2395712, 2313038, 2312966, 2395415 |
| Total | 1,050 samples |
| Hardware | NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), Google Colab |
| Software | torch 2.15.0, numpy 1.26.4, Python 3.11 for Boltz; Python 3.13 with PoseBusters 0.6.5, RDKit 2026.9.1, gemmi 0.7.5 for selection |

Ten samples took about 49 seconds once the model was loaded. The 9 extra-seed ligands were chosen after the first round because their winning cluster was small or because many samples were lost to the stereochemistry filter (section 4). The choice used only sample statistics, not any scoring against reference structures.

### 4. Filtering and selection

For each ligand:

1. Parse each sample. Chain B is the ligand and chain C is the heme (Boltz labels both non-protein residues LIG, so they were identified by chain).
2. Superpose all samples, for all ligands, onto one reference frame using the C-alpha atoms of residues within 14 angstrom of the heme iron in the first parsed sample.
3. Assign bond orders from the official SMILES (RDKit AssignBondOrdersFromTemplate) and reject any sample whose 3D stereochemistry does not match the SMILES.
4. Run PoseBusters (dock configuration, protein plus heme as conditioning structure). Keep only passing samples. If none pass, all samples are kept.
5. Compute the pairwise ligand heavy-atom RMSD between all remaining samples, minimized over the symmetry permutations of the ligand graph (capped at 500 permutations).
6. Weight each sample by exp((c - c_max) / 0.05), where c is the Boltz confidence score.
7. Cluster greedily. A sample's density is the summed weight of all samples within 2.5 angstrom of it. The densest sample defines the first cluster and removes its members, then the next densest defines the second, and so on, up to five clusters.
8. Submit the defining sample of the heaviest cluster.

An analog tie-break was also implemented. A ligand's top cluster would be replaced only if another cluster held at least 60 percent of the top cluster's weight and agreed at least 1.5 angstrom better with the poses of already-confident close analogs (maximum common substructure covering at least 60 percent of the smaller ligand and at least 10 atoms). In this submission the rule did not change any ligand.

### 5. Per-ligand sampling statistics

Usable samples are those that parsed and matched the SMILES stereochemistry. Weights are the confidence-weighted cluster masses described above.

| Ligand | Samples generated | Usable | PoseBusters pass | Top cluster size | Top weight | Second weight |
|-|-:|-:|-:|-:|-:|-:|
| OCNT-2395436 | 30 | 30 | 29 | 23 | 22.18 | 4.56 |
| OCNT-2395503 | 30 | 30 | 29 | 29 | 28.17 | 0.00 |
| OCNT-2395342 | 80 | 80 | 50 | 25 | 21.26 | 10.20 |
| OCNT-2395712 | 80 | 80 | 49 | 25 | 23.50 | 18.08 |
| OCNT-2313429 | 80 | 80 | 61 | 44 | 40.23 | 5.38 |
| OCNT-2395464 | 30 | 30 | 28 | 28 | 25.83 | 0.00 |
| OCNT-2313397 | 30 | 30 | 30 | 30 | 29.12 | 0.00 |
| OCNT-2395415 | 80 | 60 | 56 | 56 | 52.60 | 0.00 |
| OCNT-2395645 | 30 | 30 | 25 | 23 | 21.81 | 0.97 |
| OCNT-2395789 | 30 | 30 | 28 | 28 | 27.35 | 0.00 |
| OCNT-2312554 | 30 | 28 | 26 | 26 | 25.34 | 0.00 |
| OCNT-2395457 | 80 | 80 | 76 | 49 | 45.86 | 7.53 |
| OCNT-2313038 | 80 | 56 | 47 | 39 | 34.94 | 5.70 |
| OCNT-2313300 | 30 | 30 | 28 | 23 | 22.18 | 2.74 |
| OCNT-2395460 | 80 | 79 | 56 | 34 | 31.34 | 12.12 |
| OCNT-2315379 | 30 | 30 | 28 | 26 | 24.34 | 1.91 |
| OCNT-2535422 | 30 | 30 | 25 | 21 | 19.81 | 1.99 |
| OCNT-2535519 | 30 | 30 | 28 | 24 | 22.66 | 3.26 |
| OCNT-2395328 | 80 | 79 | 51 | 17 | 15.15 | 12.09 |
| OCNT-2312966 | 80 | 21 | 14 | 5 | 4.62 | 3.46 |

Samples were lost mainly to the stereochemistry filter. Among the ligands with a defined stereocentre, Boltz returned the wrong stereoisomer in a substantial fraction of samples: 59 of 80 for OCNT-2312966 (about 74 percent), 24 of 80 for OCNT-2313038 (30 percent), 20 of 80 for OCNT-2395415 (25 percent) and 2 of 30 for OCNT-2312554. The single missing sample for OCNT-2395328 and OCNT-2395460 failed the SMILES template match.

Three ligands remain ambiguous after sampling: OCNT-2395328 (two clusters of weight 15.2 and 12.1), OCNT-2395712 (23.5 and 18.1) and OCNT-2312966 (only 14 PoseBusters-passing samples, weights 4.6 and 3.5). For these, the submitted pose is the heaviest cluster but there is little evidence separating it from the second.

For OCNT-2313429 and OCNT-2395457, the submitted pose is 6.53 and 5.05 angstrom from the single highest-confidence sample, because most samples agree with a different pose.

### 6. Heme geometry

With heme supplied as the chemical component HEM, the submitted poses show iron to ligand-nitrogen distances of 2.0 to 2.3 angstrom for the 11 ligands whose nitrogen sits within 2.6 angstrom of the iron, and Cys442 sulfur to iron distances of 2.28 to 2.40 angstrom across all 20 files. In an earlier set of Boltz-2.1 runs by the same participant, where heme was supplied as a SMILES string, the same distances were about 2.8 to 2.9 angstrom and 3.0 to 3.1 angstrom. Coordination was not forced by any restraint. Which ligands coordinate the iron in the real structures is not known to the author, and a nitrogen heterocycle does not imply coordination.

### 7. Validation of the submission file

- One PDB per official ID, 20 files, ligand residue name LIG, heme residue name HEM, three chains (A protein, B ligand, C heme), unique residue and atom numbering.
- The checks of the tutorial validation function were reproduced and passed (ID set, LIG residue, chain count, numbering, ligand connectivity against the SMILES). The official function itself should be run on the final zip before upload.
- Final structures: PoseBusters (dock configuration) passes for 20 of 20, and ligand stereochemistry and bond orders match the official SMILES for 20 of 20.

### 8. Relation to earlier submissions

The same participant's earlier leaderboard entries (LDDT-PLI 0.194, 0.363 and 0.369 on 2026-10-01, 10-09 and 10-12) came from earlier single-sample and reranked Boltz-2.1 predictions with SMILES-defined heme. The Oct 9 entry appears to be sample 0 of those runs and the Oct 12 entry a contact-count rerank, but the archive-to-score mapping was not recorded, so this is an inference. The difference between the last two (0.006) is within the per-ligand spread of the metric (about 0.04), so no conclusion is drawn about the reranking. The present submission differs in heme representation, number of samples, filtering and selection rule, and is not a continuation of that rerank.

### 9. Data and disclosures

- Proprietary data: none.
- External data and models: Boltz-2 pretrained weights, ColabFold MSA server, UniProt P08684 sequence, official challenge ligand SMILES.
- Public PDB structures: not used.
- Code: this repository (`cyp3a4_boltz_consensus.py`), MIT licensed.
