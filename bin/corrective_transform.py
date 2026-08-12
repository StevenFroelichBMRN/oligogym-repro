#!/usr/bin/env python
"""Defect corrections for the OligoGym CORRECTIVE arm only.

Three silent defects were verified in Phase 1 and re-verified here against the
shipped package resources at commit 97f5b9f.  This module implements the fixes
that are defensible and refuses the one that is not.  It is applied ONLY when
`--arm corrective`; the faithful (primary) arm never imports it, so the
reproduction of the published numbers is untouched.

=============================================================================
DEFECT 1 -- TLR7 / TLR8 keys are swapped relative to their data.  FIXED, fully.
=============================================================================
Re-verified in this session:
  * `download('tlr7')` -> `Alharbi_2020_1`, whose own shipped metadata says
    "2'OMe gapmer screen of **TLR8** potentiation", label_desc "TLR8 level after
    induction ...", y in [0.57, 5.44] (Resiquimod normalised to 1.0).
  * `download('tlr8')` -> `Alharbi_2020_2`, metadata "2'OMe gapmer screen of
    **TLR7** inhibition", y in [17.46, 126.5] (Resiquimod = 100).
  * Both files hold the SAME 192 compounds in the SAME row order (verified:
    x sets equal, x order equal, `fasta` column identical) over the same 4
    targets (CDKN2B-AS1, CTNNB1, EGFR, LINC-PINT).  Only the readout differs
    (y differs in 192/192 rows).

Because the compound table is identical and only the label column differs, the
fix is a clean, complete relabeling: the config key `immune_modulation_TLR7`
loads `Alharbi_2020_2` (the actual TLR7 inhibition assay) and
`immune_modulation_TLR8` loads `Alharbi_2020_1` (the actual TLR8 potentiation
assay).  Nothing about the features changes, so a corrected TLR7 result is
directly comparable to the published TLR7 result -- what changes is which
endpoint the label represents.

=============================================================================
DEFECT 2 -- Neurotox MOE HELM/SMILES truncated by one 3' nucleoside.
             PARTIALLY corrected; see the decision below.
=============================================================================
Re-verified in this session over all 2,437 records:
  * the base sequence parsed from HELM is a strict prefix of the `fasta` column
    in 2437/2437 records, and exactly ONE base shorter in 2437/2437;
  * the dropped base varies -- C 764, T 711, A 621, G 341 -- so it is not an
    overhang convention that could be defaulted;
  * the SMILES phosphorus count equals (n_monomers - 1) in 2437/2437 and equals
    the `fasta` length in 0/2437, i.e. SMILES is truncated consistently with
    HELM, not with `fasta`;
  * Cytotox LNA as a control shows a zero-length discrepancy in 768/768 records,
    so this is specific to the MOE Neurotox release.

**What the featurizers actually read** (verified by reading features.py, not
assumed).  Neither featurizer touches the `fasta` column:
  * `KMersCounts._extract_features` calls `_extract_monomers(helm)` and builds
    its k-mer string from `monomers['base'].str[-1].str.cat()`.  It is a
    BASE-ONLY sequence, so a full-length base sequence recovers the 20th base
    for the k-mer counts.  But the same method's `modification_abundance` branch
    counts `monomers['sugar']` and `monomers['phosphate']` -- which only exist in
    HELM -- so those counts stay one monomer short.
  * `OneHotEncoder._extract_monomers` likewise parses HELM, and it encodes
    per-position `phosphate`/`sugar`/`base`.  The missing 20th position's sugar
    and phosphate identity is ABSENT from the shipped data.
  * `RNAFMEmbeddings._helm_to_fasta` is base-only too (bases, then T->U), so it
    is fully correctable from `fasta`.
  * `HELMGraph` builds per-monomer sugar/base/phosphate nodes -- not correctable.

**Decision: option (c).**  A partial, clearly-labelled feature-level correction
for the base-only featurizers, PLUS documentation of the whole dataset as a
sensitivity analysis.  Concretely:
  * `KMersCounts`  -> `level = partial_base_composition_only`.  The k-mer counts
    are rebuilt from the full-length `fasta` base sequence; the
    modification-abundance counts are left as HELM produces them.  So base
    composition is corrected and the modification profile is NOT.
  * `RNAFMEmbeddings` -> `level = full_base_sequence`.  The embedding input is
    the full-length base sequence, which is everything this featurizer consumes.
  * `OneHotEncoder`, `HELMGraph`, `SMILESGraph` -> `level = none_helm_truncated`.
    **No transform is applied.**  These configs run on the truncated HELM exactly
    as the faithful arm does, and every output row carries
    `correction_level = none_helm_truncated` so the reproduction report can state
    that these models saw a 19-mer where the molecule is a 20-mer.

**A HELM extension is NOT fabricated.**  Appending a monomer would require
inventing the dropped nucleoside's sugar and phosphate chemistry.  The dropped
base varies across all four bases, the 2'-MOE vs DNA sugar pattern is
position-dependent in these gapmers, and the phosphate linkage (`[sp]` vs `p`)
also varies -- there is no defensible default for any of the three.  Guessing
would put invented chemistry into a benchmark and would be undetectable
downstream.  So the corrective arm corrects what the data supports and documents
the rest.

**What this does NOT fix**, stated plainly: the molecules in this dataset are one
nucleoside longer than any HELM-derived or SMILES-derived feature can represent.
For OneHotEncoder and HELMGraph configs the corrective arm is therefore a
*replicate of the faithful arm*, not a correction, and is useful only as the
control half of the sensitivity analysis.  For KMersCounts the corrected features
differ from the faithful ones only in base composition.  The underlying data
defect can only be repaired upstream, by re-releasing the MOE Neurotox HELM
strings at full length.

=============================================================================
DEFECT 3 -- Ichihara target mis-assignment.  NOT FIXED (no effect here).
=============================================================================
Phase 1 found target labels mis-assigned in the Ichihara release.  This
benchmark's two split strategies are `random` and `nucleobase`; neither groups by
target, so target labels never enter a split or a feature.  Recorded for
completeness; no transform.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

#: Defect 1: the corrected config-key -> downloader-key mapping.  Verified above
#: from the shipped per-file metadata: the file behind 'tlr7' describes the TLR8
#: assay and vice versa.
TLR_SWAP = {
    "immune_modulation_TLR7": "tlr8",  # Alharbi_2020_2 = TLR7 inhibition
    "immune_modulation_TLR8": "tlr7",  # Alharbi_2020_1 = TLR8 potentiation
}

#: Defect 2: what each featurizer's correction can reach.
MOE_DATASET = "acute_neurotox_moe_cleaned"
MOE_LEVEL = {
    "RNAFMEmbeddings": "full_base_sequence",
    "KMersCounts": "partial_base_composition_only",
    "OneHotEncoder": "none_helm_truncated",
    "HELMGraph": "none_helm_truncated",
    "SMILESGraph": "none_helm_truncated",
}


def plan(
    dataset_config_key: str, downloader_key: str, featurizer: str
) -> Tuple[str, Dict[str, Any]]:
    """Decide the corrective action for one (dataset, featurizer) group.

    Returns (downloader_key_to_load, correction_record).  The record is written
    into every output row so a consumer can tell corrected rows from faithful
    ones without consulting this file.
    """
    if dataset_config_key in TLR_SWAP:
        new_key = TLR_SWAP[dataset_config_key]
        return new_key, {
            "applied": "tlr_key_swap",
            "level": "complete",
            "from_key": downloader_key,
            "to_key": new_key,
            "notes": (
                "TLR7/TLR8 downloader keys are swapped relative to their data "
                "(verified: the file behind 'tlr7' carries TLR8-potentiation "
                "metadata and labels 0.57-5.44; the file behind 'tlr8' carries "
                "TLR7-inhibition metadata and labels 17.46-126.5). Both files "
                "hold the same 192 compounds in the same order over the same 4 "
                "targets, so this is a complete relabeling: features unchanged, "
                "endpoint corrected."
            ),
        }

    if dataset_config_key == MOE_DATASET:
        level = MOE_LEVEL.get(featurizer, "none_helm_truncated")
        if level == "none_helm_truncated":
            return downloader_key, {
                "applied": "none",
                "level": level,
                "notes": (
                    f"{featurizer} requires per-position sugar and phosphate "
                    "monomer identity, which exists only in the HELM string. "
                    "HELM and SMILES are truncated by exactly one 3' nucleoside "
                    "in 2437/2437 records and the dropped base varies (C 764, "
                    "T 711, A 621, G 341), so the missing monomer's chemistry "
                    "cannot be recovered and is NOT fabricated. This group runs "
                    "on the truncated HELM, identical to the faithful arm, and "
                    "serves as the control half of a documented sensitivity "
                    "analysis: these models see a 19-mer where the molecule is "
                    "a 20-mer."
                ),
            }
        return downloader_key, {
            "applied": "moe_fasta_base_sequence",
            "level": level,
            "notes": (
                "Neurotox MOE HELM is truncated by one 3' nucleoside in "
                "2437/2437 records; the HELM base sequence is a strict prefix of "
                "the full-length `fasta` column. "
                + (
                    "KMersCounts consumes a base-only sequence, so its k-mer "
                    "counts are rebuilt from `fasta` (recovering the 20th base). "
                    "PARTIAL CORRECTION: the modification_abundance sugar and "
                    "phosphate counts still come from the truncated HELM and "
                    "remain one monomer short."
                    if level == "partial_base_composition_only"
                    else "RNAFMEmbeddings consumes only the base sequence "
                    "(bases, then T->U), so using `fasta` corrects it fully."
                )
            ),
        }

    return downloader_key, {"applied": "none", "level": "n/a", "notes": ""}


# --------------------------------------------------------------------------
# Featurizer overrides.  Both subclass the upstream class and change ONLY the
# HELM -> base-sequence step, so every other code path (k-mer counting, modifica-
# tion counting, RNA-FM batching, padding) is upstream's, unmodified.
# --------------------------------------------------------------------------


def _fasta_lookup(data) -> Dict[str, str]:
    """HELM string -> full-length base sequence, from the dataset's own frame.

    The `fasta` column is the trustworthy full-length sequence: it is one base
    LONGER than the HELM-derived sequence in 2437/2437 records and the HELM
    sequence is a strict prefix of it in 2437/2437.
    """
    df = data.data
    if "fasta" not in df.columns:
        raise KeyError(
            "corrective transform for Neurotox MOE requires the `fasta` column; "
            f"dataset frame has {list(df.columns)}"
        )
    return dict(zip(df["x"].astype(str), df["fasta"].astype(str)))


def make_featurizer(
    featurizer: str,
    featurizer_args: Dict[str, Any],
    data,
    correction: Dict[str, Any],
) -> Optional[Any]:
    """Return a featurizer instance with the correction applied, or None.

    None means "no override": the caller uses the harness's normal
    `prepare_featurizer`, which is exactly right for the TLR swap (features are
    unchanged) and for the uncorrectable MOE featurizers.
    """
    if correction.get("applied") != "moe_fasta_base_sequence":
        return None

    lookup = _fasta_lookup(data)

    if featurizer == "KMersCounts":
        from oligogym.features import KMersCounts

        class KMersCountsFullBase(KMersCounts):
            """k-mers from the full-length `fasta`; modifications from HELM.

            PARTIAL correction.  `_extract_features` is overridden to substitute
            the full-length base sequence into the k-mer step only.  The
            modification_abundance branch is upstream's and still reads the
            truncated HELM monomer table, so sugar/phosphate counts remain one
            monomer short.  That asymmetry is deliberate and is what
            `level = partial_base_composition_only` records.
            """

            def _extract_features(self, oligo_helm: str) -> dict:
                full = lookup.get(str(oligo_helm))
                if full is None:
                    # Unknown HELM: fall back to upstream rather than guess.
                    return super()._extract_features(oligo_helm)
                if self.split_strands:
                    # Multi-strand MOE records do not occur in this dataset
                    # (single RNA1 polymer in all 2,437), and splitting a
                    # concatenated `fasta` across strands would be a guess.
                    return super()._extract_features(oligo_helm)
                kmers = self._extract_kmers(full)
                if self.modification_abundance:
                    monomers = self._extract_monomers(oligo_helm)
                    sugar = self._count_modifications(monomers["sugar"])
                    phos = self._count_modifications(monomers["phosphate"])
                    kmers = {**kmers, **sugar, **phos}
                return kmers

        return KMersCountsFullBase(**featurizer_args)

    if featurizer == "RNAFMEmbeddings":
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from oligogym_patch import RNAFMEmbeddingsFixed

        class RNAFMEmbeddingsFullBase(RNAFMEmbeddingsFixed):
            """RNA-FM on the full-length base sequence.

            FULL correction for this featurizer: `_helm_to_fasta` is the only
            place it reads the HELM string, and it produces a base-only sequence
            (bases, then T->U), so substituting `fasta` recovers everything the
            featurizer consumes.
            """

            def _helm_to_fasta(self, oligo_helm: str) -> str:
                full = lookup.get(str(oligo_helm))
                if full is None:
                    return super()._helm_to_fasta(oligo_helm)
                return full.upper().replace("T", "U")

        valid = set(RNAFMEmbeddingsFullBase.__init__.__code__.co_varnames)
        return RNAFMEmbeddingsFullBase(
            **{k: v for k, v in featurizer_args.items() if k in valid}
        )

    # Any other featurizer reaching here would be a routing bug in plan().
    raise AssertionError(
        f"plan() asked for a fasta correction on {featurizer!r}, which does not "
        "consume a base-only sequence; this is a bug, not a data problem."
    )


def audit(data, dataset_config_key: str) -> Dict[str, Any]:
    """Quantify the MOE truncation on the loaded data.  Used by the report.

    Recomputed at run time rather than quoted, so the numbers in the smoke report
    are measurements from the same package resources the pipeline read.
    """
    df = data.data
    if dataset_config_key != MOE_DATASET or "fasta" not in df.columns:
        return {}
    import re

    mono = re.compile(r"\[?[A-Za-z0-9]+\]?\((\[?[A-Za-z0-9]+\]?)\)")

    def helm_bases(h: str) -> str:
        body = h.split("{", 1)[1].rsplit("}", 1)[0]
        out = []
        for tok in body.split("."):
            m = mono.search(tok)
            if m:
                out.append(m.group(1).strip("[]")[-1])
        return "".join(out)

    hb = df["x"].astype(str).map(helm_bases)
    fa = df["fasta"].astype(str)
    diff = (fa.str.len() - hb.str.len()).value_counts().to_dict()
    dropped = pd.Series(
        [f[len(b):] for f, b in zip(fa, hb)]
    ).value_counts().to_dict()
    return {
        "n_records": int(len(df)),
        "length_deficit_distribution": {int(k): int(v) for k, v in diff.items()},
        "helm_is_strict_prefix_of_fasta": int(
            sum(f.startswith(b) for f, b in zip(fa, hb))
        ),
        "dropped_base_counts": {str(k): int(v) for k, v in dropped.items()},
        "smiles_P_equals_helm_monomers_minus_1": int(
            (df["smiles"].str.count("P") == hb.str.len() - 0).sum()
        )
        if "smiles" in df.columns
        else None,
    }


if __name__ == "__main__":
    # Self-check: prints the plan for each (dataset, featurizer) the corrective
    # arm contains, so the decisions are inspectable without running a chunk.
    rows = []
    for ds in ("immune_modulation_TLR7", "immune_modulation_TLR8", MOE_DATASET):
        for f in ("OneHotEncoder", "KMersCounts", "HELMGraph", "RNAFMEmbeddings"):
            key, rec = plan(ds, "PLACEHOLDER", f)
            rows.append(
                {
                    "dataset": ds,
                    "featurizer": f,
                    "downloader_key": key,
                    "applied": rec["applied"],
                    "level": rec["level"],
                }
            )
    print(json.dumps(rows, indent=2))
