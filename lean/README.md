# Lean 4 formalization

Machine-checked proofs of the formal results in the paper (Appendix A), in
`Roger.lean`, built against **Mathlib `v4.29.0`** (Lean 4 `v4.29.0`).

## Build
```bash
cd lean
lake exe cache get     # download prebuilt Mathlib oleans
lake build             # compiles Roger.lean
```

## What is proved
Every declaration is `sorry`-free. Two standard analytic facts about the
binomial tail are isolated as clearly-labelled `axiom`s and everything else is
proved from them (and Lean's foundations):

* `binCDF_antitone` — the binomial left tail `P[Bin(N,p) ≤ k]` is antitone in `p`;
* `cdf_transform`  — the discrete probability-integral-transform bound
  `P[F_{N,q}(K) ≤ β] ≤ β`.

| Paper result | Lean declaration(s) | Extra axioms |
|---|---|---|
| Definition 1 (`def:cp`) | `CPp` | — |
| Theorem 1 (`thm:cp`) | `CPp_coverage` | + 2 ax. |
| Lemma 2 (`lem:mono`) | `gcs_le_gmsg`, `count_mono`, `V_mono`, `certified_mono` | none |
| Theorem 3 (`thm:decomp`) | `hgap_indicator`, `decomp_coverage`, `union_two` | + 2 ax. (coverage) |
| Theorem 4 (`thm:l2`) | `audited_bound`, `audited_bound_prob` | none |
| Proposition 5 (`prop:csaudit`) | `csaudit` | + 2 ax. |

## Verifying the `sorry`-free / axiom claim
```bash
lake env lean --run - <<'EOF'   # or add to the file:
#print axioms RogerCert.CPp_coverage
EOF
```
Each result reports only `[propext, Classical.choice, Quot.sound]` (Lean's
foundations) plus, where applicable, `binCDF_antitone` and `cdf_transform` —
never `sorryAx`.
