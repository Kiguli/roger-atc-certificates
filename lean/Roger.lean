import Mathlib

/-!
# Machine-checked formalization of the certificates in
"Roger? Probabilistic Certificates for Safety-Critical ATC Speech Recognition"

| Paper result            | Lean declaration                              |
|-------------------------|-----------------------------------------------|
| Definition 1 (def:cp)   | `RogerCert.CPp`                               |
| Theorem 1  (thm:cp)     | `RogerCert.CPp_coverage`                      |
| Lemma 2    (lem:mono)   | `RogerCert.gcs_le_gmsg`, `count_mono`,        |
|                         | `V_mono`, `certified_mono`                    |
| Theorem 3  (thm:decomp) | `RogerCert.hgap_indicator`, `decomp_coverage`,|
|                         | `union_two`                                   |
| Theorem 4  (thm:l2)     | `RogerCert.audited_bound`, `audited_bound_prob`|
| Proposition 5 (prop:csaudit) | `RogerCert.csaudit`                      |

The two classical binomial facts are isolated as clearly-labelled `axiom`s
(`binCDF_antitone`, `cdf_transform`); everything else is proved with no `sorry`.
-/

namespace RogerCert
open Finset
open scoped BigOperators Classical

/-- Binomial pmf term `P[Bin(N,·)=j]`. -/
noncomputable def binPMF (N j : ℕ) (p : ℝ) : ℝ :=
  (N.choose j : ℝ) * p ^ j * (1 - p) ^ (N - j)

/-- Binomial left-tail CDF `F_{N,p}(k) = P[Bin(N,p) ≤ k]`. -/
noncomputable def binCDF (N k : ℕ) (p : ℝ) : ℝ :=
  ∑ j ∈ range (k + 1), binPMF N j p

/-- Probability, under `Bin(N,q)`, of an event `A` on the count. -/
noncomputable def Pn (N : ℕ) (q : ℝ) (A : ℕ → Prop) : ℝ :=
  ∑ k ∈ range (N + 1), if A k then binPMF N k q else 0

lemma binPMF_nonneg {N j : ℕ} {p : ℝ} (h0 : 0 ≤ p) (h1 : p ≤ 1) :
    0 ≤ binPMF N j p := by
  unfold binPMF
  have h2 : 0 ≤ 1 - p := by linarith
  positivity

/-- Monotonicity of the count-probability in the event. -/
lemma Pn_mono {N : ℕ} {q : ℝ} (hq : q ∈ Set.Icc (0:ℝ) 1) {A B : ℕ → Prop}
    (hAB : ∀ k, A k → B k) : Pn N q A ≤ Pn N q B := by
  obtain ⟨h0, h1⟩ := hq
  refine Finset.sum_le_sum (fun k _ => ?_)
  by_cases hA : A k
  · simp [hA, hAB k hA]
  · have hnn : (0:ℝ) ≤ if B k then binPMF N k q else 0 := by
      by_cases hB : B k <;> simp [hB, binPMF_nonneg h0 h1]
    simp [hA]; exact hnn

/-- Candidate set for the Clopper–Pearson upper limit; `1` is always included so
that the paper's `inf ∅ = 1` convention holds. -/
def cpSet (N k : ℕ) (β : ℝ) : Set ℝ :=
  {p : ℝ | p ∈ Set.Icc (0:ℝ) 1 ∧ binCDF N k p ≤ β} ∪ {1}

/-- **Definition 1.** Exact-binomial (Clopper–Pearson) upper limit `CP⁺(k,N,β)`. -/
noncomputable def CPp (N k : ℕ) (β : ℝ) : ℝ := sInf (cpSet N k β)

lemma cpSet_nonempty (N k : ℕ) (β : ℝ) : (cpSet N k β).Nonempty :=
  ⟨1, Or.inr rfl⟩

-- ## The two classical binomial facts, isolated as labelled axioms.

/-- **Axiom (classical).** The binomial left tail `F_{N,p}(k) = P[Bin(N,p) ≤ k]` is
antitone in the success probability `p` on `[0,1]`. Standard: provable by a monotone
coupling of `Bin(N,p)` in `p`, or from the sign of the derivative
`dF/dp = -N·C(N-1,k)·p^k(1-p)^{N-1-k} ≤ 0`. -/
axiom binCDF_antitone (N k : ℕ) :
    AntitoneOn (fun p => binCDF N k p) (Set.Icc (0:ℝ) 1)

/-- **Axiom (classical).** Discrete probability-integral-transform bound: for
`K ~ Bin(N,q)`, `P[F_{N,q}(K) ≤ β] ≤ β`. Standard super-uniformity of the (left-tail)
CDF transform for a discrete variable. -/
axiom cdf_transform {N : ℕ} {β q : ℝ} (hq : q ∈ Set.Icc (0:ℝ) 1) :
    Pn N q (fun k => binCDF N k q ≤ β) ≤ β

/-- If the true probability `q` exceeds `CP⁺(k,N,β)`, then `F_{N,q}(k) ≤ β`. -/
lemma CPp_lt_imp_cdf_le {N k : ℕ} {β q : ℝ} (hq : q ∈ Set.Icc (0:ℝ) 1)
    (h : CPp N k β < q) : binCDF N k q ≤ β := by
  obtain ⟨p, hpS, hpq⟩ := exists_lt_of_csInf_lt (cpSet_nonempty N k β) h
  rcases hpS with hp | hp
  · obtain ⟨hpIcc, hpβ⟩ := hp
    exact le_trans (binCDF_antitone N k hpIcc hq (le_of_lt hpq)) hpβ
  · rw [Set.mem_singleton_iff] at hp
    exfalso; rw [hp] at hpq; linarith [hq.2]

/-- **Theorem 1** (per-message validity). With `K ~ Bin(N,V_x)`,
`P[V_x > CP⁺(K,N,β)] ≤ β`. -/
theorem CPp_coverage {N : ℕ} {β q : ℝ} (hq : q ∈ Set.Icc (0:ℝ) 1) :
    Pn N q (fun k => CPp N k β < q) ≤ β :=
  le_trans (Pn_mono hq (fun _ hk => CPp_lt_imp_cdf_le hq hk)) (cdf_transform hq)

/-- **Proposition 5** (assumption-free call-sign audit). The forgiven-flip count
`m ~ Bin(N,η)` satisfies `P[η > CP⁺(m,N,β)] ≤ β`. This is Theorem 1 applied to the
audited indicator; no new content. -/
theorem csaudit {N : ℕ} {β η : ℝ} (hη : η ∈ Set.Icc (0:ℝ) 1) :
    Pn N η (fun m => CPp N m β < η) ≤ β :=
  CPp_coverage hη

-- ## Lemma 2 : per-draw / count / expectation monotonicity of the specifications.

section Specs
variable {Transcript CallSign : Type*} (T0 : Transcript) (Φcs : Transcript → CallSign)

/-- Per-draw exact-transcript violation indicator (0/1). -/
noncomputable def gmsg (t : Transcript) : ℕ := if t = T0 then 0 else 1
/-- Per-draw call-sign violation indicator (0/1). -/
noncomputable def gcs (t : Transcript) : ℕ := if Φcs t = Φcs T0 then 0 else 1

/-- **Lemma 2** (per-draw monotonicity, call-sign case): `g_cs ≤ g_msg` pointwise.
The same argument gives `g_sem ≤ g_msg`. -/
lemma gcs_le_gmsg (t : Transcript) : gcs T0 Φcs t ≤ gmsg T0 t := by
  unfold gcs gmsg
  by_cases h : t = T0
  · subst h; simp
  · rw [if_neg h]; split <;> norm_num

/-- Count inequality `k_cs ≤ k_msg` over `N` draws. -/
lemma count_mono (N : ℕ) (τ : ℕ → Transcript) :
    ∑ i ∈ range N, gcs T0 Φcs (τ i) ≤ ∑ i ∈ range N, gmsg T0 (τ i) :=
  Finset.sum_le_sum (fun i _ => gcs_le_gmsg T0 Φcs (τ i))

/-- A message certified under `msg` (count `≤ k*`) is certified under `cs`. -/
lemma certified_mono (N kstar : ℕ) (τ : ℕ → Transcript)
    (hcert : ∑ i ∈ range N, gmsg T0 (τ i) ≤ kstar) :
    ∑ i ∈ range N, gcs T0 Φcs (τ i) ≤ kstar :=
  le_trans (count_mono T0 Φcs N τ) hcert

/-- Expectation inequality `V_cs ≤ V_msg` under any monotone noise-law expectation. -/
lemma V_mono (Eμ : (Transcript → ℝ) → ℝ)
    (Eμ_mono : ∀ f g : Transcript → ℝ, (∀ t, f t ≤ g t) → Eμ f ≤ Eμ g) :
    Eμ (fun t => (gcs T0 Φcs t : ℝ)) ≤ Eμ (fun t => (gmsg T0 t : ℝ)) :=
  Eμ_mono _ _ (fun t => by exact_mod_cast gcs_le_gmsg T0 Φcs t)

end Specs

-- ## Theorem 3 : nested-specification decomposition.

/-- The changed-but-forgiven per-draw indicator `g_msg − g_sem` is the 0/1 indicator
of `{g_msg = 1 ∧ g_sem = 0}` (difference of nested indicators). -/
lemma hgap_indicator {gm gs : ℕ} (hle : gs ≤ gm) (hm : gm ≤ 1) :
    gm - gs = if gm = 1 ∧ gs = 0 then 1 else 0 := by
  interval_cases gm <;> interval_cases gs <;> simp_all

/-- **Theorem 3** (tight certificate on the changed-but-forgiven mass). The
difference count `d ~ Bin(N,D_x)` gives `P[D_x > CP⁺(d,N,β)] ≤ β`. -/
theorem decomp_coverage {N : ℕ} {β D : ℝ} (hD : D ∈ Set.Icc (0:ℝ) 1) :
    Pn N D (fun d => CPp N d β < D) ≤ β :=
  CPp_coverage hD

/-- Union bound: from two `β`-level coverage guarantees on the same draws, the joint
failure has probability `≤ 2β` (independence not needed). -/
lemma union_two {β pbad1 pbad2 pboth : ℝ} (h1 : pbad1 ≤ β) (h2 : pbad2 ≤ β)
    (hsub : pboth ≤ pbad1 + pbad2) : pboth ≤ 2 * β := by linarith

-- ## Theorem 4 : audited semantic bound.

/-- **Theorem 4** (deterministic audited bound). Under the three-way decomposition
of `P[g*=1]`, the vanishing of the impossible term (`p_out = 0`, since `g* ≤ g_msg`),
`p_sem1 ≤ V_sem`, and Assumption 1 (`p_mid ≤ ρ_FP·D_x`), one has
`V*_x ≤ V_sem + ρ_FP·D_x`. -/
theorem audited_bound {Vstar Vsem Dx rhoFP p_sem1 p_mid p_out : ℝ}
    (hdecomp : Vstar = p_sem1 + p_mid + p_out) (hout : p_out = 0)
    (hsem1 : p_sem1 ≤ Vsem) (hmid : p_mid ≤ rhoFP * Dx) :
    Vstar ≤ Vsem + rhoFP * Dx := by
  rw [hdecomp, hout]; linarith

/-- **Theorem 4** (probabilistic form). On the `1−2β` event where the two Clopper–Pearson
bounds hold (`V_sem ≤ c_sem`, `D_x ≤ c_d`), the audited bound becomes
`V*_x ≤ c_sem + ρ_FP·c_d`. -/
theorem audited_bound_prob {Vstar Vsem Dx rhoFP csem cd : ℝ}
    (hbound : Vstar ≤ Vsem + rhoFP * Dx)
    (hVsem : Vsem ≤ csem) (hDx : Dx ≤ cd) (hrho : 0 ≤ rhoFP) :
    Vstar ≤ csem + rhoFP * cd := by
  have h : rhoFP * Dx ≤ rhoFP * cd := mul_le_mul_of_nonneg_left hDx hrho
  linarith

end RogerCert
