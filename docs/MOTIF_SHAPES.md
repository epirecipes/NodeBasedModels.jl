# Motif Shapes and State-Class Reference

Shape catalogue for `src/motif_based.jl`. Supported: k=2 (m=2–6), k=3 (m=2–4).

**Counting convention (locked):** variables count *unordered* induced embeddings
partitioned by canonical state under Aut(shape). Orbit sizes convert to labelled
quantities: `L_σ = E_canon · |orb(σ)|`. P₂ conversion: `[SS]_pw = 2·E_SS`,
`[SI]_pw = E_IS`, `[II]_pw = 2·E_II` (κ = (k−1)/k is the Keeling factor).

## Variable-count summary

| (k,m)  | Shapes tracked                                         | Variables |
|--------|-------------------------------------------------------|----------:|
| (2, 2) | singleton + P₂                                        |         5 |
| (2, 3) | singleton + P₂ + P₃                                   |        11 |
| (2, 4) | singleton + P₂ + P₄                                   |        15 |
| (2, 5) | singleton + P₂ + P₅                                   |        25 |
| (2, 6) | singleton + P₂ + P₆                                   |        41 |
| (3, 2) | singleton + P₂                                        |         5 |
| (3, 3) | singleton + P₂ + P₃ + C₃                              |        15 |
| (3, 4) | singleton + P₂ + P₃ + C₃ + P₄ + K₁,₃ + paw + C₄ + K₄−e + K₄ |    65 |

For k=2 and m≥4, only `singleton + P₂ + P_m` are tracked; lower paths are
recovered as on-the-fly marginals of the P_m variables.

## Shape state-class tables

Format: `name  n  |E|  |Aut|  variables` followed by `[canonical_state] orbit`.

---
**:singleton**  n=1  |E|=0  |Aut|=1  **vars=2**  
`[I] 1`  `[S] 1`

---
**:P2**  n=2  edges=(1,2)  |Aut|=2  **vars=3**  
Aut: {id, [2,1]}  
`[I,I] 1`  `[I,S] 2`  `[S,S] 1`

---
**:P3**  n=3  edges=(1,2),(2,3)  |Aut|=2  **vars=6**  
Aut: {id, [3,2,1]}  
`[I,I,I] 1`  `[I,I,S] 2`  `[I,S,I] 1`  `[I,S,S] 2`  `[S,I,S] 1`  `[S,S,S] 1`

---
**:C3**  n=3  edges=(1,2),(2,3),(1,3)  |Aut|=6  **vars=4**  
Aut: S₃ (all 6 permutations of {1,2,3})  
`[I,I,I] 1`  `[I,I,S] 3`  `[I,S,S] 3`  `[S,S,S] 1`

---
**:P4**  n=4  edges=(1,2),(2,3),(3,4)  |Aut|=2  **vars=10**  
Aut: {id, [4,3,2,1]}; palindromic (orbit-1): IIII, ISSI, SIIS, SSSS  

| state   | orb | state   | orb |
|---------|:---:|---------|:---:|
| [I,I,I,I] | 1 | [I,S,S,I] | 1 |
| [I,I,I,S] | 2 | [I,S,S,S] | 2 |
| [I,I,S,I] | 2 | [S,I,I,S] | 1 |
| [I,I,S,S] | 2 | [S,I,S,S] | 2 |
| [I,S,I,S] | 2 | [S,S,S,S] | 1 |

---
**:P5**  n=5  edges=(1,2)–(4,5)  |Aut|=2  **vars=20**  
Orbit-1 (palindromes s[i]=s[6-i]): 8 classes. Orbit-2: 12 classes.

---
**:P6**  n=6  edges=(1,2)–(5,6)  |Aut|=2  **vars=36**  
Orbit-1 (palindromes s[i]=s[7-i]): 8 classes. Orbit-2: 28 classes.

---
**:K13**  (claw K_{1,3})  n=4  edges=(1,2),(1,3),(1,4)  |Aut|=6  **vars=8**  
Aut: S₃ on leaves {2,3,4}; center fixed. Classes by (center, multiset of leaves).  

| state     | orb | state     | orb |
|-----------|:---:|-----------|:---:|
| [I,I,I,I] | 1   | [S,I,I,I] | 1   |
| [I,I,I,S] | 3   | [S,I,I,S] | 3   |
| [I,I,S,S] | 3   | [S,I,S,S] | 3   |
| [I,S,S,S] | 1   | [S,S,S,S] | 1   |

---
**:paw**  (triangle + pendant)  n=4  edges=(1,2),(1,3),(2,3),(1,4)  |Aut|=2  **vars=12**  
Aut: {id, [1,3,2,4]} (swap positions 2 and 3). Orbit-1 (s[2]=s[3]): 8. Orbit-2: 4.  

| state     | orb | state     | orb | state     | orb |
|-----------|:---:|-----------|:---:|-----------|:---:|
| [I,I,I,I] | 1   | [I,S,S,I] | 1   | [S,I,I,S] | 1   |
| [I,I,I,S] | 1   | [I,S,S,S] | 1   | [S,I,S,I] | 2   |
| [I,I,S,I] | 2   | [S,I,I,I] | 1   | [S,I,S,S] | 2   |
| [I,I,S,S] | 2   | [S,S,S,I] | 1   | [S,S,S,S] | 1   |

---
**:C4**  (4-cycle)  n=4  edges=(1,2),(2,3),(3,4),(4,1)  |Aut|=8  **vars=6**  
Aut: dihedral D₄. Classes indexed by orbit under rotations + reflections.  
`[I,I,I,I] 1`  `[I,I,I,S] 4`  `[I,I,S,S] 4`  `[I,S,I,S] 2`  `[I,S,S,S] 4`  `[S,S,S,S] 1`

---
**:K4me**  (diamond K₄−e)  n=4  edges=(1,2),(2,3),(3,4),(4,1),(1,3)  |Aut|=4  **vars=9**  
Aut: Klein 4-group {id, (1↔3), (2↔4), (1↔3)(2↔4)}. Vertices 1,3 degree-3; 2,4 degree-2.  

| state     | orb | state     | orb | state     | orb |
|-----------|:---:|-----------|:---:|-----------|:---:|
| [I,I,I,I] | 1   | [I,I,S,S] | 4   | [S,I,S,I] | 1   |
| [I,I,I,S] | 2   | [I,S,I,S] | 1   | [S,I,S,S] | 2   |
| [I,I,S,I] | 2   | [I,S,S,S] | 2   | [S,S,S,S] | 1   |

---
**:K4**  (complete graph)  n=4  all 6 edges  |Aut|=24  **vars=5**  
Aut: S₄. Classes by number of I's.  
`[I,I,I,I] 1`  `[I,I,I,S] 4`  `[I,I,S,S] 6`  `[I,S,S,S] 4`  `[S,S,S,S] 1`

---

## Kirkwood closure identities

**(k=2, m=3)** — P₄ closure at an endpoint (more stable single-denominator form):

$$L_{(\sigma_0,\sigma_1,\sigma_2,\sigma_3)}^{P_4} \approx L_{(\sigma_0,\sigma_1,\sigma_2)}^{P_3}\cdot L_{(\sigma_1)}^{\rm node}\;/\;\langle\sigma_1\rangle$$

**(k=2, m≥4)** — generic chain (implemented via `_build_sis_k2_chain_rhs`):

$$L_{P_{m+1}} \approx L_{P_m}(\sigma_0,\ldots,\sigma_{m-1})\cdot L_{P_m}(\sigma_1,\ldots,\sigma_m)\;/\;L_{P_{m-1}}(\sigma_1,\ldots,\sigma_{m-1})$$

## Natural next extensions

- **(k=2, m=7):** add P₇ (orbit-1: 8 classes, orbit-2: 56 classes = **64 new vars**).
  The generic chain builder extends directly with no new code.
- **(k=3, m=5):** open extension. The current `(k=3,m=4)` Kirkwood refinement
  is already subject to the Lean-certified m=4→m=3 marginalisation obstruction
  (T3b/T7 in `EdgeBasedModels.jl/proofs/EBCMCategory/`), so higher-order
  extensions should not be interpreted as automatic monotone improvements.
  A non-Kirkwood or constrained-consistency closure is the likely next
  research direction.
- **(k=4, m=2):** trivial — the `(k=2,m=2)` RHS builder is already parameterised
  by `κ=(k−1)/k`; 5 variables.
