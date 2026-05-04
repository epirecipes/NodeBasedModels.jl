import Lake
open Lake DSL

package "PairwiseProofs" where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

require "leanprover-community" / "mathlib"

@[default_target]
lean_lib «PairwiseProofs» where
  globs := #[.submodules `PairwiseProofs]
