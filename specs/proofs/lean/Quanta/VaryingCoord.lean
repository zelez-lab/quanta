-- Vertex→fragment varying location coordination.
--
-- Theorem T8: vertex output Location[i] matches fragment input Location[i].
-- Convention: vertex param[0] = position (not forwarded).
-- Vertex param[k] for k ≥ 1 → output Location[k-1].
-- Fragment param[j] → input Location[j].

namespace Quanta.VaryingCoord

/-- Vertex output location for the k-th attribute parameter (0-indexed). -/
def vertexOutputLocation (paramIndex : Nat) : Option Nat :=
  if paramIndex = 0 then none  -- position goes to gl_Position, not a varying
  else some (paramIndex - 1)

/-- Fragment input location for the j-th parameter (0-indexed). -/
def fragmentInputLocation (paramIndex : Nat) : Nat :=
  paramIndex

-- Theorem: vertex param[1] outputs to Location 0, matching fragment param[0]
theorem first_varying_matches :
    vertexOutputLocation 1 = some (fragmentInputLocation 0) := by
  rfl

-- Theorem: vertex param[k+1] outputs to Location k, matching fragment param[k]
theorem varying_coordination :
    ∀ k, vertexOutputLocation (k + 1) = some (fragmentInputLocation k) := by
  intro k; simp [vertexOutputLocation, fragmentInputLocation]

-- Theorem: position (param 0) is never forwarded as a varying
theorem position_not_forwarded :
    vertexOutputLocation 0 = none := by
  rfl

end Quanta.VaryingCoord
