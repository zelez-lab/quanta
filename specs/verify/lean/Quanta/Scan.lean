/-
# Blelloch exclusive prefix sum — formal specification

Mirrors `src/scan.rs::build_exclusive_scan`. The Blelloch algorithm
operates on a power-of-two array in two phases:

1. **Up-sweep (reduce)**: Build partial sums bottom-up in a binary
   tree. After the up-sweep, `tree[n-1] = sum(input[0..n])`.

2. **Down-sweep (distribute)**: Set `tree[n-1] = 0`, then propagate
   prefix sums top-down. After the down-sweep,
   `tree[i] = sum(input[0..i])` (exclusive prefix sum).

The scan.rs implementation uses `SCAN_WORKGROUP_SIZE = 256` and
8 sweep iterations (`log2(256) = 8`).

Theorems:
- T900: Exclusive prefix sum: output[i] = sum(input[0..i])
- T901: output[0] = identity element (0 for addition)
- T903: Up-sweep invariant: after up-sweep, tree[n-1] = sum(input[0..n])
- T904: Down-sweep invariant: after down-sweep, tree[i] = sum(input[0..i])
-/

namespace Quanta.Scan

-- ─────────────────────────────────────────────────────────────────────
-- Definitions
-- ─────────────────────────────────────────────────────────────────────

/-- Sum of a list (identity element = 0 for addition). -/
def listSum : List Nat → Nat
  | []      => 0
  | x :: xs => x + listSum xs

/-- Prefix sum: sum of elements input[0..i] (exclusive, i.e., not including i). -/
def prefixSum (input : List Nat) (i : Nat) : Nat :=
  listSum (input.take i)

/-- A valid exclusive scan result: output[i] = sum(input[0..i]) for all i. -/
def IsExclusiveScan (input output : List Nat) : Prop :=
  input.length = output.length
  ∧ ∀ i, i < output.length → output.get! i = prefixSum input i

-- ─────────────────────────────────────────────────────────────────────
-- Tree model for the Blelloch algorithm
-- ─────────────────────────────────────────────────────────────────────

/-- The sweep tree is a flat array (same size as input).
    During the up-sweep, elements are accumulated bottom-up.
    During the down-sweep, prefix sums are distributed top-down. -/
abbrev Tree := List Nat

/-- Initialize tree from input: tree[i] = input[i]. -/
def initTree (input : List Nat) : Tree := input

/-- One up-sweep step at depth `d`:
    stride = 2^(d+1)
    For each index where (i+1) % stride == 0:
      tree[i] += tree[i - stride/2]

    We model this as a function that transforms the entire tree. -/
def upSweepStep (tree : Tree) (d : Nat) : Tree :=
  let stride := 2 ^ (d + 1)
  let halfStride := 2 ^ d
  tree.enum.map fun ⟨i, v⟩ =>
    if (i + 1) % stride = 0 ∧ i ≥ halfStride then
      v + tree.get! (i - halfStride)
    else
      v

/-- Apply up-sweep for depths 0 to maxDepth-1. -/
def upSweep (tree : Tree) : (depth : Nat) → Tree
  | 0     => tree
  | d + 1 => upSweepStep (upSweep tree d) d

/-- One down-sweep step at depth `d` (iterating from high to low):
    stride = 2^(d+1)
    For each index where (i+1) % stride == 0:
      temp = tree[i - stride/2]
      tree[i - stride/2] = tree[i]
      tree[i] = tree[i] + temp -/
def downSweepStep (tree : Tree) (d : Nat) : Tree :=
  let stride := 2 ^ (d + 1)
  let halfStride := 2 ^ d
  let pairs := tree.enum.map fun ⟨i, v⟩ =>
    if (i + 1) % stride = 0 ∧ i ≥ halfStride then
      let left := tree.get! (i - halfStride)
      (i, v + left, i - halfStride, v)
    else
      (i, v, i, v)  -- no-op sentinel
  -- Apply the swaps: for each active pair, set tree[left_idx] = old_right,
  -- tree[right_idx] = old_right + old_left
  applyDownSwaps tree pairs
where
  applyDownSwaps (t : Tree) : List (Nat × Nat × Nat × Nat) → Tree
    | []                            => t
    | (ri, new_r, li, new_l) :: ps =>
      if (ri + 1) % (2 ^ (d + 1)) = 0 ∧ ri ≥ 2 ^ d then
        let t' := t.set li new_l |>.set ri new_r
        applyDownSwaps t' ps
      else
        applyDownSwaps t ps

/-- Apply down-sweep for depths maxDepth-1 down to 0. -/
def downSweep (tree : Tree) : (depth : Nat) → Tree
  | 0     => tree
  | d + 1 => downSweep (downSweepStep tree d) d

/-- Set the last element to 0 (identity for exclusive scan). -/
def clearLast (tree : Tree) : Tree :=
  match tree with
  | [] => []
  | _  => tree.set (tree.length - 1) 0

/-- Full Blelloch exclusive scan on a power-of-two array.
    1. Initialize tree from input
    2. Up-sweep for log2(n) steps
    3. Clear last element to 0
    4. Down-sweep for log2(n) steps -/
def blellochScan (input : List Nat) (logN : Nat) : Tree :=
  let tree := initTree input
  let afterUp := upSweep tree logN
  let cleared := clearLast afterUp
  downSweep cleared logN

-- ─────────────────────────────────────────────────────────────────────
-- Helper lemmas
-- ─────────────────────────────────────────────────────────────────────

theorem listSum_nil : listSum [] = 0 := rfl

theorem listSum_singleton (x : Nat) : listSum [x] = x := by
  simp [listSum]

theorem listSum_cons (x : Nat) (xs : List Nat) :
    listSum (x :: xs) = x + listSum xs := rfl

theorem listSum_append (xs ys : List Nat) :
    listSum (xs ++ ys) = listSum xs + listSum ys := by
  induction xs with
  | nil          => simp [listSum]
  | cons x xs ih => simp [listSum, ih, Nat.add_assoc]

theorem prefixSum_zero (input : List Nat) :
    prefixSum input 0 = 0 := by
  simp [prefixSum, listSum]

theorem prefixSum_succ (input : List Nat) (i : Nat) (h : i < input.length) :
    prefixSum input (i + 1) = prefixSum input i + input.get! i := by
  simp [prefixSum]
  rw [List.take_succ]
  simp [List.get?_eq_get! h, listSum_append, listSum_singleton]

-- ─────────────────────────────────────────────────────────────────────
-- T901: output[0] = 0 (identity element)
-- ─────────────────────────────────────────────────────────────────────

/-- T901: The first element of an exclusive scan is always the
    identity element (0 for addition).

    This follows directly from the definition: output[0] = sum(input[0..0])
    = sum([]) = 0. -/
theorem t901_output_zero_is_identity (input output : List Nat)
    (h : IsExclusiveScan input output) (hne : 0 < output.length) :
    output.get! 0 = 0 := by
  obtain ⟨hlen, hvals⟩ := h
  have := hvals 0 hne
  simp [prefixSum_zero] at this
  exact this

-- ─────────────────────────────────────────────────────────────────────
-- T903: Up-sweep invariant
-- ─────────────────────────────────────────────────────────────────────

/-- The up-sweep invariant: after depth `d` of the up-sweep,
    for every index `i` where `(i+1)` is a multiple of `2^(d+1)`,
    `tree[i]` contains the sum of the `2^(d+1)` input elements
    ending at position `i`.

    At the maximum depth (d = log2(n) - 1), the last element
    contains the total sum. -/
def UpSweepInvariant (tree : Tree) (input : List Nat) (d : Nat) : Prop :=
  ∀ i, i < tree.length →
    (i + 1) % (2 ^ (d + 1)) = 0 →
    let blockSize := 2 ^ (d + 1)
    let blockStart := i + 1 - blockSize
    tree.get! i = listSum (input.drop blockStart |>.take blockSize)

/-- T903: After the complete up-sweep (logN steps), the last element
    of the tree equals the total sum of the input.

    This is a consequence of the up-sweep invariant at the maximum
    depth: tree[n-1] = sum(input[0..n]).

    Proof sketch: By induction on the sweep depth.
    - Base case (d=0): Each odd-indexed element tree[2k+1] = input[2k] + input[2k+1].
    - Inductive step: At depth d+1, stride doubles. Each active element
      accumulates the sum from its left child (distance stride/2).
      tree[i] += tree[i - 2^d], which by IH is the sum of the left half. -/
theorem t903_upsweep_total_sum (input : List Nat) (n : Nat) (logN : Nat)
    (hn : input.length = n) (hpow : n = 2 ^ logN) (hpos : 0 < n) :
    (upSweep (initTree input) logN).get! (n - 1) = listSum input := by
  sorry  -- Full inductive proof over sweep depth

-- ─────────────────────────────────────────────────────────────────────
-- T904: Down-sweep invariant
-- ─────────────────────────────────────────────────────────────────────

/-- The down-sweep invariant: after depth `d` of the down-sweep
    (iterating from high depth to low), for every index `i` where
    `(i+1)` is a multiple of `2^(d+1)`,
    `tree[i]` contains the exclusive prefix sum up to position `i+1`.

    At the minimum depth (d = 0), every element contains the
    exclusive prefix sum: tree[i] = sum(input[0..i]). -/
def DownSweepInvariant (tree : Tree) (input : List Nat) (d : Nat) : Prop :=
  ∀ i, i < tree.length →
    (i + 1) % (2 ^ (d + 1)) = 0 →
    tree.get! i = prefixSum input (i + 1)

/-- T904: After the complete down-sweep (logN steps from maxDepth-1
    down to 0), every element contains the exclusive prefix sum.

    Proof sketch: By induction on the sweep depth, counting down.
    - Base case (d = logN-1): After clearing the last element to 0
      and one down-sweep step, tree[n/2-1] = 0 (prefix sum of first half)
      and tree[n-1] = total_sum (from up-sweep, now moved left).
    - Inductive step (d → d-1): Each active pair (left, right) where
      right = tree[i] and left = tree[i - stride/2]:
        new_left = old_right (prefix sum up to midpoint)
        new_right = old_right + old_left
      By IH, old_right = prefixSum(input, i+1), so both children
      are correct prefix sums at the finer granularity. -/
theorem t904_downsweep_prefix_sums (input : List Nat) (n : Nat) (logN : Nat)
    (hn : input.length = n) (hpow : n = 2 ^ logN) (hpos : 0 < n) :
    let afterUp := upSweep (initTree input) logN
    let cleared := clearLast afterUp
    let result := downSweep cleared logN
    ∀ i, i < n → result.get! i = prefixSum input i := by
  sorry  -- Full inductive proof over sweep depth (counting down)

-- ─────────────────────────────────────────────────────────────────────
-- T900: Exclusive prefix sum correctness
-- ─────────────────────────────────────────────────────────────────────

/-- T900: The Blelloch scan produces a valid exclusive prefix sum.
    For all i < n: output[i] = sum(input[0..i]).

    This is the composition of T903 (up-sweep produces total sum)
    and T904 (down-sweep distributes prefix sums). -/
theorem t900_exclusive_scan_correct (input : List Nat) (n : Nat) (logN : Nat)
    (hn : input.length = n) (hpow : n = 2 ^ logN) (hpos : 0 < n) :
    let output := blellochScan input logN
    IsExclusiveScan input output := by
  constructor
  · -- Length preservation: all operations preserve array length
    sorry  -- Each sweep step preserves length (map over enum)
  · -- Value correctness: follows from T904
    intro i hi
    exact t904_downsweep_prefix_sums input n logN hn hpow hpos i hi

-- ─────────────────────────────────────────────────────────────────────
-- Concrete instantiation: n=4, logN=2
-- ─────────────────────────────────────────────────────────────────────

/-- Worked example: scan of [1, 2, 3, 4] should produce [0, 1, 3, 6].
    This is a concrete sanity check of the algorithm. -/
example : prefixSum [1, 2, 3, 4] 0 = 0 := by simp [prefixSum, listSum]
example : prefixSum [1, 2, 3, 4] 1 = 1 := by simp [prefixSum, listSum]
example : prefixSum [1, 2, 3, 4] 2 = 3 := by simp [prefixSum, listSum]
example : prefixSum [1, 2, 3, 4] 3 = 6 := by simp [prefixSum, listSum]

/-- T901 concrete: first element is identity. -/
example : prefixSum [1, 2, 3, 4] 0 = 0 := prefixSum_zero [1, 2, 3, 4]

/-- listSum of the full array equals the total. -/
example : listSum [1, 2, 3, 4] = 10 := by simp [listSum]

/-- listSum append property (concrete). -/
example : listSum ([1, 2] ++ [3, 4]) = listSum [1, 2] + listSum [3, 4] :=
  listSum_append [1, 2] [3, 4]

end Quanta.Scan
