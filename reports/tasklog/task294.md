
## 2026-06-30 (leak-audit recovery) — FIXED broken net (real 0 -> 17.05)

Leak audit (fresh_pass n=3000) flagged task294 at **100% fresh failure** AND it
failed all 265 stored examples: the deployed/source net implemented the WRONG rule
(full 3x3 9-cell gray block), so it scored ~0 on the real LB while the manifest
counted a fictional 18.19 (`local_stored_ok:False`).

True rule (cristianoc task294, verified 500/500 vs generator): 10x10 grid; a cell
-> 2 iff gray(5) AND all 4 ORTHOGONAL in-grid neighbours gray(5) (5-cell plus).

New net: Slice gray ch5 to native 10x10 -> cross Conv (count gray over the plus)
-> Equal==5 -> Pad to 30x30 -> Where(interior, color2, FREE input).
- mem=2800, params=37, **points=17.049**, stored 265/265, **fresh 5000/5000**.
