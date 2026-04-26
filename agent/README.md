# agent (stub)

Merges user intent + AI suggestions from `analysis` → EditPlan consumed by `composer`.

Input:
- user EditPlan (segments, effects)
- features.json from analysis

Output:
- augmented EditPlan with `ai_suggestions`
