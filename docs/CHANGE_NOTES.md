# Correctness and reproducibility changes

The release fixes frontend/backend contracts that previously appeared functional while silently falling back or ignoring parameters.

- The station endpoint applies viewport bounds before pagination and counting; invalid or partial bounds are rejected. Nearby search uses vectorized distances.
- Pipeline filtering preserves source indexes. Segmentation has a consistent three-value return contract, uses latest-year model averages, and leaves invalid/pre-2022 records unlabeled.
- The API recomputes segment labels using the same rules as the offline pipeline, rather than trusting stale labels in older exports. EPA IDs identify records across refreshes.
- Name grouping uses the bundled lookup and deterministic fallback. The absent live AI endpoint and AI service dependency are removed.
- The UI propagates abort signals, suppresses obsolete responses, surfaces request errors, and does not turn null segment labels into segment zero.
- Manufacturer trend gates run in shared code used by both the analysis and API. Segment labels describe measured features rather than implying body-style classification.
- Unused synthetic sales are removed from the application snapshot. Offline synthetic sales require explicit opt-in and retain provenance in metadata and chart titles.
- Setup, Docker paths/port defaults, production deep links, dependencies, and CI are corrected. Unused 3D code and local development files are removed.

## Deployment

After merging, rebuild the Docker image and verify `/ready`, `/compare`, `/chargers`, `/segments`, and `/trends`. The existing hosted demo does not receive these changes merely because a pull request exists. The data source snapshot remains May 22, 2026; changed segment results are a method correction, not fresh market data.

Normalization and enriched specifications remain heuristic. Segment membership and counts will differ from the previous release because row alignment and averaging have changed.
