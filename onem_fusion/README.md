# onem_fusion

Multimodal feature fusion module for OmniMedAI. It aligns radiomics, pathomics,
habitat, clinical, genomic, and deep-learning feature tables by patient/sample ID.

## Core Capabilities

- Align multiple feature tables by a shared ID column.
- Add modality prefixes to avoid feature-name collisions.
- Support inner, left, right, and outer joins.
- Optionally fill missing values with zero, mean, or median.
- Export modeling-ready fused CSV files.

## Quick Start

```python
from onem_fusion import FeatureFusion, FusionConfig

fusion = FeatureFusion(FusionConfig(
    id_column="patient_id",
    join_strategy="inner",
    modality_prefixes={
        "radiomics": "rad",
        "clinical": "clin",
        "pathology": "path"
    }
))

fused = fusion.fit_transform({
    "radiomics": "output/radiomics_features.csv",
    "clinical": "data/clinical.csv",
    "pathology": "output/pathology_features.csv"
})

fusion.save(fused, "output/fused_features.csv")
```

## Status

This module provides the first platform-level fusion layer. Advanced fusion
methods such as late fusion, MIL fusion, attention-based fusion, and survival
fusion can be added on top of this interface.
