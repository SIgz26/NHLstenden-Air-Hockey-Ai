# Air-hockey physical coordinate reference

`Orion Vision/core/coordinate_contract.py` is the codebase's single source of
truth for the table and object dimensions. Physics, opponent policies, the
SimBridge contract, and the Digital Twin reuse those SI constants.

## Table and object dimensions

| Object | Diameter / dimensions | Radius |
| --- | ---: | ---: |
| Table | 1940 mm (1.940 m) long × 1020 mm (1.020 m) wide | — |
| Mallet | 95.0 mm (0.095 m) | 47.5 mm (0.0475 m) |
| Puck | 73.0 mm (0.073 m) | 36.5 mm (0.0365 m) |

The mallet-to-puck contact distance, measured center to center, is the sum of
their radii: **84.0 mm (0.084 m)**.

## Physics and display

- Puck centers bounce at ±0.4735 m across the table width and ±0.9335 m along
  its length, except where the puck can enter a goal.
- The Digital Twin uses the table's exact 1940:1020 aspect ratio and scales
  puck and mallet circles from their physical radii.
- Vision-engine circle thresholds are expressed in image pixels, not
  millimeters; they remain camera-resolution-specific and are not physical
  geometry constants.

The contract also exports the corresponding millimeter values for components
whose interfaces use mm.
