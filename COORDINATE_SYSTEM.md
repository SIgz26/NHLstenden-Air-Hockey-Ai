# Orion Vision — Coördinaten- en Assenstelsel Referentie

## 1. Fysieke Werkelijkheid & CODESYS PLC (Single Source of Truth)
- **Eenheden:** Posities in `mm` (`GVL_Machine.pose.rX`, `pose.rY`), snelheden in `mm/s` (`GVL.rTargetVx`, `GVL.rTargetVy`).
- **As X (Breedte / Zijkant-naar-zijkant):**
  - Fysiek bereik (`GVL_Kin`): `-513.2 mm` tot `+420.5 mm`
  - Veilig PLC-bereik: **`-463.2 mm` tot `+370.5 mm`** (span `833.7 mm`, tafelmidden op `X = -46.35 mm`)
  - Richting: Positieve `Vx` verhoogt `X`.
- **As Y (Diepte / Goal-naar-goal):**
  - Fysiek bereik (`GVL_Kin`): `3.7 mm` (vooraanslag bij middenlijn) tot `839.5 mm` (achteraanslag bij eigen robotdoel)
  - Veilig PLC-bereik: **`53.7 mm` (voorgrens / middenlijn)** tot **`789.5 mm` (eigen robotdoel)** (span `735.8 mm`)
  - Richting: Positieve `Vy` beweegt richting eigen doel (`789.5 mm`); **negatieve `Vy` beweegt naar voren** het veld in (`53.7 mm`).
- **H-brug Kinematica & Dynamiek (`GVL_Kin` / `FB_CoupleHBot`):**
  - `X = (A + B) / 2`, `Y = (B - A) / 2` (geen motor-as-inversie).
  - **Asymmetrische snelheidslimieten:**
    - Voorwaarts (`rTargetVy < 0`, richting middenlijn `rFieldYMin = 53.7 mm`): maximaal **`1000.0 mm/s`** (`1.0 m/s`).
    - Achterwaarts (`rTargetVy > 0`, richting eigen doel `789.5 mm`) en zijwaarts (`rTargetVx`): maximaal **`850.0 mm/s`** (`0.85 m/s`).
    - Gezamenlijke motorlimiet: `rVelMaxMotor = 1000.0` as-eenheden/s met proportionele vectorbegrenzing (richting blijft behouden).
    - Versnelling/vertraging: `rAccMotor = 2800.0 mm/s²`, `rDecMotor = 4200.0 mm/s²`, `rJerkMotor = 45000.0 mm/s³`.
  - **Randdemping (`rEdgeMargin = 95.0 mm`):**
    - Negatieve Y-grens (`rFieldYMin = 53.7 mm`): lineaire snelheidsafbouw vanaf **`360.0 mm`** tot `0 mm/s` op **`95.0 mm`** vóór `rFieldYMin`.
    - Overige grenzen (`rFieldYMax`, `rFieldXMin`, `rFieldXMax`): lineaire snelheidsafbouw vanaf **`250.0 mm`** tot `0 mm/s` op **`95.0 mm`** vóór de veldgrens.

## 2. Volledige Tafel vs. Robot Werkgebied (Python / Camera / AI)
- **Volledige fysieke tafel (Camera 4-punts kalibratie & Puck):** `1940 mm` (lengte goal-to-goal) × `1020 mm` (breedte side-to-side), met middenlijn op `0 mm`.
- **4-punts Camerahomografie volgorde (`TL, TR, BR, BL`):**
  - `1 (TL)`: Tegenstander-kant, zijwand `X_min` (`-510 mm, -970 mm`)
  - `2 (TR)`: Tegenstander-kant, zijwand `X_max` (`+510 mm, -970 mm`)
  - `3 (BR)`: Eigen robot-kant, zijwand `X_max` (`+510 mm, +970 mm`)
  - `4 (BL)`: Eigen robot-kant, zijwand `X_min` (`-510 mm, +970 mm`)
- **Robot werkgebied:** Beslaat uitsluitend de eigen tafelhelft van `Y = 789.5 mm` (eigen doel) tot `Y = 53.7 mm` (middenlijn).

## 3. Overzicht per Softwarelaag
| Laag | Lengte-as (Goal ↔ Goal) | Breedte-as (Zijkant ↔ Zijkant) | Nulpunt & Richting |
| :--- | :--- | :--- | :--- |
| **CODESYS PLC** | `PLC-Y` (`53.7 .. 789.5 mm`) | `PLC-X` (`-463.2 .. 370.5 mm`) | `Y = 789.5` = eigen doel; `-Vy` = naar voren (max `1000 mm/s`), `+Vy` & `Vx` max `850 mm/s`. |
| **Virtual Twin (Scherm)** | Horizontaal (`-1` links = doel, `+1` rechts) | Verticaal (`-1` onder, `+1` boven) | `Vy`-pijl horizontaal (`-Vy` wijst naar rechts/voren), `Vx`-pijl verticaal. |
| **Scripted Opponents** | Index `[0]`: puck in full-table-frame (`-0.97 .. +0.97 m`); robot: PLC-Y `53.7 .. 789.5 mm` → `0.0 .. +0.7358 m` (1-op-1, zonder rek) | Index `[1]`: puck in full-table-frame (`-0.51 .. +0.51 m`); robot: PLC-X gecentreerd rond `-46.35 mm` (`-0.4169 .. +0.4169 m`) | Voor de robot is `0.0 m` de middenlijn en `+0.7358 m` het eigen doel; `-vel[0]` = naar voren (`-rTargetVy`, max `1.0 m/s`). |
| **SAC Agent (RL)** | Index `[0]` (`-1.0 .. 1.0`) | Index `[1]` (`-1.0 .. 1.0`) | `-1.0` = eigen doel, `0.0` = middenlijn (`Y = 53.7`); `+act[0]` = naar voren (`-rTargetVy`). |

## 4. SAC Agent (RL-Policy: `air_hockey_env.py` ↔ `SACController` / `AgentWorker`)
- **Observatievector (`obs[0..7]` in `[-1.0, 1.0]`, lengte eerst, breedte tweede):**
  - As-volgorde uit `SimBridge` herschikken via indices `[1, 0, 3, 2, 5, 4, 7, 6]`.
  - `obs[2], obs[3]`: fysieke pucksnelheid in `m/s` gedeeld door `5.0 m/s`, geclipt op `[-1.0, 1.0]`.
  - `obs[6]`: robot-lengte herprojecteren naar eigen helft `[-1.0, 0.0]` via `(sim_robot_y - 1.0) / 2.0` (`-1.0` = eigen doel `789.5 mm`, `0.0` = middenlijn `53.7 mm`).
- **Actievector (`action[0..1]` in `[-1.0, 1.0]`):**
  - `rTargetVx = action[1] * 850.0` (`mm/s`), `rTargetVy = -action[0] * (1000.0 if action[0] > 0 else 850.0)` (`mm/s`).