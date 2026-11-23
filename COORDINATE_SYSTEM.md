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
- **Mallet:** diameter `95.0 mm`, straal `47.5 mm`.
- **Puck:** diameter `73.0 mm`, straal `36.5 mm`; contactafstand hart-op-hart `84.0 mm`.
- De gedeelde SI- en mm-constanten staan in `Orion Vision/core/coordinate_contract.py`; zie ook [de fysieke coördinatenreferentie](docs/coordinate_reference.md).
- **4-punts Camerahomografie volgorde (`TL, TR, BR, BL`):**
  - `1 (TL)`: Tegenstander-kant, zijwand `X_min` (`-510 mm, -970 mm`)
  - `2 (TR)`: Tegenstander-kant, zijwand `X_max` (`+510 mm, -970 mm`)
  - `3 (BR)`: Eigen robot-kant, zijwand `X_max` (`+510 mm, +970 mm`)
  - `4 (BL)`: Eigen robot-kant, zijwand `X_min` (`-510 mm, +970 mm`)
- **Robot werkgebied:** Beslaat uitsluitend de eigen tafelhelft van `Y = 789.5 mm` (eigen doel) tot `Y = 53.7 mm` (middenlijn).

## 3. Gedeeld RL-contract (`core/coordinate_contract.py`)
- Alle fysieke toestanden gebruiken SI-eenheden en `[lengte, breedte]`-volgorde.
- De trainingslengte-as is positief naar de tegenstander; de robot speelt op negatieve lengte. PLC-Y is positief naar het eigen doel, dus `training_length = -(PLC-Y - 53.7 mm)`.
- Training- en live tafelafmetingen zijn identiek: `1.94 m` lang × `1.02 m` breed.
- Het actieve robotgebied met `95 mm` randmarge is lengte `[-0.6408, -0.095] m` en breedte `[-0.415, +0.415] m`.

### Observatie (12D, `[-1, 1]`)
`air_hockey_env.py` en de 8D SimBridge-adapter gebruiken dezelfde functie `physical_state_to_observation`:
1. `[0:2]` puckpositie `[lengte, breedte]`, gedeeld door de halve veldafmetingen.
2. `[2:4]` pucksnelheid `[lengte, breedte]`, gedeeld door `5.0 m/s`.
3. `[4:6]` tegenstanderpositie, gedeeld door de halve veldafmetingen; zonder marker staat de tegenstander op `(0.36 * veldlengte, 0)`.
4. `[6:8]` robotpositie, gedeeld door de halve veldafmetingen.
5. `[8:10]` puckpositie minus robotpositie, gedeeld door de volledige veldafmetingen.
6. `[10:12]` puck- minus robotsnelheid, gedeeld door `6.0 m/s`.

SimBridge levert voor de SAC-adapter SI-posities/snelheden uit de gekalibreerde homografie en PLC-pose. De PLC-robotsnelheid is een robuuste mediaan van recente signed `Δpos/Δt`-hellingen, met een korte laagdoorlaatfilter; zonder verse metingen wordt `None` gebruikt en vult het contract nul in.

### Actie en fysica
- SAC-acties zijn `[lengtesnelheid, breedtesnelheid]`, genormaliseerd in `[-1, 1]`.
- De gedeelde actieconversie begrenst voorwaarts op `1.0 m/s`, achterwaarts/zijwaarts op `0.85 m/s` en de gezamenlijke H-bot-motorsnelheid op `1.0 m/s`. De PLC krijgt dezelfde vector in zijn assen (`Vx = breedte`, `Vy = -lengte`) en in `mm/s`.
- De trainingsomgeving gebruikt `10 ms` stappen, `2.8 m/s²` motorversnelling, dezelfde snelheidslimieten, het actieve werkgebied en één stap actievertraging.
- Een 8D Live View-waarneming wordt alleen met gekalibreerde tafelcoördinaten naar deze 12D-contractvector omgezet; onbekende modeldimensies blijven afgewezen.