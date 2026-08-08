// DetectorConstruction.hh
//
// A single material slab (Cu or Pb, thickness set at run time) on the beam
// axis, followed immediately by a thin vacuum "scoring plane" the primary
// muon's exit angle is recorded at. Geometry parameters (material,
// thickness) are set via UI commands BEFORE /run/initialize -- Construct()
// reads them when the geometry is actually built.
//
// Manuscript Sec. 3 (Geant4 cross-check): "Slabs of copper and lead spanning
// the path-length range of the target" -- the material/thickness pairs used
// in the sweep (see run_sweep.sh) are chosen to match the x/X0 values quoted
// in the manuscript's per-material k_opt table (Sec. 5.2): Cu at x/X0 =
// 2.08 and 10.42 (t = 3.0, 15.0 cm); Pb at x/X0 = 3.57 and 14.29 (t = 2.0,
// 8.0 cm), using PDG X0: Cu 1.44 cm, Pb 0.56 cm.

#ifndef DETECTOR_CONSTRUCTION_HH
#define DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4LogicalVolume;
class DetectorMessenger;

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
  DetectorConstruction();
  ~DetectorConstruction() override;

  G4VPhysicalVolume* Construct() override;

  // UI-settable parameters (Construct() reads these at /run/initialize time)
  void SetMaterialName(const G4String& name) { fMaterialName = name; }
  void SetThickness(G4double t) { fThickness = t; }

  G4double GetSlabHalfZ() const { return 0.5 * fThickness; }
  G4double GetScoringZ() const;  // world-frame z of the scoring plane centre

private:
  G4String fMaterialName;   // "G4_Cu" or "G4_Pb"
  G4double fThickness;      // full thickness along z (Geant4 internal units)
  DetectorMessenger* fMessenger;

  static constexpr G4double kTransverseHalf = 50.0;   // cm; slab is 1 m x 1 m
  static constexpr G4double kWorldHalf = 200.0;        // cm
  static constexpr G4double kScoringHalfZ = 0.0005;    // cm (5 micron)
};

#endif
