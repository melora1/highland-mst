#ifndef DetectorConstruction_hh
#define DetectorConstruction_hh 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class DetectorMessenger;
class G4VPhysicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction {
 public:
  DetectorConstruction();
  ~DetectorConstruction() override;

  G4VPhysicalVolume* Construct() override;

  void SetMaterialName(const G4String& name) { fMaterialName = name; }
  void SetThickness(G4double thickness) { fThickness = thickness; }

  const G4String& GetMaterialName() const { return fMaterialName; }
  G4double GetThickness() const { return fThickness; }
  G4double GetSlabHalfZ() const { return 0.5 * fThickness; }
  G4double GetScoringZ() const;

 private:
  G4String fMaterialName;
  G4double fThickness = 0.0;
  DetectorMessenger* fMessenger = nullptr;

  // Geometry dimensions are expressed in cm in DetectorConstruction.cc.
  static constexpr G4double kWorldHalf = 100.0;
  static constexpr G4double kTransverseHalf = 50.0;
  static constexpr G4double kScoringHalfZ = 0.005;  // 0.05 mm half-thickness
};

#endif
