#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4VisAttributes.hh"

DetectorConstruction::DetectorConstruction()
    : fMaterialName("G4_Cu"), fThickness(15.0 * cm) {
  fMessenger = new DetectorMessenger(this);
}

DetectorConstruction::~DetectorConstruction() { delete fMessenger; }

G4double DetectorConstruction::GetScoringZ() const {
  // Scoring plane sits immediately downstream of the slab, with a small
  // (1 mm) vacuum gap so the boundary between slab and scoring volume is
  // unambiguous to the navigator.
  return GetSlabHalfZ() + 0.1 * cm + kScoringHalfZ * cm;
}

G4VPhysicalVolume* DetectorConstruction::Construct() {
  G4NistManager* nist = G4NistManager::Instance();

  G4Material* worldMat = nist->FindOrBuildMaterial("G4_Galactic");
  G4Material* slabMat = nist->FindOrBuildMaterial(fMaterialName);
  if (!slabMat) {
    G4Exception("DetectorConstruction::Construct", "MatNotFound",
                FatalException, ("Unknown material: " + fMaterialName).c_str());
  }

  // ---- world ----
  G4Box* solidWorld =
      new G4Box("World", kWorldHalf * cm, kWorldHalf * cm, kWorldHalf * cm);
  G4LogicalVolume* logicWorld =
      new G4LogicalVolume(solidWorld, worldMat, "World");
  G4VPhysicalVolume* physWorld = new G4PVPlacement(
      nullptr, G4ThreeVector(), logicWorld, "World", nullptr, false, 0);

  // ---- slab, centred at z = 0 ----
  G4double halfZ = GetSlabHalfZ();
  G4Box* solidSlab = new G4Box("Slab", kTransverseHalf * cm,
                               kTransverseHalf * cm, halfZ);
  G4LogicalVolume* logicSlab =
      new G4LogicalVolume(solidSlab, slabMat, "Slab");
  new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), logicSlab, "Slab",
                    logicWorld, false, 0);

  // ---- thin scoring plane just downstream of the slab ----
  G4Box* solidScore = new G4Box("ScoringPlane", kTransverseHalf * cm,
                                kTransverseHalf * cm, kScoringHalfZ * cm);
  G4LogicalVolume* logicScore =
      new G4LogicalVolume(solidScore, worldMat, "ScoringPlane");
  new G4PVPlacement(nullptr, G4ThreeVector(0, 0, GetScoringZ()), logicScore,
                    "ScoringPlane", logicWorld, false, 0);

  logicSlab->SetVisAttributes(
      new G4VisAttributes(G4Colour(0.6, 0.6, 0.9, 0.4)));
  logicScore->SetVisAttributes(new G4VisAttributes(G4Colour(1, 0, 0)));

  return physWorld;
}
