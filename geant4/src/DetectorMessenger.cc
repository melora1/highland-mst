#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"

DetectorMessenger::DetectorMessenger(DetectorConstruction* det)
    : fDetector(det) {
  fDetDir = new G4UIdirectory("/det/");
  fDetDir->SetGuidance("Detector (target slab) geometry commands.");

  fMaterialCmd = new G4UIcmdWithAString("/det/setMaterial", this);
  fMaterialCmd->SetGuidance("Set slab material (NIST name, e.g. G4_Cu, G4_Pb).");
  fMaterialCmd->SetParameterName("material", false);
  fMaterialCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fThicknessCmd = new G4UIcmdWithADoubleAndUnit("/det/setThickness", this);
  fThicknessCmd->SetGuidance("Set slab thickness along z.");
  fThicknessCmd->SetParameterName("thickness", false);
  fThicknessCmd->SetUnitCategory("Length");
  fThicknessCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
}

DetectorMessenger::~DetectorMessenger() {
  delete fMaterialCmd;
  delete fThicknessCmd;
  delete fDetDir;
}

void DetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue) {
  if (command == fMaterialCmd) {
    fDetector->SetMaterialName(newValue);
  } else if (command == fThicknessCmd) {
    fDetector->SetThickness(fThicknessCmd->GetNewDoubleValue(newValue));
  }
}
