// DetectorMessenger.hh
//
// Exposes:
//   /det/setMaterial G4_Cu | G4_Pb
//   /det/setThickness <value> <unit>
// Must be issued in a macro (or via UI) BEFORE /run/initialize, since
// Construct() reads DetectorConstruction's stored values when the geometry
// is actually built.

#ifndef DETECTOR_MESSENGER_HH
#define DETECTOR_MESSENGER_HH

#include "G4UImessenger.hh"
#include "globals.hh"

class DetectorConstruction;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;

class DetectorMessenger : public G4UImessenger {
public:
  explicit DetectorMessenger(DetectorConstruction* det);
  ~DetectorMessenger() override;

  void SetNewValue(G4UIcommand* command, G4String newValue) override;

private:
  DetectorConstruction* fDetector;
  G4UIdirectory* fDetDir;
  G4UIcmdWithAString* fMaterialCmd;
  G4UIcmdWithADoubleAndUnit* fThicknessCmd;
};

#endif
