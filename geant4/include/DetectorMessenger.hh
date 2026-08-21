#ifndef DetectorMessenger_hh
#define DetectorMessenger_hh 1

#include "G4UImessenger.hh"
#include "globals.hh"

class DetectorConstruction;
class G4UIcommand;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;

class DetectorMessenger : public G4UImessenger {
 public:
  explicit DetectorMessenger(DetectorConstruction* detector);
  ~DetectorMessenger() override;

  void SetNewValue(G4UIcommand* command, G4String newValue) override;

 private:
  DetectorConstruction* fDetector = nullptr;
  G4UIdirectory* fDetDir = nullptr;
  G4UIcmdWithAString* fMaterialCmd = nullptr;
  G4UIcmdWithADoubleAndUnit* fThicknessCmd = nullptr;
};

#endif
