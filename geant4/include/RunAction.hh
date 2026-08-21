#ifndef RunAction_hh
#define RunAction_hh 1

#include "G4UserRunAction.hh"
#include "globals.hh"

#include <fstream>

class G4Run;

class RunAction : public G4UserRunAction {
 public:
  explicit RunAction(const G4String& outFileName);
  ~RunAction() override;

  void BeginOfRunAction(const G4Run* run) override;
  void EndOfRunAction(const G4Run* run) override;

  void WriteTheta(G4double thetaRad);

 private:
  G4String fOutFileName;
  std::ofstream fOut;
  G4long fNWritten = 0;
};

#endif
