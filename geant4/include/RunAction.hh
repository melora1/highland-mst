// RunAction.hh
//
// Opens/closes the plain-text output file SteppingAction writes to: one
// theta_space value (rad) per line, exactly the format
// results_pipeline's geant4_compare.py expects (see its --file argument).

#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

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

  void WriteTheta(G4double thetaRad);   // called from SteppingAction

private:
  G4String fOutFileName;
  std::ofstream fOut;
  G4long fNWritten = 0;
};

#endif
