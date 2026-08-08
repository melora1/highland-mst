// SteppingAction.hh
//
// On the step where the PRIMARY track (trackID==1) crosses the boundary
// into the "ScoringPlane" volume, records theta_space = the angle between
// its post-step momentum direction and the beam axis (0,0,1) -- the exact
// quantity the manuscript's Sec. 3 quadrature-vs-Geant4 comparison and
// geant4_compare.py both use. The track is then killed (fStopAndKill) so
// it is counted exactly once, even if it re-crosses the thin plane.
//
// Secondaries (delta rays, bremsstrahlung photons, etc., trackID != 1) are
// never scored here -- only the primary's own deflection is of interest,
// matching the manuscript's "space-angle RMS ... extracted from the
// simulated angles" for the PRIMARY muon.

#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class RunAction;

class SteppingAction : public G4UserSteppingAction {
public:
  explicit SteppingAction(RunAction* runAction);
  ~SteppingAction() override = default;

  void UserSteppingAction(const G4Step* step) override;

private:
  RunAction* fRunAction;
};

#endif
