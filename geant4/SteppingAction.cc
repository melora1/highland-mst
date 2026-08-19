#include "SteppingAction.hh"
#include "RunAction.hh"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"

#include <cmath>

SteppingAction::SteppingAction(RunAction* runAction)
    : fRunAction(runAction) {}

void SteppingAction::UserSteppingAction(const G4Step* step) {
  const G4Track* track = step->GetTrack();
  if (track->GetTrackID() != 1) return;  // primary only

  const G4StepPoint* pre = step->GetPreStepPoint();
  const G4StepPoint* post = step->GetPostStepPoint();
  if (!post->GetPhysicalVolume()) return;  // stepped out of world

  const G4String preVol = pre->GetPhysicalVolume()
                              ? pre->GetPhysicalVolume()->GetName()
                              : G4String("");
  const G4String postVol = post->GetPhysicalVolume()->GetName();

  // Boundary-crossing step INTO the scoring plane (not already inside it).
  if (postVol == "ScoringPlane" && preVol != "ScoringPlane") {
    const G4ThreeVector dir = post->GetMomentumDirection();  // unit vector
    // theta_space relative to the beam axis (0,0,1); dir.z() > 0 always for
    // a forward-going primary at these momenta, so acos is well-defined.
    G4double theta = std::acos(std::min(1.0, std::max(-1.0, dir.z())));
    fRunAction->WriteTheta(theta);  // rad, Geant4's native angle unit
    const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
  }
}
