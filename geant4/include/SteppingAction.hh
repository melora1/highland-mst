#ifndef SteppingAction_hh
#define SteppingAction_hh 1

#include "G4UserSteppingAction.hh"

class G4Step;
class RunAction;

class SteppingAction : public G4UserSteppingAction {
 public:
  explicit SteppingAction(RunAction* runAction);
  ~SteppingAction() override = default;

  void UserSteppingAction(const G4Step* step) override;

 private:
  RunAction* fRunAction = nullptr;
};

#endif
