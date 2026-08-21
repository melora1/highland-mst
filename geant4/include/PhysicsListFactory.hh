#ifndef PhysicsListFactory_hh
#define PhysicsListFactory_hh 1

#include "globals.hh"

class G4VUserPhysicsList;

G4VUserPhysicsList* BuildPhysicsList(const G4String& modelIn);

#endif
