#include "PhysicsListFactory.hh"

#include "G4PhysListFactory.hh"
#include "G4EmParameters.hh"

#include <algorithm>

G4VUserPhysicsList* BuildPhysicsList(const G4String& modelIn) {
  G4String model = modelIn;
  std::transform(model.begin(), model.end(), model.begin(), ::tolower);

  G4String listName;
  if (model == "urban") {
    listName = "FTFP_BERT";       // default MSC = Urban
  } else if (model == "wentzel") {
    listName = "FTFP_BERT_WVI";   // MSC = Wentzel-VI + single scattering
  } else {
    G4Exception("BuildPhysicsList", "BadModel", FatalException,
               ("Unknown MSC model '" + model +
                "'; expected 'urban' or 'wentzel'").c_str());
  }

  G4PhysListFactory factory;
  G4VUserPhysicsList* physicsList = factory.GetReferencePhysList(listName);
  if (!physicsList) {
    G4Exception("BuildPhysicsList", "ListNotFound", FatalException,
               ("Physics list '" + listName + "' not available in this "
                "Geant4 build -- see PhysicsListFactory.hh for the manual "
                "fallback recipe.").c_str());
  }

  // Keep multiple scattering step limitation at its default (UseSafety);
  // do not override MSC parameters beyond what the reference list already
  // sets, so the comparison reflects each model's own recommended defaults
  // rather than a hand-tuned configuration.
  return physicsList;
}
