#include "PhysicsListFactory.hh"

#include "G4PhysListFactory.hh"
#include "G4EmParameters.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicsConstructor.hh"
#include "G4PhysicsListHelper.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4MuMultipleScattering.hh"
#include "G4CoulombScattering.hh"
#include "G4WentzelVIModel.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"

#include <algorithm>

// ---------------------------------------------------------------------------
// Physics constructor that adds a discrete muon single-Coulomb-scattering tail.
//
// Registered as an EXTRA constructor on the FTFP_BERT_WVI reference list, its
// ConstructProcess() runs AFTER the reference list has built its own muon
// processes -- so we can remove the reference muon MSC and install
// G4MuMultipleScattering(WentzelVI) + discrete G4CoulombScattering in its place.
//
// Diagnostic purpose: this optional constructor makes the muon process
// configuration explicit by replacing the reference-list muon MSC with a
// Wentzel-VI MSC model plus discrete G4CoulombScattering.  It is a separate
// transport configuration, not assumed a priori to be more physical than the
// unmodified reference lists; the executable prints the installed muon process
// names so benchmark provenance can be recorded with each run.
// ---------------------------------------------------------------------------
class MuonSingleScatter : public G4VPhysicsConstructor {
 public:
  MuonSingleScatter() : G4VPhysicsConstructor("MuonSingleScatter") {}
  void ConstructParticle() override {}  // particles already defined by base list
  void ConstructProcess() override {
    G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
    for (auto* part : {static_cast<G4ParticleDefinition*>(G4MuonMinus::MuonMinus()),
                       static_cast<G4ParticleDefinition*>(G4MuonPlus::MuonPlus())}) {
      G4ProcessManager* pm = part->GetProcessManager();
      if (!pm) continue;

      // Strip the reference list's muon multiple scattering so the two MSC
      // implementations do not stack.
      G4ProcessVector* plist = pm->GetProcessList();
      for (G4int i = plist->size() - 1; i >= 0; --i) {
        G4VProcess* proc = (*plist)[i];
        const G4String& pn = proc->GetProcessName();
        if (pn == "muMsc" || pn == "msc") {
          pm->RemoveProcess(proc);
        }
      }

      auto* msc = new G4MuMultipleScattering();
      msc->SetEmModel(new G4WentzelVIModel());
      ph->RegisterProcess(msc, part);
      ph->RegisterProcess(new G4CoulombScattering(), part);
    }
  }
};

G4VUserPhysicsList* BuildPhysicsList(const G4String& modelIn) {
  G4String model = modelIn;
  std::transform(model.begin(), model.end(), model.begin(), ::tolower);

  const bool enableMuSingleScatter = (model == "wvi_ss");

  G4String listName;
  if (model == "ftfp_bert") {
    listName = "FTFP_BERT";       // unmodified reference list
  } else if (model == "ftfp_bert_wvi") {
    listName = "FTFP_BERT_WVI";   // unmodified WVI reference list
  } else if (model == "wvi_ss") {
    listName = "FTFP_BERT_WVI";   // base list; muon single scattering added below
  } else {
    G4Exception("BuildPhysicsList", "BadModel", FatalException,
               ("Unknown MSC model '" + model +
                "'; expected 'ftfp_bert', 'ftfp_bert_wvi', or 'wvi_ss'").c_str());
  }

  G4PhysListFactory factory;
  G4VUserPhysicsList* physicsList = factory.GetReferencePhysList(listName);
  if (!physicsList) {
    G4Exception("BuildPhysicsList", "ListNotFound", FatalException,
               ("Physics list '" + listName + "' not available in this "
                "Geant4 build -- see PhysicsListFactory.hh for the manual "
                "fallback recipe.").c_str());
  }

  if (enableMuSingleScatter) {
    auto* modList = dynamic_cast<G4VModularPhysicsList*>(physicsList);
    if (!modList) {
      G4Exception("BuildPhysicsList", "NotModular", FatalException,
                 "wvi_ss requires a modular reference list; FTFP_BERT_WVI "
                 "did not down-cast to G4VModularPhysicsList.");
    }
    G4EmParameters::Instance()->SetMscThetaLimit(0.15);  // rad
    modList->RegisterPhysics(new MuonSingleScatter());
  }

  return physicsList;
}
