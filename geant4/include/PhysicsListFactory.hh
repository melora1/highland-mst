// PhysicsListFactory.hh
//
// Builds the two MSC-model configurations the manuscript compares
// (Sec. 3, Sec. "Geant4 cross-check"): Urban (Geant4's default MSC model)
// and Wentzel-VI + single Coulomb scattering.
//
// PRIMARY METHOD: Geant4's reference physics lists support a documented
// "_WVI" suffix that swaps the EM constructor's MSC model to Wentzel-VI
// (G4EmStandardPhysicsWVI) while keeping everything else (hadronic model,
// decay, etc.) identical to the base list. This is the recommended,
// version-stable way to get a clean Urban-vs-WentzelVI comparison:
//     "urban"   -> G4PhysListFactory::GetReferencePhysList("FTFP_BERT")
//     "wentzel" -> G4PhysListFactory::GetReferencePhysList("FTFP_BERT_WVI")
// FTFP_BERT is used only as a carrier for the EM constructor; no hadronic
// interaction physics is exercised by muons in a thin slab, so any baseline
// reference list works identically here -- FTFP_BERT is simply Geant4's
// standard default.
//
// FALLBACK: if "_WVI" is not recognized by the installed Geant4 version
// (rare; check with `geant4-config --version`, requires >= 10.0), build the
// physics list manually: register G4EmStandardPhysics_option4 for "urban",
// or write a custom G4VPhysicsConstructor that calls G4EmConfigurator to
// replace mu-/mu+ (and e-/e+, if desired) MSC models with G4WentzelVIModel
// + G4CoulombScattering for "wentzel". This is the standard low-level
// recipe (see e.g. Geant4 "local physics list" examples) but is more code
// and more version-sensitive than the factory suffix above, so it is not
// implemented here; add it only if FTFP_BERT_WVI is unavailable.

#ifndef PHYSICS_LIST_FACTORY_HH
#define PHYSICS_LIST_FACTORY_HH

#include "G4VUserPhysicsList.hh"
#include "globals.hh"

// Returns a heap-allocated physics list; caller (main) passes it to
// G4RunManager::SetUserInitialization, which takes ownership.
// model: "urban" or "wentzel" (case-insensitive).
G4VUserPhysicsList* BuildPhysicsList(const G4String& model);

#endif
