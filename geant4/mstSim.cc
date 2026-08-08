// mstSim.cc  --  Geant4 cross-check for "The Highland Constant and the
// Divergent Second Moment", Sec. 3 ("Geant4 cross-check").
//
// Fires a monoenergetic mu- beam through a single material slab and records
// each primary's exit angle (theta_space, rad) to a plain-text file --
// exactly the input format geant4_compare.py expects.
//
// Usage:
//   ./mstSim <model> <material> <thickness_cm> <p_GeV> <nEvents> <outFile>
//     model        : urban | wentzel
//     material     : Cu | Pb   (NIST names G4_Cu / G4_Pb are also accepted)
//     thickness_cm : slab thickness along z, in cm
//     p_GeV        : muon momentum in GeV/c (kinetic energy computed here)
//     nEvents      : number of primaries
//     outFile      : output text file path
//
// Example (matches one row of run_sweep.sh):
//   ./mstSim urban Cu 15.0 1.0 500000 out/Cu_t15.0_p1.0_urban.txt
//
// Then:
//   python3 geant4_compare.py --file out/Cu_t15.0_p1.0_urban.txt \
//       --material Cu --thickness_cm 15.0 --p 1.0 --model urban
//
// Not built/run in this environment -- see README.md for build instructions
// against an installed Geant4 (>= 10.7, tested pattern targets 11.x).

#include "DetectorConstruction.hh"
#include "PhysicsListFactory.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
constexpr double kMuMassMeV = 105.6583715;

double KineticEnergyMeV(double p_GeV) {
  double p = p_GeV * 1000.0;
  double E = std::sqrt(p * p + kMuMassMeV * kMuMassMeV);
  return E - kMuMassMeV;
}

std::string ResolveMaterial(const std::string& in) {
  if (in == "Cu" || in == "G4_Cu") return "G4_Cu";
  if (in == "Pb" || in == "G4_Pb") return "G4_Pb";
  std::cerr << "Unknown material '" << in << "'; expected Cu or Pb.\n";
  std::exit(1);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <urban|wentzel> <Cu|Pb> <thickness_cm> <p_GeV> "
                 "<nEvents> <outFile>\n";
    return 1;
  }
  const std::string model = argv[1];
  const std::string material = ResolveMaterial(argv[2]);
  const double thicknessCm = std::atof(argv[3]);
  const double pGeV = std::atof(argv[4]);
  const long nEvents = std::atol(argv[5]);
  const std::string outFile = argv[6];

  const double keMeV = KineticEnergyMeV(pGeV);
  std::cout << "[mstSim] model=" << model << " material=" << material
           << " t=" << thicknessCm << " cm  p=" << pGeV
           << " GeV/c  (KE=" << keMeV << " MeV)  N=" << nEvents << "\n";

  auto* runManager =
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);

  auto* detector = new DetectorConstruction();
  detector->SetMaterialName(material);
  detector->SetThickness(thicknessCm * cm);
  runManager->SetUserInitialization(detector);

  runManager->SetUserInitialization(BuildPhysicsList(model));

  auto* primaryGen = new PrimaryGeneratorAction();
  runManager->SetUserAction(primaryGen);

  auto* runAction = new RunAction(outFile);
  runManager->SetUserAction(runAction);
  runManager->SetUserAction(new SteppingAction(runAction));

  runManager->Initialize();

  G4UImanager* ui = G4UImanager::GetUIpointer();
  ui->ApplyCommand("/gun/particle mu-");
  ui->ApplyCommand("/gun/energy " + std::to_string(keMeV) + " MeV");

  runManager->BeamOn(nEvents);

  delete runManager;
  return 0;
}
