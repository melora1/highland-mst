// mstSim.cc -- single-slab Geant4 transport benchmark.
//
// Fires a monoenergetic mu- beam through one material slab and records each
// primary exit angle (theta_space, rad), exactly the input expected by
// ../geant4_compare.py.
//
// Usage:
//   ./mstSim <urban|wentzel|wvi_ss> <Cu|Pb> <thickness_cm> <p_GeV>
//            <nEvents> <seed> <outFile>
//
// Example:
//   ./mstSim urban Cu 15.0 1.0 1000000 12345 out/Cu_t15_p1_urban_s12345.txt
//   python3 ../geant4_compare.py \
//       --file urban=out/Cu_t15_p1_urban_s12345.txt \
//       --material Cu --thickness-cm 15.0 --p 1.0 --n-generated 1000000 \
//       --out out/Cu_t15_p1_compare.csv

#include "DetectorConstruction.hh"
#include "PhysicsListFactory.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

#include "G4MuonMinus.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4RunManagerFactory.hh"
#include "G4SystemOfUnits.hh"
#include "G4UImanager.hh"
#include "Randomize.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
constexpr double kMuMassMeV = 105.6583715;

double KineticEnergyMeV(double p_GeV) {
  const double p = p_GeV * 1000.0;
  const double E = std::sqrt(p * p + kMuMassMeV * kMuMassMeV);
  return E - kMuMassMeV;
}

std::string ResolveMaterial(const std::string& in) {
  if (in == "Cu" || in == "G4_Cu") return "G4_Cu";
  if (in == "Pb" || in == "G4_Pb") return "G4_Pb";
  std::cerr << "Unknown material '" << in << "'; expected Cu or Pb.\n";
  std::exit(1);
}

void PrintMuonProcesses() {
  auto* pm = G4MuonMinus::MuonMinus()->GetProcessManager();
  if (!pm) {
    std::cout << "[mstSim] mu- process manager unavailable\n";
    return;
  }
  std::cout << "[mstSim] mu- processes:";
  G4ProcessVector* processes = pm->GetProcessList();
  for (G4int i = 0; i < processes->size(); ++i) {
    std::cout << " " << (*processes)[i]->GetProcessName();
  }
  std::cout << "\n";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0]
              << " <urban|wentzel|wvi_ss> <Cu|Pb> <thickness_cm> <p_GeV> "
                 "<nEvents> <seed> <outFile>\n";
    return 1;
  }

  const std::string model = argv[1];
  const std::string material = ResolveMaterial(argv[2]);
  const double thicknessCm = std::atof(argv[3]);
  const double pGeV = std::atof(argv[4]);
  const long nEvents = std::atol(argv[5]);
  const long seed = std::atol(argv[6]);
  const std::string outFile = argv[7];

  if (thicknessCm <= 0.0 || pGeV <= 0.0 || nEvents <= 0 || seed <= 0) {
    std::cerr << "thickness, momentum, nEvents and seed must be positive.\n";
    return 1;
  }

  G4Random::setTheSeed(seed);

  const double keMeV = KineticEnergyMeV(pGeV);
  std::cout << "[mstSim] model=" << model << " material=" << material
            << " t=" << thicknessCm << " cm  p=" << pGeV
            << " GeV/c  (KE=" << keMeV << " MeV)  N=" << nEvents
            << " seed=" << seed << "\n";

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
  PrintMuonProcesses();

  G4UImanager* ui = G4UImanager::GetUIpointer();
  ui->ApplyCommand("/gun/particle mu-");
  ui->ApplyCommand("/gun/energy " + std::to_string(keMeV) + " MeV");

  runManager->BeamOn(nEvents);

  delete runManager;
  return 0;
}
