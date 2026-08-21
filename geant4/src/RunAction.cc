#include "RunAction.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"

#include <iomanip>
#include <iostream>

RunAction::RunAction(const G4String& outFileName) : fOutFileName(outFileName) {}

RunAction::~RunAction() = default;

void RunAction::BeginOfRunAction(const G4Run* /*run*/) {
  fNWritten = 0;
  fOut.open(fOutFileName, std::ios::out | std::ios::trunc);
  if (!fOut.is_open()) {
    G4Exception("RunAction::BeginOfRunAction", "FileOpenFail", FatalException,
               ("Could not open output file: " + fOutFileName).c_str());
  }
  fOut << std::setprecision(10);
}

void RunAction::EndOfRunAction(const G4Run* run) {
  fOut.close();
  G4long nEvents = run->GetNumberOfEvent();
  std::cout << "[RunAction] wrote " << fNWritten << " / " << nEvents
           << " primary exit angles to " << fOutFileName;
  if (fNWritten < nEvents) {
    std::cout << "  (** " << (nEvents - fNWritten)
              << " primaries did not reach the scoring plane -- likely "
                 "stopped or backscattered; check at low momentum/thick "
                 "slabs. **)";
  }
  std::cout << std::endl;
}

void RunAction::WriteTheta(G4double thetaRad) {
  fOut << thetaRad << "\n";
  ++fNWritten;
}
