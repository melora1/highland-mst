#include "PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction() {
  fGun = new G4ParticleGun(1);  // G4ParticleGun registers its own /gun/*
                                // UI commands automatically (particle,
                                // energy, direction, position).
  G4ParticleDefinition* muMinus =
      G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  fGun->SetParticleDefinition(muMinus);
  fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
  fGun->SetParticlePosition(G4ThreeVector(0, 0, kGunZ * cm));
  fGun->SetParticleEnergy(899.908 * MeV);  // default: 1 GeV/c mu-; override
                                            // per run with /gun/energy
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fGun; }

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  fGun->GeneratePrimaryVertex(event);
}
