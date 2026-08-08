// PrimaryGeneratorAction.hh
//
// mu- shot along +z from a fixed upstream position, on-axis (position
// is irrelevant to the angular measurement in a transversely uniform
// slab). Kinetic energy is set via the standard, automatically-available
// /gun/energy command -- see README.md / run_sweep.sh for the
// momentum -> kinetic-energy table (E_kin = sqrt(p^2+m_mu^2) - m_mu).

#ifndef PRIMARY_GENERATOR_ACTION_HH
#define PRIMARY_GENERATOR_ACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4ParticleGun;
class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
  PrimaryGeneratorAction();
  ~PrimaryGeneratorAction() override;

  void GeneratePrimaries(G4Event* event) override;

private:
  G4ParticleGun* fGun;
  static constexpr G4double kGunZ = -190.0;   // cm; well upstream of any slab
};

#endif
