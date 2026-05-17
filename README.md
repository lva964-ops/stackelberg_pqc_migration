# Stackelberg Games for Active Deception in PQC Migration of Critical Infrastructure

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the numerical implementation of a **hierarchical Stackelberg game** model for managing the migration of **Critical Information Infrastructure (CII)** to **Post-Quantum Cryptography (PQC)** under the threat of *active deception* attacks.

Unlike passive "Harvest Now, Decrypt Later" (HNDL) strategies, an active adversary does not just collect encrypted data – they actively manipulate the migration process by injecting trojaned PQC algorithms, downgrading hybrid protocols, or mimicking successful migration while leaving classical backdoors open.

The model answers the following key questions:

- How should a defender (leader) allocate limited resources between *actual PQC migration* and *deployment of deceptive decoy nodes*?
- How does an attacker (follower) rationally respond to observed deception signals?
- What is the **Active Deception Equilibrium** – a state where the cost of maintaining camouflage exceeds the attacker’s expected benefit?

The mathematical framework is a **differential Stackelberg game with incomplete information**, solved numerically via the **Forward-Backward Sweep Method (FBSM)**.

---

## Key Features

- **Three realistic CII scenarios** (calibrated to Kazakhstan’s infrastructure data):
  - *A – High friction*: legacy SCADA/ICS with expensive PQC migration.
  - *B – High dynamics*: flexible cloud/financial sector with low migration cost.
  - *C – Strategic*: state secrets with maximum data value and moderate friction.
- **Optimal controls**:
  - `α(t)` – migration intensity (real nodes `V_C → V_P`).
  - `β(t)` – deception intensity (probability that a perceived vulnerable node is actually a decoy).
- **Attacker’s best response** `a(t)` – attack intensity on nodes believed to be vulnerable.
- **Non‑linear technological friction** `F(α) = k·exp(η·α)` capturing performance degradation during accelerated PQC rollout.
- **Separate visual outputs** for each scenario (controls + state variables).
- **Integral metrics**: total defender cost, total HNDL damage, and summary table.

---

## Mathematical Model (Brief)

State vector:  
`X(t) = [x_C(t), x_P(t), x_D(t)]` – fractions of classical, PQC‑migrated, and decoy nodes.

Defender’s control: `u(t) = {α(t), β(t)}`  
Attacker’s control: `a(t)`

Dynamics:
