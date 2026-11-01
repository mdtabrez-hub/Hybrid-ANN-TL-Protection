"""
network.py -- pandapower model of the calibrated 220 kV network, plus an
independent symmetrical-component fault solver.

Topology (Blueprint 4.1, Fig. 15):

    G1,G2 --| T1 |-- BUS A ===== line, 100 km ===== BUS B --| T2 |-- G3,G4
                       |<-- f -->X fault
                     R_A                              R_B

Each generating station is represented by an IEC 60909 network feeder whose
positive- and zero-sequence impedances are those recovered in calibration.json.
The machines and step-up transformers are not modelled individually because the
measurements determine only their aggregate; see calibration_report.md, A1-A6.

Two independent solvers are provided and cross-checked in run_study.py:
  * pandapower's IEC 60909 calc_sc  (3ph, 2ph, 1ph)
  * seq_fault(), a direct symmetrical-component solution that additionally
    handles LLG and LLLG and arbitrary fault resistance.
"""
import json, os, cmath
import numpy as np
import pandapower as pp

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, "calibration.json")))

U_N   = CAL["base"]["U_N_kV"] * 1e3
F_HZ  = CAL["base"]["f_Hz"]
LKM   = CAL["base"]["line_km"]
C_MAX = 1.10                      # IEC 60909 voltage factor, max, for Un > 1 kV

Z1_SRC = complex(CAL["source"]["R1"], CAL["source"]["X1"])
Z0_SRC = complex(CAL["source"]["R0"], CAL["source"]["X0"])
WEAK   = CAL["source"]["weak_scale"]

z1_km = complex(CAL["line"]["r1_ohm_km"], CAL["line"]["x1_ohm_km"])
z0_km = complex(CAL["line"]["r0_ohm_km"], CAL["line"]["x0_ohm_km"])
C1_NF = CAL["line"]["c1_nf_km"]
C0_NF = C1_NF * 0.6               # published ratio for a single-circuit line

Z1_LINE = z1_km * LKM
Z0_LINE = z0_km * LKM

A = cmath.exp(2j * cmath.pi / 3)   # 120 deg operator


# ------------------------------------------------------------------ pandapower
def build_net(frac=0.5, strong=True):
    """220 kV network with the line split at `frac` so a fault bus exists there."""
    net = pp.create_empty_network(f_hz=F_HZ, sn_mva=100.0)
    bA = pp.create_bus(net, vn_kv=U_N / 1e3, name="BUS A")
    bF = pp.create_bus(net, vn_kv=U_N / 1e3, name="FAULT")
    bB = pp.create_bus(net, vn_kv=U_N / 1e3, name="BUS B")

    k = 1.0 if strong else WEAK
    z1 = Z1_SRC * k                       # generation outage raises Z1 ...
    z0 = Z0_SRC                           # ... but not the transformer-set Z0
    # IEC 60909 defines a feeder by Z_Q = c U_n^2 / S_k"; invert that so the
    # impedance pandapower builds is exactly the calibrated Z1_SRC.
    s_sc = C_MAX * U_N ** 2 / abs(z1) / 1e6

    for b, nm in ((bA, "Station A"), (bB, "Station B")):
        pp.create_ext_grid(net, bus=b, vm_pu=1.0, va_degree=0.0, name=nm,
                           s_sc_max_mva=s_sc, s_sc_min_mva=s_sc,
                           rx_max=z1.real / z1.imag, rx_min=z1.real / z1.imag,
                           x0x_max=abs(z0.imag / z1.imag),
                           r0x0_max=abs(z0.real / z0.imag))

    for (f, t, km) in ((bA, bF, LKM * frac), (bF, bB, LKM * (1 - frac))):
        if km <= 0:
            km = 1e-3
        pp.create_line_from_parameters(
            net, from_bus=f, to_bus=t, length_km=km,
            r_ohm_per_km=z1_km.real, x_ohm_per_km=z1_km.imag, c_nf_per_km=C1_NF,
            r0_ohm_per_km=z0_km.real, x0_ohm_per_km=z0_km.imag, c0_nf_per_km=C0_NF,
            max_i_ka=1.0, endtemp_degree=80.0)

    pp.create_load(net, bus=bB, p_mw=CAL["load"]["P_MW"], q_mvar=CAL["load"]["Q_MVAr"])
    net.bus_geodata = None
    return net, dict(A=bA, F=bF, B=bB)


# -------------------------------------------- symmetrical-component fault solver
def _paths(frac, strong):
    """Sequence impedance of the path from each station to the fault point."""
    k = 1.0 if strong else WEAK
    return dict(
        z1a=Z1_SRC * k + Z1_LINE * frac,
        z1b=Z1_SRC * k + Z1_LINE * (1 - frac),
        z0a=Z0_SRC + Z0_LINE * frac,
        z0b=Z0_SRC + Z0_LINE * (1 - frac),
    )


def _par(x, y):
    return x * y / (x + y)


def seq_fault(kind, frac, strong, rf=0.0, rg=0.0):
    """
    Solve a shunt fault at `frac` of the line.

    kind : 'LLL' | 'LLLG' | 'LG' | 'LL' | 'LLG'
    rf   : fault resistance in each faulted phase, ohm
    rg   : additional resistance in the ground path, ohm

    Returns a dict with the sequence currents at the fault, the phase currents
    and voltages seen by relay A, and the total fault-point current.
    Phase convention matches the dataset: LG on phase A, LL on phases B-C,
    LLG on phases A-B (see Section 4.3 of the manuscript).
    """
    p = _paths(frac, strong)
    Z1 = _par(p["z1a"], p["z1b"])
    Z2 = Z1                                  # static network
    Z0 = _par(p["z0a"], p["z0b"])
    E = C_MAX * U_N / np.sqrt(3)             # IEC 60909 equivalent source voltage
    Zf, Zg = complex(rf, 0), complex(rg, 0)

    if kind in ("LLL", "LLLG"):
        # a balanced three-phase fault draws no zero-sequence current whether or
        # not the fault also contacts earth: LLLG is electrically LLL.
        I1 = E / (Z1 + Zf)
        I2 = I0 = 0j
        rot = 0                              # faulted phases: a, b, c
    elif kind == "LG":
        I1 = I2 = I0 = E / (Z1 + Z2 + Z0 + 3 * Zf + 3 * Zg)
        rot = 0                              # faulted phase: a
    elif kind == "LL":
        I1 = E / (Z1 + Z2 + 2 * Zf)
        I2, I0 = -I1, 0j
        rot = 0                              # faulted phases: b, c
    elif kind == "LLG":
        zz2, zz0 = Z2 + Zf, Z0 + Zf + 3 * Zg
        I1 = E / (Z1 + Zf + _par(zz2, zz0))
        I2 = -I1 * zz0 / (zz2 + zz0)
        I0 = -I1 * zz2 / (zz2 + zz0)
        rot = 1                              # faulted phases b,c -> rotate to a,b
    else:
        raise ValueError(kind)

    # current division between the two ends, per sequence network
    da = p["z1b"] / (p["z1a"] + p["z1b"])
    d0 = p["z0b"] / (p["z0a"] + p["z0b"])
    I1a, I2a, I0a = I1 * da, I2 * da, I0 * d0

    def phases(i0, i1, i2, rot_by):
        v = np.array([i0 + i1 + i2,
                      i0 + A ** 2 * i1 + A * i2,
                      i0 + A * i1 + A ** 2 * i2])
        return np.roll(v, rot_by)

    Iph_f = phases(I0, I1, I2, rot)                 # at the fault point
    Iph_r = phases(I0a, I1a, I2a, rot)              # in relay A

    # voltages at bus A: E behind the source impedance, carrying relay A current
    k = 1.0 if strong else WEAK
    V1a = E - I1a * (Z1_SRC * k)
    V2a = -I2a * (Z1_SRC * k)
    V0a = -I0a * Z0_SRC
    Vph_r = phases(V0a, V1a, V2a, rot)

    return dict(kind=kind, frac=frac, strong=strong, rf=rf, rg=rg,
                I1=I1, I2=I2, I0=I0, I1a=I1a, I2a=I2a, I0a=I0a,
                If_phase=Iph_f, Ir_phase=Iph_r, Vr_phase=Vph_r,
                If_max=float(np.abs(Iph_f).max()),
                Ir_max=float(np.abs(Iph_r).max()),
                Ir_res=float(abs(3 * I0a)),
                Z1th=Z1, Z0th=Z0, rot=rot)


def apparent_impedance(res):
    """
    Impedance seen by the distance relay at bus A: the minimum over the three
    phase-to-phase loops and the three phase-to-earth loops, the earth loops
    being compensated with the residual factor k0 = (Z0L - Z1L) / (3 Z1L).
    """
    I, V = res["Ir_phase"], res["Vr_phase"]
    k0 = (Z0_LINE - Z1_LINE) / (3 * Z1_LINE)
    Ires = I.sum()
    loops = {}
    for i, j, nm in ((0, 1, "ab"), (1, 2, "bc"), (2, 0, "ca")):
        dI = I[i] - I[j]
        if abs(dI) > 1e-6:
            loops[nm] = (V[i] - V[j]) / dI
    for i, nm in ((0, "ag"), (1, "bg"), (2, "cg")):
        den = I[i] + k0 * Ires
        if abs(den) > 1e-6:
            loops[nm] = V[i] / den
    # only loops actually carrying fault current are meaningful
    valid = {k: z for k, z in loops.items() if z.real > -5 and 0 < z.imag < 1e4}
    if not valid:
        return None, None
    nm = min(valid, key=lambda k: abs(valid[k]))
    return nm, valid[nm]
