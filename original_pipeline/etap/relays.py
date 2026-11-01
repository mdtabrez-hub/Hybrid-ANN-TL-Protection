"""
relays.py -- protection settings and operating-time models.

Elements, per the Blueprint's protection scheme:
  21    distance, quadrilateral, three zones, at both line ends
  50/51 phase overcurrent, IEC Standard Inverse, line and transformer backup
  51G   earth-fault overcurrent on the residual connection

Operating times follow IEC 60255-151 for the inverse-time elements and the
zone timers plus a fixed measuring time for the distance elements.
"""
import numpy as np

# ------------------------------------------------------------- IEC 60255-151
CURVES = {                      # name: (k, alpha)
    "SI": (0.14, 0.02),         # standard inverse
    "VI": (13.5, 1.00),         # very inverse
    "EI": (80.0, 2.00),         # extremely inverse
}
T_RELAY   = 0.020               # s, measuring time of a numerical relay
T_SECURITY = 0.300              # s, grading margin between primary and backup


def iec_time(I, Is, tms, curve="SI"):
    """Operating time of an inverse-time element; inf if it does not pick up."""
    k, a = CURVES[curve]
    m = I / Is
    if m <= 1.05:               # IEC: no operation below 1.05 x setting
        return np.inf
    return tms * k / (m ** a - 1.0) + T_RELAY


def iec_tms_for(I, Is, t_target, curve="SI"):
    """TMS that makes the element operate in t_target seconds at current I."""
    k, a = CURVES[curve]
    m = I / Is
    return (t_target - T_RELAY) * (m ** a - 1.0) / k


class DistanceRelay:
    """Three-zone quadrilateral distance element."""

    def __init__(self, zl_mag, zl_ang_deg, reaches=(0.80, 1.20, 2.00),
                 times=(0.00, 0.40, 0.80), r_reach=(20.0, 30.0, 50.0)):
        self.zl = zl_mag
        self.ang = np.deg2rad(zl_ang_deg)
        self.x_reach = [r * zl_mag * np.sin(self.ang) for r in reaches]
        self.r_reach = list(r_reach)
        self.t_zone = list(times)
        self.reaches = list(reaches)

    def zone(self, z_app):
        """Lowest zone whose quadrilateral contains z_app, else None."""
        if z_app is None:
            return None
        R, X = z_app.real, z_app.imag
        for i in range(3):
            if -2.0 <= X <= self.x_reach[i] and abs(R) <= self.r_reach[i]:
                return i + 1
        return None

    def time(self, z_app):
        z = self.zone(z_app)
        return (np.inf, None) if z is None else (self.t_zone[z - 1] + T_RELAY, z)


class OvercurrentRelay:
    def __init__(self, name, pickup_a, tms, curve="SI", inst_a=None,
                 inst_time=T_RELAY):
        self.name, self.Is, self.tms, self.curve = name, pickup_a, tms, curve
        self.inst_a, self.inst_time = inst_a, inst_time

    def time(self, I):
        if self.inst_a is not None and I >= self.inst_a:
            return self.inst_time, "50"
        t = iec_time(I, self.Is, self.tms, self.curve)
        return (t, "51") if np.isfinite(t) else (np.inf, None)
