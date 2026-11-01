# Proteus HIL configuration and test plan

**STATUS: PENDING EXTERNAL EXECUTION.** Nothing in this file is a measured
result. It is the configuration and the test protocol required to obtain one.
Proteus 8.13+ with VSM for Arduino and the COMPIM model is required.

## 1. Bill of materials (Blueprint v1.0, Section 3.1)

| Component | Model | Qty | Purpose |
|---|---|---|---|
| Microcontroller | Arduino Uno (ATmega328P) | 1 | Runs the generated ANN C code |
| Current sensor | ACS712 | 3 | Emulates Ia, Ib, Ic |
| Voltage sensor | ZMPT101B | 3 | Emulates Va, Vb, Vc |
| Relay driver | ULN2003 or 2N2222 | 1 | Drives the trip relay |
| Relay | 5 V SPDT | 1 | Circuit-breaker proxy |
| Serial interface | COMPIM (RS232) | 1 | MATLAB co-simulation |
| Virtual terminal | — | 1 | Classification output |
| Indicator LEDs | — | 4 | G, C, B, A flags |

## 2. Pin mapping

| Signal | Pin | Signal | Pin |
|---|---|---|---|
| Ia / Ib / Ic | A0 / A1 / A2 | G / C / B / A LED | D8 / D9 / D10 / D11 |
| Va / Vb / Vc | A3 / A4 / A5 | Trip output | D12 |
| COMPIM RX/TX | D0 / D1 | | |

## 3. Unresolved scaling problem — must be settled first

The dataset currents reach approximately **±900 A**, while an ACS712-30A
saturates at ±30 A. The sensor chain therefore cannot reproduce the training
distribution as-is. One of the following must be adopted **and stated in the
manuscript** before any agreement figure is claimed:

1. Insert an explicit CT ratio (e.g. 1000:1) ahead of the ACS712 and document it.
2. Replace the ACS712 model with a scaled ideal transducer.
3. Drive the ANN over the serial link instead of through the analog front end,
   and report the result as a software-in-the-loop rather than hardware-in-the-
   loop test.

Option 3 is the honest fallback if the analog chain cannot be matched; it must
not be described as hardware-in-the-loop.

## 4. Co-simulation (Blueprint 3.3)

- VSPE creates the virtual pair COM1 ↔ COM2.
- COMPIM on COM1, 9600 baud, 8N1. MATLAB `serialport("COM2",9600)` on the host.
- VSPE adds roughly 1–5 ms of latency; this must be measured separately and
  **subtracted** from the end-to-end figure before any trip latency is quoted.

## 5. Test protocol

| ID | Test | Procedure | Acceptance |
|---|---|---|---|
| P1 | Build | Compile the sketch with the Coder output | Fits flash and 2 KB SRAM |
| P2 | Static agreement | Send the 1,573 rows of `X_test.csv` over COMPIM; compare with `y_pred_python.csv` | Agreement > 99 % |
| P3 | Loop time | Toggle a spare pin around `ann_predict()`; measure on the Proteus scope | Report in µs |
| P4 | Serial latency | Loop-back test with no inference | Report mean ± sd |
| P5 | Trip latency | Inject a fault transient; measure fault-onset to relay-close, then subtract P4 | Target < 20 ms |
| P6 | Quantisation | Repeat P2 with int8 weights | Agreement loss < 1 pp |

## 6. Reporting rules

- Report P3–P5 as mean ± standard deviation over at least 100 repetitions.
- State the Proteus timestep and whether real-time mode was enabled; Proteus
  wall-clock time is not the emulated time.
- If P2 fails, do not proceed to P5: a latency figure from a mis-classifying
  relay is meaningless.

## 7. Items the user must supply

- `[USER INPUT REQUIRED]` CT ratio and voltage-divider values actually used.
- `[USER INPUT REQUIRED]` Proteus version, timestep, and the `.pdsprj` file.
- `[USER INPUT REQUIRED]` avr-size output for flash and SRAM occupancy.
- `[USER INPUT REQUIRED]` Oscilloscope captures evidencing P5.
