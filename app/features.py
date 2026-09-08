"""Motion features computed over the rolling window of normalised poses.

Every feature is scale-free: because the poses arrive in torso units, a metre
of arm swing reads the same whether the crew member is beside the camera or
across the module. Speeds are per second, so they do not change if the frame
rate drops.
"""

import numpy as np

from . import config

# MediaPipe landmark indices used here.
NOSE = 0
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
# On a real face the mouth sits within this far of the nose, in torso units.
MOUTH_NEAR_NOSE = 0.35
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24


class MotionFeatures:
    """Plain container so the dashboard can show the evidence behind a label."""

    __slots__ = ("wrist_speed", "peak_wrist_speed", "body_speed", "rhythm",
                 "periodicity", "cycle_hz", "wrist_range",
                 "head_zone_fraction", "workstation_fraction", "elevated_fraction",
                 "extended_fraction", "wrist_spread", "vertical_swing",
                 "confidence", "duration")

    def as_dict(self) -> dict:
        return {name: round(float(getattr(self, name)), 4) for name in self.__slots__}


# Human repetitive motion of interest sits in this band: slower is a single
# deliberate movement, faster is tracking noise.
MIN_CYCLE_HZ = 0.8
MAX_CYCLE_HZ = 3.5
SPEED_STRIDE = 3                 # frames between the samples speed is measured over
MIN_RHYTHM_AMPLITUDE = 0.12      # torso units; below this there is nothing to
                                 # be periodic about, so do not trust a peak


def _periodicity(signal: np.ndarray, duration: float, amplitude: float):
    """How strongly `signal` repeats, and at what rate.

    Returns (strength 0-1, cycles per second). Strength is the height of the
    best autocorrelation peak inside the human-motion band, so a steady wave
    scores high while drift, a single lift, or jitter score near zero.
    """
    n = signal.size
    if n < 8 or duration <= 0 or amplitude < MIN_RHYTHM_AMPLITUDE:
        return 0.0, 0.0

    # Light smoothing first, so landmark jitter cannot masquerade as a cycle.
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(signal, kernel, mode="valid")
    n = smoothed.size
    if n < 8:
        return 0.0, 0.0

    # Remove any straight-line trend. Without this a hand travelling steadily
    # in one direction - a single slow lift - produces a high autocorrelation
    # and is mistaken for a repeating movement.
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, smoothed, 1)
    centred = smoothed - (slope * x + intercept)

    spread = float(np.sqrt(np.mean(centred ** 2)))
    if spread < 1e-6:
        return 0.0, 0.0
    centred = centred / spread

    # Divide by the number of samples that actually overlap at each lag. With
    # a flat 1/n the estimate sags at longer lags simply because less data
    # contributes, which buries slower movements like a steady arm swing.
    raw = np.correlate(centred, centred, mode="full")[n - 1:]
    overlap = np.arange(n, 0, -1, dtype=np.float64)
    correlation = raw / overlap

    dt = duration / max(1, n - 1)
    # Round outward so the searched lags stay inside the intended frequency
    # band rather than spilling past it through integer rounding.
    low = max(2, int(np.ceil(1.0 / (MAX_CYCLE_HZ * dt))))
    high = min(n - 2, int(np.floor(1.0 / (MIN_CYCLE_HZ * dt))))
    if high <= low:
        return 0.0, 0.0

    # A real oscillation swings away from itself before coming back, so the
    # correlation must dip below zero before the peak. A one-way movement
    # never does, which is what separates a rep from a single reach.
    peaks = []
    for lag in range(low, high + 1):
        value = correlation[lag]
        is_local_peak = (value >= correlation[lag - 1]
                         and value >= correlation[min(lag + 1, n - 1)])
        if not is_local_peak:
            continue
        if correlation[1:lag].min() > -0.05:
            continue                      # never swung away: not oscillating
        peaks.append((lag, float(value)))

    if not peaks:
        return 0.0, 0.0

    strength = max(value for _, value in peaks)
    if strength <= 0.0:
        return 0.0, 0.0        # every candidate anti-correlates: no rhythm

    # Harmonics of a fast movement also show up as peaks, so take the shortest
    # lag that is nearly as strong - that is the true cycle, not a multiple.
    best = min(lag for lag, value in peaks if value >= strength * 0.9)

    strength = float(max(0.0, min(1.0, strength)))
    lag_seconds = best * dt
    hz = (1.0 / lag_seconds) if lag_seconds > 0 else 0.0
    return strength, float(hz)


def _smooth(track: np.ndarray, span: int = 5) -> np.ndarray:
    """Moving average along time, so jitter does not read as movement.

    Differencing raw landmarks frame by frame measures noise as much as
    motion: every wobble adds to the total, and over a two second window the
    noise accumulates while real movement does not. Smoothing first is what
    makes a still person read as still.
    """
    n = track.shape[0]
    if n < span or span < 2:
        return track

    flat = track.reshape(n, -1).astype(np.float64)
    kernel = np.ones(span) / span
    out = np.empty_like(flat)
    for column in range(flat.shape[1]):
        out[:, column] = np.convolve(flat[:, column], kernel, mode="same")

    # A centred average has nothing to average at the ends, so it pulls the
    # first and last samples toward zero. Keep the originals there.
    half = span // 2
    out[:half] = flat[:half]
    out[-half:] = flat[-half:]
    return out.reshape(track.shape)


def extract(window) -> MotionFeatures:
    """Turn a PoseWindow into the motion features the classifier scores."""
    frames = window.frames
    f = MotionFeatures()

    points = np.stack([fr.points for fr in frames])            # (N, 33, 2)
    times = np.array([fr.timestamp for fr in frames])
    duration = float(times[-1] - times[0])
    f.duration = duration

    # Guard against a stalled window: without elapsed time, speeds are undefined.
    if duration < 1e-3:
        for name in MotionFeatures.__slots__:
            if name != "duration":
                setattr(f, name, 0.0)
        f.confidence = float(np.mean([fr.confidence for fr in frames]))
        return f

    deltas = np.diff(times)
    deltas[deltas <= 0] = 1e-3

    # ---------------------------------------------------------- arm motion --
    wrists = points[:, [L_WRIST, R_WRIST], :]                  # (N, 2, 2)
    wrists_smooth = _smooth(wrists)

    # Speed is measured across a stride of several frames rather than between
    # neighbours. Real movement covers ground steadily, so a longer baseline
    # captures the same speed, while random jitter partly cancels over it
    # instead of being counted at every step.
    stride = min(SPEED_STRIDE, max(1, len(frames) - 1))
    displacement = np.linalg.norm(wrists_smooth[stride:] - wrists_smooth[:-stride],
                                  axis=2)
    elapsed = times[stride:] - times[:-stride]
    elapsed[elapsed <= 0] = 1e-3
    wrist_speeds = displacement / elapsed[:, None]
    per_frame_speed = wrist_speeds.mean(axis=1)
    f.wrist_speed = float(per_frame_speed.mean())
    f.peak_wrist_speed = float(np.percentile(per_frame_speed, 90))

    # ------------------------------------------------- whole-body movement --
    # Normalised hips sit at the origin by construction, so translation across
    # the module has to be read from the raw image position of the hip centre.
    #
    # Measured as how far the body actually roamed across the window, not as
    # the summed length of its frame-to-frame path. A path length counts every
    # wobble, so a seated crew member accumulated enough of them to read as
    # walking; the span of the trajectory does not move unless they do.
    hips = _smooth(np.array([fr.hip_center for fr in frames]))
    hip_span = float(np.linalg.norm(hips.max(axis=0) - hips.min(axis=0)))
    f.body_speed = hip_span / duration

    # --------------------------------------------------- travel and rhythm ---
    # How far the hands actually roam. This is what separates a rep of an
    # exercise from adjusting a switch: both can be quick, but only one sweeps
    # a large arc. Measured peak-to-peak in torso units.
    travel = np.linalg.norm(wrists_smooth.max(axis=0) - wrists_smooth.min(axis=0),
                            axis=1)
    f.wrist_range = float(travel.max())

    vertical = wrists_smooth[:, :, 1].mean(axis=1)
    f.vertical_swing = float(vertical.max() - vertical.min())

    # Counting direction reversals looked like a rhythm detector but mostly
    # measured noise - motionless hands scored dozens of reversals from
    # landmark jitter alone, and the count changed with the frame rate.
    # Autocorrelation asks the right question instead: does this movement
    # repeat itself at a steady interval?
    #
    # Each wrist is tested on each axis separately. Averaging the two wrists
    # cancels out alternating movement - punches, running arms - and looking
    # only at height misses movement that is purely side to side.
    f.periodicity, f.cycle_hz = 0.0, 0.0
    for wrist in range(2):
        for axis in range(2):
            track = wrists_smooth[:, wrist, axis]
            amplitude = float(track.max() - track.min())
            strength, hz = _periodicity(track, duration, amplitude)
            if strength > f.periodicity:
                f.periodicity, f.cycle_hz = strength, hz

    f.rhythm = f.periodicity * 10.0        # kept for the dashboard readout

    # --------------------------------------------------------- hand zones ---
    # Distance is taken to the mouth, not the bridge of the nose: a hand
    # raised to eat sits by the mouth, and measuring to the nose made the
    # same gesture look further from the face than it is.
    mouth = (points[:, MOUTH_LEFT, :] + points[:, MOUTH_RIGHT, :]) / 2.0
    nose = points[:, NOSE, :]
    # Some frames have no usable mouth landmarks; fall back to the nose.
    # Trust the mouth only where it sits plausibly close to the nose. Testing
    # merely that it is non-zero is not enough: a missing landmark reads as
    # near the origin, which is the hip centre, and hands resting at the hips
    # were then scored as hands at the face.
    usable = np.linalg.norm(mouth - nose, axis=1) < MOUTH_NEAR_NOSE
    reference = np.where(usable[:, None], mouth, nose)

    to_head = np.linalg.norm(wrists - reference[:, None, :], axis=2)   # (N, 2)
    f.head_zone_fraction = float(np.mean(to_head.min(axis=1)
                                         < config.HEAD_ZONE_RADIUS))

    in_x = np.abs(wrists[:, :, 0]) < config.WORKSTATION_X
    in_y = ((wrists[:, :, 1] > config.WORKSTATION_Y_TOP)
            & (wrists[:, :, 1] < config.WORKSTATION_Y_BOTTOM))
    f.workstation_fraction = float(np.mean(np.any(in_x & in_y, axis=1)))

    # ------------------------------------------------- arm posture ----------
    shoulders = points[:, [L_SHOULDER, R_SHOULDER], :]
    # Screen "up" is negative Y, so a raised wrist sits above the shoulder line.
    raised = wrists[:, :, 1] < (shoulders[:, :, 1] - config.ELEVATED_MARGIN)
    f.elevated_fraction = float(np.mean(np.any(raised, axis=1)))

    reach = np.linalg.norm(wrists - shoulders, axis=2)
    f.extended_fraction = float(np.mean(np.any(reach > config.EXTENDED_REACH, axis=1)))

    # How far the hands roam: small for focused work, large for exercise.
    f.wrist_spread = float(np.linalg.norm(wrists_smooth.std(axis=0), axis=1).mean())

    f.confidence = float(np.mean([fr.confidence for fr in frames]))
    return f
