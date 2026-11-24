import numpy as np
from grb_trigger.simulations import detector_counts, generate_grb_pulse, generate_false_alert, generate_saa_crossing
from grb_trigger.features import features_extraction_loop

def generate_labels(n_bins, events, background_bins_number=100, foreground_bins_number=10):
    """
    Produces the 1D target array matching the exact rows returned by features.feature_extraction_loop().

    Parameters
    ----------
    n_bins : int
        signal length in bin number.
    events : list of dict
        Each dict has the keys ``"signal"`` (``(low, high)`` arrays),
        ``"start_bin"`` (int, > 0), ``"type"`` (str, e.g. ``"grb"``) and
        optionally ``"is_grb"`` (0 or 1).
    background_bins_number : int, default 100
        Number of bins in the background window.
    foreground_bins_number : int, default 10
        Number of bins in the foreground window.

    Returns
    -------
    y : numpy.ndarray
        Target column: true if more than half of the window values are is_grb=1

    Notes
    -----
    Returned array has shape (n,) with
    `n = n_bins - background_bins_number - foreground_bins_number + 1`.

    """

    # 1. Create a blank 1D mask of 0s (length = total bins in the orbit)
    grb_mask = np.zeros(n_bins, dtype=int)

    # 2. Mark the exact bins where a real GRB ("is_grb" == 1) is active
    for event in events:
        if event["is_grb"] == 1:
            start = event["start_bin"]
            duration = len(event["signal"][0])
            grb_mask[start : start + duration] = 1

    # 3. Slide across the exact same valid indices as feature_extraction_loop
    y = []
    for i in range(background_bins_number, n_bins - foreground_bins_number + 1):
        # grb foreground slide of our mask
        fg_mask = grb_mask[i : i + foreground_bins_number]

        # check if at least half of the foreground bins contain a GRB (1)
        if fg_mask.mean()>=0.5:
            y.append(1)
        else:
            y.append(0)

    return np.array(y)

def generate_dataset(n_orbits=50, n_bins=1000, probs=[0.4, 0.3, 0.2], background_bins_number=100, foreground_bins_number=10):
    """
    Simulate n_orbits separate CubeSat orbits in a loop, where each orbit generates its own X_orbit and y_orbit with length `(n_bins - background_bins_number - foreground_bins_number + 1)`.

    Parameters
    ----------
    n_orbits : int, default 50
        Number of separate orbits to simulate.
    n_bins : int, default 1000
        Number of bins in an orbit, each bin is a photon count recording.
    probs : list of floats, default [0.4, 0.3, 0.2]
        Probabilities for generating a GRB, False alarm or SAA signal.
    background_bins_number : int, default 100
        Number of bins in the background window.
    foreground_bins_number : int, default 10
        Number of bins in the foreground window.

    Returns
    -------
    X_orbit : numpy.ndarray (n, 7)
        The feature matrix compiled with the results of the
        simulation module
    y_orbit : numpy.ndarray (n, )
        The target array with the information returned by generate_labels

    Raises
    ------
    ValueError
        If any value in `probs` is larger than 1.0.
    ValueError
        If n_bins-`background_bins_number`-`foreground_bins_number` is
        less than 1.
    ValueError
        If `n_orbits` is smaller than 1.

    Notes
    -----
    Each orbit has a signal with length n_bins.

    The length each orbit signal after applying the background
    and foreground cuts (length of the output) `n = n_bins -
    background_bins_number - foreground_bins_number + 1`.

    The probabilities `probs` are not physically correction, in reality
    we should detect around 1 GRB signal per 1000 orbits. However, for
    the initial purposes of trainign the ML model, we want to boost our
    dataset.

    """
    # checks
    for p in probs:
        if p>1.0:
            raise ValueError("Probabilities in `probs` cannot be larger than 1.0!")

    valid_bins = n_bins - background_bins_number - foreground_bins_number
    if valid_bins<=0:
        raise ValueError(f"There will be no valid bins after applying background and foreground margins (currently, {valid_bins} valid bins.")

    if n_orbits <=0:
        raise ValueError(f"n_orbits ({n_orbits}) must be positive integer.")

    X_list = []
    y_list = []

    for _ in range(n_orbits):
        # 1. Generate background
        bkg = generate_background(n_bins)

        # 2. Generate random events list
        events = []

        # generate GRBs
        if np.random.random() < probs[0]:
            # Randomize start_bin ensuring it fits within n_bins
            start = np.random.randint(background_bins_number, n_bins-200)

            # applies gaussian (normal) distribution to compute random
            # values for pulse parameters
            # in the future maybe we want to add more options for
            # paramter generation
            duration = np.random.normal(loc=120, scale=50)
            amplitude = np.random.uniform(50, 300)
            tau_rise = np.random.normal(loc=3, scale=1)
            tau_decay = np.random.normal(loc=15, scale=5)

            pulse = generate_grb_pulse(duration, amplitude, tau_rise, tau_decay)
            events.append({
                "signal": pulse,
                "start_bin": start,
                "type": "grb",
                "is_grb":1,
            })


        # generate CRs
        if np.random.random() < probs[1]:
            start = np.random.randint(background_bins_number, n_bins-200)

            # for duration use choice instead of normal because just 2 values
            duration = np.random.choice([1,2])
            amplitude = np.random.uniform(200, 800)

            pulse = generate_false_alarm(duration, amplitude)
            events.append({
                "signal": pulse,
                "start_bin": start,
                "type": "false alarm",
                "is_grb": 0,
            })


        # generate SAA
        if np.random.random() < probs[2]:
             start = np.random.randint(background_bins_number, n_bins-200)

             duration = np.random.normal(loc=100, scale=20)
             amplitude = np.random.uniform(30, 100)

             pulse = generate_saa_crossing(duration, amplitude)
             events.append({
                 "signal": pulse,
                 "start_bin": start,
                 "type": "SAA",
                 "is_grb": 0,
             })


        # 3. Combine and extract
        counts = detect_counts(bkg, events)
        X_orbit = feature_extraction_loop(counts, background_bins_number, foreground_bins_number)
        y_orbit = generate_labels(n_bins, events, background_bins_number, foreground_bins_number)

        X_list.append(X_orbit)
        y_list.append(y_orbit)

        # Stack per-orbit feature matrices into a single (total_windows, 7) array
        # and concatenate 1D label arrays into a matching (total_windows,) vector
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

    return X, y
