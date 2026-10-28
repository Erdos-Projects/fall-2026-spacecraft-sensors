""" 
Shared anomaly detection procedure

This module contains the reusable functions for the anomaly detection procedure examined in the Rios_Telemon_P1_Reproduction notebook which uses the authors' saved P-1 LSTM predictions.

The goal is to provide one shared implementation that someone can use after cloning this repostiory, which channel specific data and settings supplied as inputs.

The main detection functions will:
- Select thresholds within windows of smoothed prediction errors
- Identify and buffer candidate anomalies
- Prune candidate intervals using the peak separation rule

The complete procedure also includes calculating and smoothing prediction errors, advancing through overlapping windows, analyzing both the original and inverted errors, and merging retained detections

The reproduction notebook retains the explanations and walkthrough. Other notebooks should import these functions to apply the same detection procedure to saved or newly generated predictions

Model training is a separate step and is not a part of this module.

"""

import numpy as np
import pandas as pd

# this first function takes a window of smoothed errors and selects its threshold

def select_threshold(window_errors, error_buffer = 100, inverted = False):
    """Return the selected threshold and z multiplier for one error window

    Parameters
    ----------
    window_errors : array-like
        One window of smoothed absolute prediction errors.
        
    error_buffer : int
        Buffer setting. Threshold scoring includes error_buffer - 1
        neighboring positions on each side of each flagged position.
        
    inverted : bool
        If True, analyze errors reflected around the window mean.
    """
    window_errors = np.asarray(window_errors, dtype = float)
    
    mean_error = np.mean(window_errors)
    
    std_error = np.std(window_errors)
    
    errors = (
        2 * mean_error - window_errors
        if inverted
        else window_errors
    )
    
    # Default if no candidate threshold qualifies.
    selected_z = 12.0
    selected_threshold = mean_error + selected_z * std_error
    best_score = -10_000_000

    for z in np.arange(2.5, 12.0, 0.5):
        epsilon = mean_error + z * std_error

        remaining = errors[errors < epsilon]
        flagged = np.flatnonzero(errors >= epsilon)

        if len(flagged) == 0:
            continue

        # Threshold scoring uses error_buffer - 1 neighbors per side.
        offsets = np.arange(-(error_buffer - 1), error_buffer)
        buffered = (flagged[:, None] + offsets).ravel()

        buffered = np.unique(
            buffered[(buffered >= 0) & (buffered < len(errors))]
        )

        groups = np.split(
            buffered,
            np.flatnonzero(np.diff(buffered) > 1) + 1
        )

        n_sequences = sum(len(group) > 1 for group in groups)
        n_flagged = len(buffered)

        mean_decrease = (
            mean_error - np.mean(remaining)
        ) / mean_error

        std_decrease = (
            std_error - np.std(remaining)
        ) / std_error

        score = (
            mean_decrease + std_decrease
        ) / (n_sequences**2 + n_flagged)

        eligible = (
            n_sequences <= 5
            and n_flagged < 0.5 * len(errors)
        )

        # Preserve the notebook's handling of tied scores.
        if eligible and score >= best_score:
            best_score = score
            selected_z = z
            selected_threshold = epsilon

    return selected_threshold, selected_z


# This function applies the threshold and error size checks, buffers flagged positions, and groups them into candidate anomaly intervals for later pruning

def find_window_candidates(
    window_errors,
    window_targets,
    threshold,
    window_number,
    inverted=False,
    error_buffer=100,
    history_length=250,
    batch_size=70,
    initial_ignore=None):
    
    """Return buffered candidate positions and candidate anomaly intervals.

    Parameters
    ----------
    window_errors : array-like, shape (n,)
        One window of smoothed absolute prediction errors
        
    window_targets : array-like
        Actual telemetry targets associated with the predictions in this
        window. May contain multiple future targets per prediction.
        Their standard deviation and percentile range set the error-size
        checks
        
    threshold : float
        Threshold selected by select_threshold for the corresponding
        original or inverted error branch
        
    window_number : int
        Zero-based window number. The first window applies the initial
        ignored region; later windows retain candidates only from their
        newest batch
        
    inverted : bool, default False
        If True, identify candidates using errors reflected around the
        window mean. The size check still uses the original errors
        
    error_buffer : int, default 100
        Number of neighboring positions added on each side of every
        flagged position, clipped to the window boundaries
        
    history_length : int, default 250
        Number of past readings used to make a prediction. Used here
        only to calculate the default initial ignored region
        
    batch_size : int, default 70
        Number of error positions in the newest batch of a later window
        
    initial_ignore : int or None, default None
        Number of initial error positions to ignore in the first window.
        None uses twice history_length, matching our P-1 reproduction.
        Set to 0 to disable this exclusion

    Returns
    -------
    buffered : numpy.ndarray
        Sorted, unique candidate positions after buffering and window
        exclusions. Positions are relative to this error window
        
    intervals : list of tuple
        Inclusive (start, end) pairs grouping consecutive candidate
        positions. Only groups containing at least two positions are
        included. These intervals have not yet been pruned
    """
    window_errors = np.asarray(window_errors, dtype=float)
    window_targets = np.asarray(window_targets, dtype=float)

    mean_error = np.mean(window_errors)
    std_error = np.std(window_errors)

    errors = (
        2 * mean_error - window_errors
        if inverted
        else window_errors
    )

    # Measure the spread of the actual telemetry targets.
    telemetry_std = np.std(window_targets)
    low, high = np.percentile(window_targets, [5, 95])
    telemetry_range = high - low

    # Both branches use the ORIGINAL errors for this size check.
    passes_size_check = (
        (
            std_error > 0.05 * telemetry_std
            or np.max(window_errors) > 0.05 * telemetry_range
        )
        and np.max(window_errors) > 0.05
    )

    if not passes_size_check:
        return np.array([], dtype=int), []

    candidates = np.flatnonzero(
        (errors >= threshold)
        & (errors > 0.05 * telemetry_range)
    )

    if len(candidates) == 0:
        return np.array([], dtype=int), []

    # Detection uses error_buffer neighbors on each side.
    offsets = np.arange(-error_buffer, error_buffer + 1)
    buffered = (candidates[:, None] + offsets).ravel()

    buffered = np.unique(
        buffered[(buffered >= 0) & (buffered < len(errors))]
    )

    if window_number == 0:
        ignore = (
            2 * history_length
            if initial_ignore is None
            else initial_ignore
        )
        buffered = buffered[buffered >= ignore]
    else:
        # Later windows contribute candidates from their newest batch.
        buffered = buffered[buffered >= len(errors) - batch_size]

    groups = np.split(
        buffered,
        np.flatnonzero(np.diff(buffered) > 1) + 1
    )

    intervals = [
        (int(group[0]), int(group[-1]))
        for group in groups
        if len(group) > 1
    ]

    return buffered, intervals

# This function removes candidate intervals whose peak errors do not stand
# sufficiently apart from other candidate peaks and the largest outside error

def prune_window_candidates(
    window_errors,
    candidate_positions,
    candidate_intervals,
    window_start=0,
    previous_anomaly_positions=None,
    inverted=False,
    p=0.13):
    """Prune candidate intervals using the peak-separation rule.

    Parameters
    ----------
    window_errors : array-like, shape (n,)
        One window of smoothed absolute prediction errors
        
    candidate_positions : array-like of int
        Buffered candidate positions returned by find_window_candidates.
        Positions are relative to this error window
        
    candidate_intervals : list of tuple
        Inclusive (start, end) candidate intervals, with endpoints
        relative to this error window
        
    window_start : int, default 0
        Position where this window begins in the full error array.
        This is not an original test-data index
        
    previous_anomaly_positions : array-like of int or None
        Previously retained anomaly positions in the full error array.
        These positions are excluded when finding the outside peak.
        None means there are no previous detections
        
    inverted : bool, default False
        If True, calculate peaks using errors reflected around the
        window mean. Use the same branch as candidate detection
        
    p : float, default 0.13
        Minimum relative separation between successive ranked peaks.
        A value of 0.13 corresponds to a 13% separation.

    Returns
    -------
    kept_positions : numpy.ndarray
        Candidate positions belonging to surviving intervals.
        Positions remain relative to this error window.
    kept_intervals : list of tuple
        Surviving inclusive (start, end) intervals, with endpoints
        relative to this error window.
    """
    if len(candidate_intervals) == 0:
        return np.array([], dtype=int), []

    window_errors = np.asarray(window_errors, dtype=float)
    candidate_positions = np.asarray(candidate_positions, dtype=int)

    mean_error = np.mean(window_errors)

    errors = (
        2 * mean_error - window_errors
        if inverted
        else window_errors
    )

    if previous_anomaly_positions is None:
        previous_anomaly_positions = np.array([], dtype=int)

    # Full array error positions covered by this window.
    window_positions = window_start + np.arange(len(errors))

    # Exclude current candidates and previously retained anomalies.
    excluded_positions = np.concatenate([
        np.asarray(previous_anomaly_positions, dtype=int),
        window_start + candidate_positions
    ])

    outside_positions = np.setdiff1d(
        window_positions, excluded_positions
    ) - window_start

    outside_peak = np.max(errors[outside_positions])

    # Find each candidate interval's peak.
    peaks = np.array([
        np.max(errors[start:end + 1])
        for start, end in candidate_intervals
    ])

    # Rank candidate peaks and append the largest outside error.
    ranked_peaks = np.append(
        np.sort(peaks)[::-1],
        outside_peak
    )

    indices_to_remove = []

    for i in range(len(ranked_peaks) - 1):
        separation = (
            ranked_peaks[i] - ranked_peaks[i + 1]
        ) / ranked_peaks[i]

        if separation < p:
            indices_to_remove.extend(
                np.flatnonzero(peaks == ranked_peaks[i]).tolist()
            )
        else:
            # Preserve the authors' rule: a sufficient gap clears
            # previously marked removals.
            indices_to_remove = []

    kept_intervals = [
        interval
        for i, interval in enumerate(candidate_intervals)
        if i not in indices_to_remove
    ]

    # Retain only positions belonging to surviving intervals.
    keep_mask = np.zeros(len(errors), dtype=bool)

    for start, end in kept_intervals:
        keep_mask[start:end + 1] = True

    kept_positions = candidate_positions[
        keep_mask[candidate_positions]
    ]

    return kept_positions, kept_intervals

# This function calculates absolute errors from aligned actual values and predictions, smooths them, and optionally applies the initial adjustment.

def calculate_prediction_errors(
    actual,
    predictions,
    batch_size=70,
    window_size=30,
    smoothing_perc=0.05,
    history_length=250,
    initial_adjustment=True):
    
    """Return raw and smoothed absolute prediction errors.

    Parameters
    ----------
    actual : array-like, shape (n,) or (n, 1)
        Actual telemetry values aligned with the predictions.
        actual[i] must correspond to the same time step as predictions[i].
        Supply only the target sensor values, not all feature columns
        
    predictions : array-like, shape (n,) or (n, 1)
        One scalar prediction for each corresponding actual value.
        Both arrays must use the same scaling
        
    batch_size : int, default 70
        Number of readings per detection batch. Used with window_size
        and smoothing_perc to calculate the smoothing span
        
    window_size : int, default 30
        Number of batches per detection window
        
    smoothing_perc : float, default 0.05
        Fraction of the detection-window length used for the EWMA span:
        int(batch_size * window_size * smoothing_perc)
        
    history_length : int, default 250
        Number of past readings used to make a prediction.
        Used here only for the optional initial adjustment
        
    initial_adjustment : bool, default True
        Replace the first history_length smoothed errors with the mean
        of the first 2 * history_length smoothed errors, matching the
        P-1 reproduction. Available values are used for shorter arrays.
        Set False to disable this adjustment.

    Returns
    -------
    raw_errors : numpy.ndarray, shape (n,)
        Absolute differences between aligned actual values and predictions
        
    smoothed_errors : numpy.ndarray, shape (n,)
        Exponentially smoothed errors, including the initial adjustment
        when enabled
    """
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    # Accept column vectors while preventing accidental broadcasting.
    if actual.ndim == 2 and actual.shape[1] == 1:
        actual = actual[:, 0]

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]

    if actual.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            "actual and predictions must have shape (n,) or (n, 1)."
        )

    if actual.shape != predictions.shape or len(actual) == 0:
        raise ValueError(
            "actual and predictions must have the same nonzero length."
        )

    if not np.all(np.isfinite(actual)) or not np.all(
        np.isfinite(predictions)
    ):
        raise ValueError("actual and predictions must contain finite values.")

    smoothing_span = int(batch_size * window_size * smoothing_perc)

    if smoothing_span < 1:
        raise ValueError("The calculated smoothing span must be at least 1.")

    if not isinstance(history_length, (int, np.integer)) or history_length < 1:
        raise ValueError("history_length must be a positive integer.")

    raw_errors = np.abs(actual - predictions)

    smoothed_errors = (
        pd.Series(raw_errors)
        .ewm(span=smoothing_span, adjust=True)
        .mean()
        .to_numpy(copy=True)
    )

    if initial_adjustment:
        smoothed_errors[:history_length] = np.mean(
            smoothed_errors[:2 * history_length]
        )

    return raw_errors, smoothed_errors

# This function runs the complete detection procedure and returns detected
# intervals, prediction errors, and diagnostics for each window and branch.

def detect_anomalies(
    test_data,
    predictions,
    history_length=250,
    n_predictions=10,
    target_column=0,
    batch_size=70,
    window_size=30,
    smoothing_perc=0.05,
    error_buffer=100,
    p=0.13,
    initial_adjustment=True,
    initial_ignore=None
):
    """Run the full detection procedure using corrected pruning indexing.

    Parameters
    ----------
    test_data : array-like, shape (n_test,) or (n_test, n_features)
        Full test telemetry for one channel. For a two-dimensional array,
        target_column selects the sensor being predicted
        
    predictions : array-like, shape (n,) or (n, 1)
        One scalar prediction per time step, using the same scaling as
        test_data. predictions[i] must correspond to test-data position
        history_length + i. Aggregate multi-horizon model outputs into
        this scalar series before calling
        
    history_length : int, default 250
        Number of readings before the first prediction. Also controls
        the optional initial smoothing adjustment and the default
        initial ignored region
        
    n_predictions : int, default 10
        Number of consecutive actual targets associated with each
        prediction for the telemetry-scale checks. This was 10 in our
        P-1 reproduction. It is not the width of predictions
        
    target_column : int, default 0
        Sensor column to use when test_data is two-dimensional.
    batch_size : int, default 70
        Number of positions by which detection windows advance
        
    window_size : int, default 30
        Number of batches per detection window
        
    smoothing_perc : float, default 0.05
        Fraction of the detection-window length used for the EWMA span
        
    error_buffer : int, default 100
        Buffer setting passed to threshold selection and candidate
        detection
        
    p : float, default 0.13
        Peak-separation setting passed to pruning
        
    initial_adjustment : bool, default True
        Apply the initial smoothing adjustment used for P-1
        
    initial_ignore : int or None, default None
        Number of initial error positions excluded from first-window
        candidates. None uses twice history_length; 0 disables exclusion.

    Returns
    -------
    results : dict
        detected_intervals:
            Inclusive (start, end) pairs in ORIGINAL test-data indices.
        retained_positions:
            Retained positions relative to the FULL prediction/error array.
        prediction_indices:
            Original test-data index corresponding to each prediction.
        raw_errors:
            Absolute errors before smoothing.
        smoothed_errors:
            Errors used for detection.
        window_results:
            DataFrame with thresholds and retained counts for both
            branches of every window.

    Notes
    -----
    Candidate positions are local to each window. Previous detections
    are stored in full-error-array coordinates. window_start connects
    these coordinate systems during pruning.

    The final window includes all remaining errors. If fewer errors
    than one full window are available, process them in one window.
    """
    test_data = np.asarray(test_data, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if test_data.ndim == 2:
        telemetry = test_data[:, target_column]
    elif test_data.ndim == 1:
        telemetry = test_data
    else:
        raise ValueError("test_data must be one- or two-dimensional.")

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]

    if predictions.ndim != 1 or len(predictions) == 0:
        raise ValueError(
            "predictions must have shape (n,) or (n, 1), with n > 0."
        )

    for name, value in [
        ("history_length", history_length),
        ("n_predictions", n_predictions),
        ("batch_size", batch_size),
        ("window_size", window_size),
        ("error_buffer", error_buffer)
    ]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    if initial_ignore is not None:
        if (
            not isinstance(initial_ignore, (int, np.integer))
            or initial_ignore < 0
        ):
            raise ValueError(
                "initial_ignore must be a nonnegative integer or None."
            )

    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")

    # Ensure all future actual targets required by the scale checks exist.
    required_length = (
        history_length + len(predictions) + n_predictions - 1
    )

    if len(telemetry) < required_length:
        raise ValueError(
            "test_data is too short for the supplied predictions, "
            "history_length, and n_predictions."
        )

    if not np.all(np.isfinite(telemetry)):
        raise ValueError("Telemetry must contain finite values.")

    prediction_indices = (
        history_length + np.arange(len(predictions))
    )

    actual = telemetry[prediction_indices]

    raw_errors, smoothed_errors = calculate_prediction_errors(
        actual=actual,
        predictions=predictions,
        batch_size=batch_size,
        window_size=window_size,
        smoothing_perc=smoothing_perc,
        history_length=history_length,
        initial_adjustment=initial_adjustment
    )

    window_length = batch_size * window_size

    n_advances = max(
        0,
        (len(smoothed_errors) - window_length) // batch_size
    )

    retained_positions = np.array([], dtype=int)
    window_rows = []

    for window_number in range(n_advances + 1):
        start = window_number * batch_size
        stop = start + window_length

        if window_number == n_advances:
            stop = len(smoothed_errors)

        errors = smoothed_errors[start:stop]

        target_indices = (
            prediction_indices[start:stop, None]
            + np.arange(n_predictions)
        )
        targets = telemetry[target_indices]

        # Both branches use detections from PREVIOUS windows only.
        previous_positions = retained_positions.copy()
        branch_positions = []

        for inverted in [False, True]:
            threshold, selected_z = select_threshold(
                errors,
                error_buffer=error_buffer,
                inverted=inverted
            )

            candidates, intervals = find_window_candidates(
                window_errors=errors,
                window_targets=targets,
                threshold=threshold,
                window_number=window_number,
                inverted=inverted,
                error_buffer=error_buffer,
                history_length=history_length,
                batch_size=batch_size,
                initial_ignore=initial_ignore
            )

            kept_positions, _ = prune_window_candidates(
                window_errors=errors,
                candidate_positions=candidates,
                candidate_intervals=intervals,
                window_start=start,
                previous_anomaly_positions=previous_positions,
                inverted=inverted,
                p=p
            )

            # Convert local positions to full-error-array positions.
            branch_positions.append(start + kept_positions)

            window_rows.append({
                "window": window_number,
                "branch": "low errors" if inverted else "high errors",
                "z": selected_z,
                "threshold": threshold,
                "retained readings": len(kept_positions)
            })

        retained_positions = np.unique(np.concatenate([
            retained_positions,
            *branch_positions
        ]))

    # Merge consecutive retained positions into final intervals.
    groups = np.split(
        retained_positions,
        np.flatnonzero(np.diff(retained_positions) > 1) + 1
    )

    detected_intervals = [
        (
            int(prediction_indices[group[0]]),
            int(prediction_indices[group[-1]])
        )
        for group in groups
        if len(group) > 1
    ]

    return {
        "detected_intervals": detected_intervals,
        "retained_positions": retained_positions,
        "prediction_indices": prediction_indices,
        "raw_errors": raw_errors,
        "smoothed_errors": smoothed_errors,
        "window_results": pd.DataFrame(window_rows)
    }


# This function compares detected intervals with labeled events and
# reports event-level detection counts, precision, recall, and F1.

def evaluate_event_detection(detected_intervals, labeled_intervals):
    """Evaluate detections using inclusive interval overlap.

    Parameters
    ----------
    detected_intervals : iterable of (start, end)
        Detected anomaly intervals, such as the output from
        detect_anomalies. Endpoints are inclusive and must refer
        to original test-data indices.
    labeled_intervals : iterable of (start, end)
        Ground-truth anomaly intervals for the same channel,
        using the same indices and inclusive endpoints.

    Returns
    -------
    evaluation : dict
        TP:
            Number of labeled events overlapping at least one detection.
        FP:
            Number of detected intervals overlapping no labeled event.
        FN:
            Number of labeled events overlapping no detection.
        precision, recall, f1:
            Event-level metrics. Undefined values are returned as np.nan.
        detected_event_mask:
            Boolean array indicating which labeled events were detected.
        false_alert_mask:
            Boolean array indicating which detected intervals are false
            alerts.

    Notes
    -----
    Any overlap, including a shared endpoint, counts as detection.
    Each labeled event is counted once, even if multiple alerts overlap it.
    One alert can detect multiple labeled events if it overlaps them.

    These metrics measure event detection, not boundary accuracy or
    point-by-point classification. They do not penalize multiple alerts
    overlapping the same event or measure the duration of false alerts.
    """
    detected = list(detected_intervals)
    labeled = list(labeled_intervals)

    # Validate that every interval has two ordered integer endpoints.
    for name, intervals in [
        ("detected_intervals", detected),
        ("labeled_intervals", labeled)
    ]:
        for interval in intervals:
            if len(interval) != 2:
                raise ValueError(
                    f"Each interval in {name} must contain two endpoints."
                )

            start, end = interval

            if not all(
                isinstance(value, (int, np.integer))
                for value in (start, end)
            ):
                raise ValueError(
                    f"Endpoints in {name} must be integers."
                )

            if start < 0 or end < start:
                raise ValueError(
                    f"Intervals in {name} must satisfy 0 <= start <= end."
                )

    detected_event_mask = np.zeros(len(labeled), dtype=bool)
    false_alert_mask = np.ones(len(detected), dtype=bool)

    for i, (detected_start, detected_end) in enumerate(detected):
        for j, (labeled_start, labeled_end) in enumerate(labeled):

            # Inclusive intervals overlap when neither lies wholly
            # before the other.
            overlaps = (
                detected_start <= labeled_end
                and labeled_start <= detected_end
            )

            if overlaps:
                detected_event_mask[j] = True
                false_alert_mask[i] = False

    tp = int(np.sum(detected_event_mask))
    fp = int(np.sum(false_alert_mask))
    fn = int(np.sum(~detected_event_mask))

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan

    f1_denominator = 2 * tp + fp + fn
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0
        else np.nan
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detected_event_mask": detected_event_mask,
        "false_alert_mask": false_alert_mask
    }
    