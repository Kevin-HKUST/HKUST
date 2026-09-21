import numpy as np

def create_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates an artificial dataset with dimension (1826, 50).
    """
    np.random.seed(seed)
    temperatures = np.random.normal(15, 10, (1826, 50))

    days = np.arange(1826)
    seasonal_effect = 10 * np.sin(2 * np.pi * days / 365.25)
    temperatures += seasonal_effect.reshape(-1, 1)

    mask = np.random.random(temperatures.shape) < 0.05
    temperatures[mask] = np.nan

    temperatures_no_nan = np.nan_to_num(temperatures, nan=np.nanmean(temperatures))
    return (temperatures, temperatures_no_nan)


# ============================================================
# Task 1: Basic Threshold Counting
# ============================================================

def count_special_days(temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the number of special days per city.
    Tropical: >30 | Freezing: <0 | Pleasant: 15~25 inclusive
    """
    tropical = np.sum(temperatures > 30, axis=0)
    freezing = np.sum(temperatures < 0, axis=0)
    pleasant = np.sum((temperatures >= 15) & (temperatures <= 25), axis=0)
    return tropical, freezing, pleasant


def find_most_extreme_city(tropical_days: np.ndarray, freezing_days: np.ndarray) -> int:
    """
    Finds the city with the largest number of tropical + freezing days.
    """
    return int(np.argmax(tropical_days + freezing_days))


# ============================================================
# Task 2: Consecutive Pattern Detection
# ============================================================

def find_true_sequence_lengths(mask: np.ndarray) -> np.ndarray:
    """
    Computes the lengths of all continuous sequences of True in the given boolean mask.
    e.g. [F, F, T, T, F, T, F] -> [2, 1]
    """
    if mask.size == 0:
        return np.array([], dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))

    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]

    return ends - starts


def compute_heatwave_stats(temperatures: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Heatwave = 3+ consecutive days with temperature > 35.
    Returns (heatwave_count_per_city, (city_of_longest, length_of_longest)).
    """
    num_cities = temperatures.shape[1]
    heatwave_counts = np.zeros(num_cities, dtype=int)
    longest_city = 0
    longest_length = 0

    for city in range(num_cities):
        mask = temperatures[:, city] > 35
        seq_lengths = find_true_sequence_lengths(mask)
        heatwave_lengths = seq_lengths[seq_lengths >= 3]
        heatwave_counts[city] = len(heatwave_lengths)

        if len(heatwave_lengths) > 0:
            max_len = int(np.max(heatwave_lengths))
            if max_len > longest_length:
                longest_length = max_len
                longest_city = city

    return heatwave_counts, (longest_city, longest_length)


# ============================================================
# Task 3: Running Window Analysis
# ============================================================

def compute_n_day_mean(temperatures: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Computes the n-day moving average using sliding_window_view.
    Output shape: (n_days - window_size + 1, n_cities)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    # windows shape: (n_days - window_size + 1, n_cities, window_size)
    windows = sliding_window_view(temperatures, window_size, axis=0)
    return np.mean(windows, axis=2)


def count_warm_periods(temperatures: np.ndarray) -> np.ndarray:
    """
    Counts the number of 7-day windows with mean > 25°C per city.
    """
    means = compute_n_day_mean(temperatures, 7)
    return np.sum(means > 25, axis=0)


def find_sharpest_changes(temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Groups into 30-day months, computes monthly means, 
    finds sharpest increase/decrease via np.diff.
    """
    n_months = temperatures.shape[0] // 30
    trimmed = temperatures[:n_months * 30]
    grouped = trimmed.reshape(n_months, 30, -1)             # (60, 30, 50)
    monthly_means = np.mean(grouped, axis=1)                # (60, 50)
    diffs = np.diff(monthly_means, axis=0)                  # (59, 50)

    largest_increase = np.argmax(diffs, axis=0)
    largest_decrease = np.argmin(diffs, axis=0)
    return largest_increase, largest_decrease


# ============================================================
# Task 4: Multi-Criteria Counting
# ============================================================

def find_record_breaking_temperatures(temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Record-breaking high: cumulative max 刷新次数
    Record-breaking low:  cumulative min 刷新次数
    Rapid change: |day-to-day diff| > 10
    """
    # ---- Record-breaking highs ----
    cummax = np.maximum.accumulate(temperatures, axis=0)
    record_high_mask = np.zeros_like(temperatures, dtype=bool)

    record_high_mask[1:] = cummax[1:] > cummax[:-1]
    record_high = np.sum(record_high_mask, axis=0)

    # ---- Record-breaking lows ----
    cummin = np.minimum.accumulate(temperatures, axis=0)
    record_low_mask = np.zeros_like(temperatures, dtype=bool)

    record_low_mask[1:] = cummin[1:] < cummin[:-1]
    record_low = np.sum(record_low_mask, axis=0)

    # ---- Rapid temperature changes ----
    rapid_changes = np.sum(np.abs(np.diff(temperatures, axis=0)) > 10, axis=0)

    return record_high, record_low, rapid_changes


def find_threshold_crossings(temperatures: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Counts crossings: strictly below → strictly above, or vice versa.
    """
    below = temperatures < threshold
    above = temperatures > threshold
    crossings = (below[:-1] & above[1:]) | (above[:-1] & below[1:])
    return np.sum(crossings, axis=0)


# ============================================================
# Task 5: Distribution Analysis
# ============================================================

def convert_to_histogram(temperatures: np.ndarray) -> np.ndarray:
    """
    Bins: -10 to 45, step 5 (11 bins). Handles NaN by filtering them out.
    Returns shape (11, n_cities).
    """
    bins = np.arange(-10, 50, 5)   # [-10, -5, 0, 5, ..., 40, 45]  — 12 edges, 11 bins
    num_cities = temperatures.shape[1]
    result = np.zeros((len(bins) - 1, num_cities), dtype=int)

    for city in range(num_cities):
        city_data = temperatures[:, city]
        city_data = city_data[~np.isnan(city_data)]
        hist, _ = np.histogram(city_data, bins=bins)
        result[:, city] = hist

    return result


def find_correlated_pair(temperatures: np.ndarray) -> tuple[int, int]:
    """
    Finds the pair of cities whose histograms have the highest Pearson correlation.
    """
    hist = convert_to_histogram(temperatures)               # (11, 50)
    corr = np.corrcoef(hist.T)                              # (50, 50)
    np.fill_diagonal(corr, -np.inf)
    idx = np.argmax(corr)
    i, j = np.unravel_index(idx, corr.shape)
    return (min(i, j), max(i, j))


# ============================================================
# Task 6: Pattern Matching
# ============================================================

def count_extreme_weather_patterns(temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    mid   = temperatures[1:-1]
    left  = temperatures[:-2]
    right = temperatures[2:]
    spikes = (mid - left > 5) & (mid - right > 5)
    heat_spikes = np.sum(spikes, axis=0)

    num_cities = temperatures.shape[1]
    cold_snaps = np.zeros(num_cities, dtype=int)
    for city in range(num_cities):
        cold_mask = temperatures[:, city] < 5
        seq_lengths = find_true_sequence_lengths(cold_mask)
        cold_snaps[city] = np.sum(seq_lengths >= 3)

    return heat_spikes, cold_snaps


if __name__ == "__main__":
    print("This is a library and should not be run directly. Please import it instead.")