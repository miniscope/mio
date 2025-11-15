"""
Script to find frames that light up in behavior and neural videos,
map them to unix timestamps, and compare delays between them.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from mio.io import VideoReader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def find_video_csv_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """
    Find all AVI video files and their corresponding CSV files in a folder.
    
    Args:
        folder: Path to folder containing videos and CSVs
        
    Returns:
        List of tuples (video_path, csv_path)
    """
    pairs = []
    avi_files = sorted(folder.glob("*.avi"))
    
    for avi_file in avi_files:
        # Try to find matching CSV with same stem
        csv_file = folder / f"{avi_file.stem}.csv"
        if csv_file.exists():
            pairs.append((avi_file, csv_file))
        else:
            print(f"Warning: No CSV found for {avi_file.name}")
    
    return pairs


def load_behavior_timestamps(csv_path: Path) -> pd.DataFrame:
    """Load behavior CSV: frame_index,unix_timestamp_ms (already in ms)."""
    return pd.read_csv(csv_path, header=None, names=['frame_index', 'unix_timestamp_ms'])


def load_neural_timestamps(csv_path: Path) -> pd.DataFrame:
    """Load neural CSV, convert seconds to ms. Returns frame_index, unix_timestamp_ms."""
    df = pd.read_csv(csv_path)
    valid_df = df[df['reconstructed_frame_index'] != -1].copy()
    frame_timestamps = (
        valid_df.groupby('reconstructed_frame_index')['buffer_recv_unix_time']
        .first()
        .reset_index()
    )
    frame_timestamps.columns = ['frame_index', 'unix_timestamp']
    # Convert seconds to ms if needed
    if frame_timestamps['unix_timestamp'].max() < 1e11:
        frame_timestamps['unix_timestamp_ms'] = frame_timestamps['unix_timestamp'] * 1000.0
    else:
        frame_timestamps['unix_timestamp_ms'] = frame_timestamps['unix_timestamp']
    return frame_timestamps[['frame_index', 'unix_timestamp_ms']]


def detect_rises_falls(
    thresholded_signal: np.ndarray,
    time_bins: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Detect rises (0->1) and falls (1->0), ignoring NaN."""
    rises, falls = [], []
    for i in range(1, len(thresholded_signal)):
        prev_val, curr_val = thresholded_signal[i-1], thresholded_signal[i]
        if np.isnan(prev_val) or np.isnan(curr_val):
            continue
        if prev_val < 0.5 and curr_val >= 0.5:
            rises.append(time_bins[i])
        elif prev_val >= 0.5 and curr_val < 0.5:
            falls.append(time_bins[i])
    return rises, falls


def match_rises_falls_and_calculate_delays(
    beh_rises: List[float],
    beh_falls: List[float],
    neur_rises: List[float],
    neur_falls: List[float],
    max_time_window_ms: float = 2000.0
) -> Tuple[List[dict], List[dict]]:
    """Match rises/falls and calculate delays."""
    def match_events(beh_times: List[float], neur_times: List[float]) -> List[dict]:
        neur_array = np.array(neur_times)
        used = set()
        delays = []
        for beh_time in beh_times:
            diffs = neur_array - beh_time
            valid = [i for i in np.where(np.abs(diffs) <= max_time_window_ms)[0] if i not in used]
            if valid:
                idx = valid[np.argmin(np.abs(diffs[valid]))]
                delays.append({
                    'behavior_time_ms': beh_time,
                    'neural_time_ms': neur_times[idx],
                    'delay_ms': neur_times[idx] - beh_time
                })
                used.add(idx)
        return delays
    return match_events(beh_rises, neur_rises), match_events(beh_falls, neur_falls)


def plot_waveforms_and_thresholded(
    behavior_normalized_data: List[Tuple[str, Tuple, np.ndarray]],
    neural_normalized_data: List[Tuple[str, Tuple, np.ndarray]],
    beh_thresholded: np.ndarray,
    neur_thresholded: np.ndarray,
    time_bins: np.ndarray,
    threshold: float,
    min_spike_frames: int,
    output_path: Optional[Path] = None,
    interactive: bool = True
) -> None:
    """
    Plot normalized brightness waveforms and thresholded signals.
    
    Args:
        behavior_normalized_data: List of (name, times, normalized_values) for behavior
        neural_normalized_data: List of (name, times, normalized_values) for neural
        beh_thresholded: Thresholded behavior signal
        neur_thresholded: Thresholded neural signal
        time_bins: Time bins for thresholded signals
        threshold: Threshold value used
        min_spike_frames: Minimum spike frames used
        output_path: Optional path to save the plot
    """
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is not available. Install it with: pip install matplotlib"
        )
    
    # Create plots: normalized overlay + thresholded overlay
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot all normalized data on same subplot
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray']
    for idx, (beh_name, beh_times, beh_normalized) in enumerate(behavior_normalized_data):
        color = colors[idx % len(colors)]
        ax1.plot(
            beh_times, beh_normalized, color=color, alpha=0.7, linewidth=0.5,
            label=f'Behavior: {beh_name}'
        )
    
    neur_start_idx = len(behavior_normalized_data)
    for idx, (neur_name, neur_times, neur_normalized) in enumerate(neural_normalized_data):
        color = colors[(neur_start_idx + idx) % len(colors)]
        # Handle NaN values by masking them to avoid disconnected traces
        neur_times_array = np.array(neur_times)
        neur_normalized_array = np.array(neur_normalized)
        valid_mask = ~np.isnan(neur_normalized_array)
        if np.any(valid_mask):
            ax1.plot(
                neur_times_array[valid_mask], neur_normalized_array[valid_mask],
                color=color, alpha=0.7, linewidth=0.5,
                label=f'Neural: {neur_name}'
            )
    
    ax1.set_ylabel('Normalized Brightness')
    ax1.set_title('Normalized Brightness (All)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.1, 1.1)
    ax1.axhline(y=threshold, color='k', linestyle='--', alpha=0.5)
    
    # Plot thresholded overlay (handle NaN values by masking them)
    beh_masked = np.ma.masked_invalid(beh_thresholded)
    neur_masked = np.ma.masked_invalid(neur_thresholded)
    ax2.plot(time_bins, beh_masked, 'b-', alpha=0.7, linewidth=1.0, label='Behavior')
    ax2.plot(
        time_bins, neur_masked, 'r-', alpha=0.7, linewidth=1.0, label='Neural (combined)'
    )
    ax2.set_xlabel('Unix Time (milliseconds)')
    ax2.set_ylabel('Thresholded (0 or 1)')
    title = f'Thresholded Overlay (threshold={threshold}, min={min_spike_frames} frames)'
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-0.1, 1.1)
    
    # Share x-axis
    ax1.sharex(ax2)
    ax1.set_xlabel('')  # Remove duplicate x-label from top plot
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"    Waveform plot saved to: {output_path}")
    
    if interactive:
        plt.show(block=True)
    elif not output_path:
        plt.show()
    else:
        plt.close()


def plot_delay_scatter(
    all_delays: List[dict],
    output_path: Optional[Path] = None,
    title: str = "Delay vs Unix Time",
    interactive: bool = True
) -> None:
    """
    Create a scatter plot of delay vs unix time.
    
    Args:
        all_delays: List of delay dictionaries with 'behavior_time_ms' and 'delay_ms'
        output_path: Optional path to save the plot
        title: Plot title
    """
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is not available. Install it with: pip install matplotlib"
        )
    
    if not all_delays:
        print("    No delays to plot")
        return
    
    times = [d['behavior_time_ms'] for d in all_delays]
    delays = [d['delay_ms'] for d in all_delays]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(times, delays, alpha=0.6, s=20)
    ax.set_xlabel('Unix Time (milliseconds)')
    ax.set_ylabel('Delay (ms)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add mean and SD lines
    mean_delay = np.mean(delays)
    std_delay = np.std(delays)
    ax.axhline(y=mean_delay, color='r', linestyle='--', label=f'Mean: {mean_delay:.2f} ms')
    mean_plus_sd = mean_delay + std_delay
    mean_minus_sd = mean_delay - std_delay
    ax.axhline(
        y=mean_plus_sd, color='orange', linestyle=':', alpha=0.7,
        label=f'Mean + SD: {mean_plus_sd:.2f} ms'
    )
    ax.axhline(
        y=mean_minus_sd, color='orange', linestyle=':', alpha=0.7,
        label=f'Mean - SD: {mean_minus_sd:.2f} ms'
    )
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"    Delay scatter plot saved to: {output_path}")
    
    if interactive:
        plt.show(block=True)
    elif not output_path:
        plt.show()
    else:
        plt.close()


def process_behavior_neural_pair(
    behavior_pairs: List[Tuple[Path, Path]],
    neural_pairs: List[Tuple[Path, Path]],
    threshold: float = 0.5,
    min_spike_frames: int = 5,
    frame_interval_ms: float = 50.0
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]]:
    """
    Process behavior and neural videos to create thresholded signals.
    
    Args:
        behavior_pairs: List of (video_path, csv_path) tuples for behavior files
        neural_pairs: List of (video_path, csv_path) tuples for neural files
        threshold: Brightness threshold for normalization (0-1)
        min_spike_frames: Minimum spike duration in frames
        frame_interval_ms: Frame interval in milliseconds
        
    Returns:
        Tuple of (behavior_thresholded, neural_thresholded, time_bins, 
                 behavior_normalized_data, neural_normalized_data) or None if processing fails
    """
    print("    Calculating brightness for all frames...")
    
    # Process all behavior videos (same way as neural)
    # Ignore first 5 frames of each recording
    frames_to_skip = 5
    behavior_data = []
    for beh_video, beh_csv in behavior_pairs:
        beh_brightness = []
        beh_timestamps = []
        reader = VideoReader(str(beh_video))
        beh_timestamps_df = load_behavior_timestamps(beh_csv)
        
        try:
            for frame_idx, frame in reader.read_frames():
                # Skip first 5 frames
                if frame_idx < frames_to_skip:
                    continue
                
                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if len(frame.shape) == 3
                    else frame
                )
                mean_brightness = np.mean(gray)
                beh_brightness.append(mean_brightness)
                
                matches = beh_timestamps_df[beh_timestamps_df['frame_index'] == frame_idx]
                if len(matches) > 0:
                    timestamp_ms = matches.iloc[0]['unix_timestamp_ms']
                    beh_timestamps.append(timestamp_ms)
                else:
                    beh_timestamps.append(None)
        finally:
            reader.release()
        
        # Filter out None timestamps
        beh_data_filtered = [
            (ts, br) for ts, br in zip(beh_timestamps, beh_brightness) if ts is not None
        ]
        if beh_data_filtered:
            behavior_data.append((beh_video.stem, beh_data_filtered))
    
    # Process all neural videos
    # Ignore first 5 frames of each recording
    neural_data = []
    for neur_video, neur_csv in neural_pairs:
        neur_brightness = []
        neur_timestamps = []
        reader = VideoReader(str(neur_video))
        neur_timestamps_df = load_neural_timestamps(neur_csv)
        
        try:
            for frame_idx, frame in reader.read_frames():
                # Skip first 5 frames
                if frame_idx < frames_to_skip:
                    continue
                
                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if len(frame.shape) == 3
                    else frame
                )
                mean_brightness = np.mean(gray)
                neur_brightness.append(mean_brightness)
                
                matches = neur_timestamps_df[neur_timestamps_df['frame_index'] == frame_idx]
                if len(matches) > 0:
                    timestamp_ms = matches.iloc[0]['unix_timestamp_ms']
                    neur_timestamps.append(timestamp_ms)
                else:
                    neur_timestamps.append(None)
        finally:
            reader.release()
        
        # Filter out None timestamps
        neur_data_filtered = [
            (ts, br) for ts, br in zip(neur_timestamps, neur_brightness) if ts is not None
        ]
        if neur_data_filtered:
            neural_data.append((neur_video.stem, neur_data_filtered))
    
    if not behavior_data or not neural_data:
        print("    Warning: No valid timestamp data for plotting")
        return
    
    # Normalize each behavior file separately (for individual plots)
    behavior_normalized_data = []
    for beh_name, beh_data_filtered in behavior_data:
        beh_times, beh_vals = zip(*beh_data_filtered)
        beh_vals_array = np.array(beh_vals)
        beh_min = beh_vals_array.min()
        beh_max = beh_vals_array.max()
        beh_range = beh_max - beh_min if beh_max > beh_min else 1.0
        beh_normalized = (beh_vals_array - beh_min) / beh_range
        behavior_normalized_data.append((beh_name, beh_times, beh_normalized))
    
    # Normalize each neural file separately (for individual plots)
    neural_normalized_data = []
    for neur_name, neur_data_filtered in neural_data:
        neur_times, neur_vals = zip(*neur_data_filtered)
        neur_vals_array = np.array(neur_vals)
        neur_min = neur_vals_array.min()
        neur_max = neur_vals_array.max()
        neur_range = neur_max - neur_min if neur_max > neur_min else 1.0
        neur_normalized = (neur_vals_array - neur_min) / neur_range
        neural_normalized_data.append((neur_name, neur_times, neur_normalized))
    
    # Find overall time range for thresholded plot - combine all behavior data
    all_behavior_times = []
    all_behavior_vals = []
    for _, beh_data_filtered in behavior_data:
        beh_times, beh_vals = zip(*beh_data_filtered)
        all_behavior_times.extend(beh_times)
        all_behavior_vals.extend(beh_vals)
    
    if not all_behavior_times:
        print("    Warning: No behavior data to plot")
        return
    
    # Find overall time range for thresholded plot - combine all neural data
    all_neural_times = []
    all_neural_vals = []
    for _, neur_data_filtered in neural_data:
        neur_times, neur_vals = zip(*neur_data_filtered)
        all_neural_times.extend(neur_times)
        all_neural_vals.extend(neur_vals)
    
    if not all_neural_times:
        print("    Warning: No neural data to plot")
        return
    
    # Normalize combined behavior for thresholded plot
    beh_vals_array = np.array(all_behavior_vals)
    beh_min, beh_max = beh_vals_array.min(), beh_vals_array.max()
    beh_range = beh_max - beh_min if beh_max > beh_min else 1.0
    beh_normalized_combined = (beh_vals_array - beh_min) / beh_range
    
    # Normalize combined neural for thresholded plot
    neur_vals_array = np.array(all_neural_vals)
    neur_min, neur_max = neur_vals_array.min(), neur_vals_array.max()
    neur_range = neur_max - neur_min if neur_max > neur_min else 1.0
    neur_normalized_combined = (neur_vals_array - neur_min) / neur_range
    
    # Sort for interpolation
    beh_sorted_indices = np.argsort(all_behavior_times)
    beh_times_sorted = np.array(all_behavior_times)[beh_sorted_indices]
    beh_normalized_sorted = beh_normalized_combined[beh_sorted_indices]
    
    neur_sorted_indices = np.argsort(all_neural_times)
    neur_times_sorted = np.array(all_neural_times)[neur_sorted_indices]
    neur_normalized_sorted = neur_normalized_combined[neur_sorted_indices]
    
    min_time = min(min(all_behavior_times), min(all_neural_times))
    max_time = max(max(all_behavior_times), max(all_neural_times))
    time_bins = np.arange(min_time, max_time + 1, 1)
    
    # Interpolate with forward fill (keep previous state instead of zero)
    def interp_with_forward_fill(
        x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray
    ) -> np.ndarray:
        """Interpolate with forward fill for missing values."""
        result = np.interp(x_new, x_old, y_old, left=np.nan, right=np.nan)
        # Forward fill NaN values
        mask = np.isnan(result)
        if np.any(mask):
            # Find first valid value
            first_valid_idx = np.where(~mask)[0]
            if len(first_valid_idx) > 0:
                first_valid = first_valid_idx[0]
                # Forward fill from first valid value
                last_valid = result[first_valid]
                for i in range(len(result)):
                    if not np.isnan(result[i]):
                        last_valid = result[i]
                    else:
                        result[i] = last_valid
            else:
                # All NaN, fill with 0
                result[mask] = 0
        return result
    
    beh_continuous = interp_with_forward_fill(time_bins, beh_times_sorted, beh_normalized_sorted)
    neur_continuous = interp_with_forward_fill(time_bins, neur_times_sorted, neur_normalized_sorted)
    
    # Threshold and filter
    beh_thresholded = (beh_continuous >= threshold).astype(float)
    neur_thresholded = (neur_continuous >= threshold).astype(float)
    
    min_spike_duration_bins = int(np.ceil(min_spike_frames * frame_interval_ms))
    
    def filter_spikes(signal: np.ndarray, min_duration: int) -> np.ndarray:
        """
        Remove spikes/dips shorter than min_duration.
        Marks unstable segments as NaN (ignored).
        """
        filtered = signal.copy().astype(float)
        i = 0
        while i < len(filtered):
            val = filtered[i]
            start = i
            while i < len(filtered) and filtered[i] == val:
                i += 1
            duration = i - start
            if duration < min_duration:
                # Mark unstable segment as NaN (will be ignored)
                filtered[start:i] = np.nan
        return filtered
    
    beh_thresholded = filter_spikes(beh_thresholded, min_spike_duration_bins)
    neur_thresholded = filter_spikes(neur_thresholded, min_spike_duration_bins)
    
    # Keep NaN values as NaN - they represent ignored unstable segments
    
    return (
        beh_thresholded, neur_thresholded, time_bins,
        behavior_normalized_data, neural_normalized_data
    )


def main() -> None:
    """Main function to process all video/CSV pairs and compare delays."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    behavior_folder = project_root / "user_data" / "sync" / "behavior"
    neural_folder = project_root / "user_data" / "sync" / "neural"
    
    # Configuration
    threshold = 0.5  # Normalized brightness threshold
    min_spike_frames = 5
    frame_interval_ms = 50.0
    max_time_window_ms = 2000.0  # For matching rises/falls
    
    print("Looking for videos in:")
    print(f"  Behavior: {behavior_folder}")
    print(f"  Neural: {neural_folder}")
    print()
    
    # Find all video/CSV pairs
    behavior_pairs = find_video_csv_pairs(behavior_folder)
    neural_pairs = find_video_csv_pairs(neural_folder)
    
    print(f"Found {len(behavior_pairs)} behavior video/CSV pairs")
    print(f"Found {len(neural_pairs)} neural video/CSV pairs")
    print()
    
    if not behavior_pairs or not neural_pairs:
        print("Error: Need at least one behavior and one neural file")
        return
    
    # Process all behavior files together with all neural files
    print(f"\n{'='*60}")
    n_beh = len(behavior_pairs)
    n_neur = len(neural_pairs)
    print(f"Processing {n_beh} behavior file(s) with {n_neur} neural file(s)")
    print(f"{'='*60}")
    
    # Process all behavior files with all neural files combined
    print("  Processing behavior and neural files...")
    result = process_behavior_neural_pair(
        behavior_pairs, neural_pairs,
        threshold=threshold,
        min_spike_frames=min_spike_frames,
        frame_interval_ms=frame_interval_ms
    )
    
    if result is None:
        print("  Skipping - could not process files")
        return
    
    (
        beh_thresholded, neur_thresholded, time_bins,
        behavior_normalized_data, neural_normalized_data
    ) = result
    
    # Create waveform and thresholded plot
    behavior_names = "_".join([p[0].stem for p in behavior_pairs])
    waveform_output = (
        project_root
        / "user_data"
        / "sync"
        / f"waveforms_{behavior_names}.png"
    )
    try:
        plot_waveforms_and_thresholded(
            behavior_normalized_data, neural_normalized_data,
            beh_thresholded, neur_thresholded, time_bins,
            threshold=threshold, min_spike_frames=min_spike_frames,
            output_path=waveform_output, interactive=True
        )
    except Exception as e:
        print(f"  Warning: Could not create waveform plot: {e}")
    
    # Detect rises and falls
    print("  Detecting rises and falls...")
    beh_rises, beh_falls = detect_rises_falls(beh_thresholded, time_bins)
    neur_rises, neur_falls = detect_rises_falls(neur_thresholded, time_bins)
    
    print(f"  Behavior: {len(beh_rises)} rises, {len(beh_falls)} falls")
    print(f"  Neural: {len(neur_rises)} rises, {len(neur_falls)} falls")
    
    # Check if counts match
    rises_match = len(beh_rises) == len(neur_rises)
    falls_match = len(beh_falls) == len(neur_falls)
    
    if not rises_match or not falls_match:
        print("  Warning: Rise/fall counts do not match!")
        print(f"    Rises match: {rises_match} (beh={len(beh_rises)}, neur={len(neur_rises)})")
        print(f"    Falls match: {falls_match} (beh={len(beh_falls)}, neur={len(neur_falls)})")
        print("  Calculating delays anyway...")
    else:
        print("  Rise/fall counts match! Calculating delays...")
    
    # Match rises and falls and calculate delays
    rise_delays, fall_delays = match_rises_falls_and_calculate_delays(
        beh_rises, beh_falls, neur_rises, neur_falls,
        max_time_window_ms=max_time_window_ms
    )
    
    all_delays = rise_delays + fall_delays
    
    if not all_delays:
        print("  Warning: Could not match any rises/falls")
        return
    
    # Calculate statistics
    delay_values = [d['delay_ms'] for d in all_delays]
    mean_delay = np.mean(delay_values)
    std_delay = np.std(delay_values)
    
    print("\n  Delay Statistics:")
    print(f"    Number of matched events: {len(all_delays)}")
    print(f"    Average delay: {mean_delay:.2f} ms ({mean_delay/1000:.4f} s)")
    print(f"    SD of delay: {std_delay:.2f} ms ({std_delay/1000:.4f} s)")
    print(f"    Min delay: {np.min(delay_values):.2f} ms")
    print(f"    Max delay: {np.max(delay_values):.2f} ms")
    
    # Create scatter plot
    behavior_names = "_".join([p[0].stem for p in behavior_pairs])
    scatter_output = (
        project_root
        / "user_data"
        / "sync"
        / f"delay_scatter_{behavior_names}.png"
    )
    
    # Add warning to title if counts don't match
    match_status = ""
    if not rises_match or not falls_match:
        match_status = (
            f" [WARNING: Counts don't match - "
            f"Rises: beh={len(beh_rises)}, neur={len(neur_rises)}; "
            f"Falls: beh={len(beh_falls)}, neur={len(neur_falls)}]"
        )
    
    plot_title = (
        f"Delay vs Unix Time (Combined){match_status}\n"
        f"Mean: {mean_delay:.2f} ms, SD: {std_delay:.2f} ms"
    )
    try:
        plot_delay_scatter(all_delays, scatter_output, plot_title, interactive=True)
    except Exception as e:
        print(f"  Warning: Could not create scatter plot: {e}")


if __name__ == "__main__":
    main()
