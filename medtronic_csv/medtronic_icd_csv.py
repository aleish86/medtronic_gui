#!/usr/bin/env python3
"""
Medtronic ICD Emulator Script - Working Sensing Algorithm

This script emulates a Medtronic ICD algorithm for VT/VF detection using the proven
sensing pipeline from the working code. It processes ZIP files specified in the CSV
database and provides comprehensive analysis including sensing, detection criteria,
and therapy decisions.

"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd
import scipy.signal
import scipy.interpolate
from collections import deque, OrderedDict, namedtuple, defaultdict
import warnings

warnings.filterwarnings('ignore')

# External library imports
try:
    import pywt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import HuberRegressor
    from sklearn.metrics import r2_score

    ADVANCED_FEATURES = True
except ImportError:
    print("Warning: Some advanced features require pywt or sklearn libraries")
    ADVANCED_FEATURES = False


class DAQ_File:
    """
    Data acquisition file processor for zip files containing physiological signals
    """

    def __init__(self, zip_dir, zip_fn, channels=None, search_for_files=False):
        if channels is None:
            channels = ["ecg", "boxa", "boxb", "BP", "bpao", "plethg", "plethi", "plethr", "plethh", "qfin"]

        self.sources = []

        if search_for_files:
            file_index = defaultdict(str)
            for root, dirs, files in os.walk(zip_dir, topdown=False):
                for name in files:
                    fn = name
                    fl = os.path.join(root, name)
                    file_index[fn] = fl
            self.zip_fl = file_index[str(zip_fn)]
        else:
            self.zip_fl = os.path.join(zip_dir, zip_fn)

        print(f"Opening DAQ File: {self.zip_fl}")

        try:
            with zipfile.ZipFile(self.zip_fl, "r") as zip_f:
                sampling_f = zip_f.open("rate.txt", 'r')
                self.sampling_rate = int(np.loadtxt(sampling_f))

                for file in channels:
                    try:
                        file_f = zip_f.open(file + ".txt", 'r')
                        file_d = self._fast_load_txt(file_f)
                        print(f'file: {file}, shape: {file_d.shape}')

                        setattr(self, file, file_d)
                        self.sources.append(file)
                    except Exception as e:
                        print(f"Exception loading {file}: {e}")

                # Create blank array for missing channels
                if hasattr(self, 'ecg'):
                    self.blank = np.zeros_like(self.ecg)
        except Exception as e:
            print(f"Error opening zip file {self.zip_fl}: {e}")
            raise

    @staticmethod
    def _fast_load_txt(file_f):
        return np.array(pd.read_csv(file_f, delimiter=' ', dtype=np.float64, header=None).iloc[:, 0])


class MedtronicSensingAlgorithm:
    """
    Medtronic Sensing Algorithm with T-wave and noise detection
    Uses the proven sensing pipeline from the working code
    """

    def __init__(self, sampling_rate=1000):
        self.original_sampling_rate = sampling_rate
        self.target_sampling_rate = 512  # Medtronic internal sampling rate
        self.processed_signals = {}

    def resample_data(self, signal, original_fs, desired_fs):
        """Resample signal to target sampling rate"""
        if original_fs == desired_fs:
            return signal

        sec = signal.size / original_fs
        new_length = int(sec * desired_fs)
        resampled_signal = scipy.signal.resample(signal, new_length)
        return resampled_signal

    def ecg_filter(self, signal):
        """Apply Medtronic ECG bandpass filter (0.5-25 Hz)"""
        ecg_sos = scipy.signal.butter(5, (0.5, 25), 'band', fs=self.target_sampling_rate, output='sos')
        filtered_signal = scipy.signal.sosfilt(ecg_sos, signal)
        return filtered_signal

    def amplifier(self, signal, channel_type='bipecg'):
        """Amplify signal based on channel type"""
        if channel_type in ['bipecg', 'ecg3', 'ecg']:
            return signal * 5  # Enhanced ECG amplification
        elif channel_type in ['rv', 'ra', 'lv', 'shock']:
            return signal * 10  # Standard lead amplification
        else:
            return signal

    def rectifier(self, signal):
        """Full-wave rectification"""
        return np.abs(signal)

    def sq_rectifier(self, signal):
        """Square rectifier with gain"""
        rectified = self.rectifier(signal)
        return (rectified ** 2) * 15  # Enhanced gain for ECG channels

    def derivatives(self, signal):
        """Calculate signal derivative for edge detection"""
        return np.gradient(signal)

    def zero_crossings(self, gradient):
        """Detect zero crossings in gradient signal"""
        return np.where(np.diff(np.sign(gradient)))[0]

    def sophisticated_peak_detection(self, signal, channel_type='bipecg'):
        """
        Sophisticated peak detection using zero crossings and adaptive thresholding
        """
        # Step 1: Square rectification
        sq_rect_signal = self.sq_rectifier(signal)

        # Step 2: Calculate derivatives
        gradient = self.derivatives(sq_rect_signal)

        # Step 3: Find zero crossings
        zero_cross_indices = self.zero_crossings(gradient)

        # Step 4: Peak detection at zero crossings
        peaks = []
        post_vs_blanking_samples = int(120 * self.target_sampling_rate / 1000)  # 120ms in samples

        prev_crossing = None
        for crossing in zero_cross_indices:
            # Ensure minimum refractory period between detections
            if prev_crossing is None or crossing - prev_crossing > post_vs_blanking_samples:
                # Find peak in window around zero crossing
                window_start = max(0, crossing - 100)
                window_end = min(len(sq_rect_signal), crossing + 100)

                if window_end > window_start:
                    local_max_idx = np.argmax(sq_rect_signal[window_start:window_end])
                    peak_index = window_start + local_max_idx
                    peaks.append(peak_index)
                    prev_crossing = crossing

        return np.array(peaks)

    def process_signal_complete_pipeline(self, signal, channel_type='bipecg'):
        """
        Complete Medtronic signal processing pipeline
        Returns processed signal and detected peaks
        """
        print(f"Processing signal through Medtronic pipeline...")

        # Step 1: Resample signal to target frequency
        resampled = self.resample_data(signal, self.original_sampling_rate, self.target_sampling_rate)
        print(f"Resampled from {self.original_sampling_rate}Hz to {self.target_sampling_rate}Hz")

        # Step 2: Apply ECG bandpass filter
        filtered = self.ecg_filter(resampled)
        print("Applied ECG bandpass filter (0.5-25 Hz)")

        # Step 3: Amplify signal
        amplified = self.amplifier(filtered, channel_type)

        # Step 4: Detect peaks
        peaks = self.sophisticated_peak_detection(amplified, channel_type)
        print(f"Detected {len(peaks)} peaks")

        # Store processed data
        self.processed_signals[channel_type] = {
            'raw': signal,
            'resampled': resampled,
            'filtered': filtered,
            'amplified': amplified,
            'peaks': peaks
        }

        return amplified, peaks


class MedtronicICDParameters:
    """Medtronic ICD parameter configuration"""

    def __init__(self):
        # Zone definitions (cycle lengths in ms)
        self.vf_tcl = 320  # VF zone (188 bpm)
        self.fvt_tcl = 400  # FVT zone (150 bpm)
        self.vt_tcl = 500  # VT zone (120 bpm)
        self.monitor_tcl = 600  # Monitor zone (100 bpm)

        # Number of intervals to detect (NID)
        self.vf_min_nid = 30  # Minimum VF NID
        self.vf_max_nid = 40  # Maximum VF NID
        self.vt_nid = 18  # VT NID

        # Detection parameters
        self.onset_pct = 0.81  # Onset percentage (nominal setting)
        self.stability = 40  # Stability criterion (ms)

        # SVT discrimination
        self.svt_limit = 260  # SVT limit (ms) - 150 bpm
        self.wavelet_threshold = 70  # Match percentage

        # Sensing thresholds
        self.bipecg_sensing_threshold = 0.2  # BipECG sensing threshold (mV)
        self.post_vs_blanking = 120  # Post V-sense blanking (ms)

        # Noise and T-wave detection
        self.twave_detection = True
        self.rv_noise_detection = True
        self.rv_noise_timeout = 0.75  # minutes

        # Episode duration parameters
        self.vt_episode_min_duration = 10.0  # 10 seconds minimum duration

        # Discriminator enables
        self.onset_enabled = True
        self.stability_enabled = True
        self.wavelet_enabled = True

    def set_vf_zone(self, tcl_ms, nid_string="30/40"):
        """Set VF zone parameters"""
        self.vf_tcl = tcl_ms
        nid_parts = nid_string.split("/")
        self.vf_min_nid = int(nid_parts[0])
        self.vf_max_nid = int(nid_parts[1])

    def set_vt_zone(self, tcl_ms, nid=18):
        """Set VT zone parameters"""
        self.vt_tcl = tcl_ms
        self.vt_nid = nid


class MedtronicICDAnalyser:
    """
    Medtronic ICD Algorithm Analyser with proven sensing
    """

    def __init__(self):
        self.icd_parameters = MedtronicICDParameters()
        self.sensing_algorithm = MedtronicSensingAlgorithm()
        self.baseline_templates = {}
        self.reset_memory()

    def reset_memory(self):
        """Reset ICD memory buffers"""
        self.icd_memory = {
            'r_peaks': deque(maxlen=40),
            'rr_intervals': deque(maxlen=40),
            'rhythm_labels': deque(maxlen=40),
            'onset_intervals': deque(maxlen=8),
            'stability_intervals': deque(maxlen=4),
            'wavelet_beats': deque(maxlen=8),
            'last_r_peak': 0,
            'active_tachycardia': False,
            'detection_time': None,
            'detection_rate': None,
            'therapy_indicated': False,
            'episode_start_time': None,
            'episode_duration': 0,
            'episode_intervals': [],
            'previous_episodes': [],
            'noise_detected': False,
            'noise_timeout_start': None,
            'twave_oversense_count': 0
        }

    def calculate_rr_intervals(self, r_peaks, sampling_rate):
        """Calculate RR intervals in milliseconds"""
        if len(r_peaks) < 2:
            return []

        rr_intervals = np.diff(r_peaks) * 1000 / sampling_rate  # Convert to ms
        return rr_intervals

    def classify_rhythm(self, rr_interval):
        """
        Classify rhythm based on RR interval
        Returns rhythm classification: 'FS' (VF), 'FT' (FVT), 'TS' (VT), 'NS' (normal)
        """
        if rr_interval <= self.icd_parameters.vf_tcl:
            zone = 'FS'  # VF zone
            rate = 60000 / rr_interval if rr_interval > 0 else 0
            print(f"    Zone Classification: VF (RR={rr_interval:.0f}ms, Rate={rate:.0f}bpm)")
        elif rr_interval <= self.icd_parameters.fvt_tcl:
            zone = 'FT'  # FVT zone
            rate = 60000 / rr_interval if rr_interval > 0 else 0
            print(f"    Zone Classification: FVT (RR={rr_interval:.0f}ms, Rate={rate:.0f}bpm)")
        elif rr_interval <= self.icd_parameters.vt_tcl:
            zone = 'TS'  # VT zone
            rate = 60000 / rr_interval if rr_interval > 0 else 0
            print(f"    Zone Classification: VT (RR={rr_interval:.0f}ms, Rate={rate:.0f}bpm)")
        elif rr_interval <= self.icd_parameters.monitor_tcl:
            zone = 'TM'  # Monitor zone
            rate = 60000 / rr_interval if rr_interval > 0 else 0
            print(f"    Zone Classification: Monitor (RR={rr_interval:.0f}ms, Rate={rate:.0f}bpm)")
        else:
            zone = 'NS'  # Normal sinus
            rate = 60000 / rr_interval if rr_interval > 0 else 0
            print(f"    Zone Classification: Normal (RR={rr_interval:.0f}ms, Rate={rate:.0f}bpm)")

        return zone

    def detect_twave_oversensing(self, rr_intervals):
        """Detect T-wave oversensing based on characteristic patterns"""
        if not self.icd_parameters.twave_detection or len(rr_intervals) < 4:
            return False

        # T-wave oversensing patterns:
        # 1. Very short intervals (< 200ms) followed by compensatory pauses
        # 2. Alternating short-long pattern

        short_intervals = np.array(rr_intervals) < 200
        very_short_count = np.sum(short_intervals)

        # If more than 25% of intervals are very short, likely T-wave oversensing
        if very_short_count > len(rr_intervals) * 0.25:
            return True

        # Check for alternating pattern
        if len(rr_intervals) >= 4:
            # Look for short-long-short-long pattern
            pattern_detected = False
            for i in range(len(rr_intervals) - 3):
                if (rr_intervals[i] < 250 and rr_intervals[i + 1] > 400 and
                        rr_intervals[i + 2] < 250 and rr_intervals[i + 3] > 400):
                    pattern_detected = True
                    break

            if pattern_detected:
                return True

        return False

    def detect_rv_noise(self, rr_intervals):
        """Detect RV lead noise based on RR patterns"""
        if not self.icd_parameters.rv_noise_detection:
            return False

        # Noise indicators: very rapid irregular intervals
        if len(rr_intervals) >= 8:
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)

            # If many very short and very variable intervals
            if mean_rr < 150 and std_rr > 50:
                return True

        return False

    def check_onset_criterion(self):
        """Medtronic Onset Criterion (sudden onset)"""
        if len(self.icd_memory['onset_intervals']) < 8:
            return None

        intervals = list(self.icd_memory['onset_intervals'])
        a = intervals[:4]  # Most recent 4 intervals
        b = intervals[4:]  # Previous 4 intervals

        mean_a = np.mean(np.array(a))
        mean_b = np.mean(np.array(b))

        # Medtronic onset formula: mean_b * onset_pct > mean_a
        onset_trigger = (mean_b * self.icd_parameters.onset_pct) > mean_a

        if onset_trigger and (len(a) + len(b)) == 8:
            print(f"\033[31mVT Onset Criteria Met\033[0m")
            print(f"Recent mean: {mean_a:.1f}ms, Previous mean: {mean_b:.1f}ms")
            print(f"Threshold: {mean_b * self.icd_parameters.onset_pct:.1f}ms")
            return True

        return False

    def check_stability_criterion(self):
        """Medtronic Stability Criterion"""
        if len(self.icd_memory['stability_intervals']) < 4:
            return None

        stability_intervals = list(self.icd_memory['stability_intervals'])
        last_cl = stability_intervals[-1]  # Last cycle length
        previous_cls = stability_intervals[:-1]  # Previous 3 cycle lengths

        # Check if last CL is within stability range of each previous CL
        stable = True
        for cl in previous_cls:
            stability_difference = abs(last_cl - cl)
            if stability_difference > self.icd_parameters.stability:
                stable = False
                print(f"Stability failed: {stability_difference:.1f}ms > {self.icd_parameters.stability}ms")
                break

        if stable:
            print(f"\033[31mStability Criterion Met\033[0m")
            print(f"Last CL: {last_cl:.1f}ms within ±{self.icd_parameters.stability}ms of previous CLs")

        return stable

    def extract_wavelet_features(self, signal, r_peak, window_size=300):
        """Extract wavelet features for morphology analysis"""
        if not ADVANCED_FEATURES:
            return np.array([])

        try:
            # Extract beat around R-peak
            start_idx = max(0, r_peak - window_size // 2)
            end_idx = min(len(signal), r_peak + window_size // 2)
            beat = signal[start_idx:end_idx]

            if len(beat) == 0:
                return np.array([])

            # Pad if necessary to maintain consistent length
            if len(beat) < window_size:
                beat = np.pad(beat, (0, window_size - len(beat)), 'constant', constant_values=0)
            elif len(beat) > window_size:
                beat = beat[:window_size]

            # Wavelet decomposition using Haar wavelet
            try:
                coeffs = pywt.wavedec(beat, 'haar', level=4)
                flattened_coeffs = np.concatenate(coeffs)
                return flattened_coeffs
            except Exception as e:
                return np.array([])

        except Exception as e:
            return np.array([])

    def compare_wavelet_templates(self, current_coeffs, template_coeffs):
        """Medtronic Wavelet Morphology Matching"""
        if template_coeffs is None or len(template_coeffs) == 0:
            return 0

        try:
            # Medtronic uses coefficient ranking method
            template_coeff_dict = {}
            unknown_coeff_dict = {}

            # Sort and rank coefficients
            template_sorted_indices = np.argsort(template_coeffs)[::-1]  # Descending order
            unknown_sorted_indices = np.argsort(current_coeffs)[::-1]

            # Create rank mappings
            for rank, idx in enumerate(template_sorted_indices[:10], 1):  # Top 10 coefficients
                template_coeff_dict[idx] = rank

            for rank, idx in enumerate(unknown_sorted_indices[:10], 1):
                unknown_coeff_dict[idx] = rank

            # Medtronic simple rank scoring system
            template_simp_rank_dict = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}

            # Calculate match score based on rank correlation
            simp_rank_list = []
            for coeff_idx in template_coeff_dict:
                if coeff_idx in unknown_coeff_dict:
                    template_rank = template_coeff_dict[coeff_idx]
                    if template_rank in template_simp_rank_dict:
                        simp_rank_list.append(template_simp_rank_dict[template_rank])

            # Calculate percentage match
            if simp_rank_list:
                sum_ranks = sum(simp_rank_list)
                max_possible_score = 55  # Sum of 1+2+3+...+10
                prop_sum_ranks = sum_ranks / max_possible_score
                pct_match_score = int(prop_sum_ranks * 100)
                match_percentage = min(pct_match_score, 100)
            else:
                match_percentage = 0

            return match_percentage

        except Exception as e:
            # Fallback to simple correlation
            correlation = np.corrcoef(current_coeffs, template_coeffs)[0, 1]
            if np.isnan(correlation):
                return 0
            else:
                return max(0, correlation * 100)

    def check_svt_discriminators(self, rhythm_labels, rr_intervals):
        """Comprehensive Medtronic SVT Discrimination"""
        svt_detected = False
        discriminator_reasons = []

        if not rr_intervals or len(rr_intervals) < 4:
            return False

        # Filter out zero or invalid intervals
        valid_intervals = [rr for rr in rr_intervals if rr > 0]
        if not valid_intervals:
            return False

        # Calculate current heart rate safely
        median_rr = np.median(valid_intervals[-12:]) if len(valid_intervals) >= 12 else np.median(valid_intervals)

        if median_rr <= 0:
            return False

        try:
            current_rate = 60000 / median_rr
            svt_limit_rate = 60000 / self.icd_parameters.svt_limit
        except ZeroDivisionError:
            return False

        print(f"SVT Analysis - Current rate: {current_rate:.0f} bpm, SVT limit: {svt_limit_rate:.0f} bpm")

        # 1. SVT V Limit Check
        if median_rr > self.icd_parameters.svt_limit:
            svt_detected = True
            discriminator_reasons.append(f"Rate below SVT V-Limit ({svt_limit_rate:.0f} bpm)")

        # 2. Atrial Fibrillation/Flutter Logic (simplified)
        if len(valid_intervals) >= 8:
            rr_variability = np.std(valid_intervals[-8:])
            if rr_variability > 50:  # High RR variability suggests AF
                svt_detected = True
                discriminator_reasons.append(f"High RR variability ({rr_variability:.1f}ms) suggests AF")

        # 3. Sinus Tachycardia Detection
        if len(valid_intervals) >= 8:
            early_intervals = valid_intervals[-8:-4]
            recent_intervals = valid_intervals[-4:]
            early_mean = np.mean(early_intervals)
            recent_mean = np.mean(recent_intervals)

            # Gradual rate increase suggests sinus tach
            if early_mean > recent_mean and (early_mean - recent_mean) < 100:  # Gradual change
                svt_detected = True
                discriminator_reasons.append("Gradual onset suggests sinus tachycardia")

        if svt_detected:
            print(f"\033[33mSVT Discriminator ACTIVE: {'; '.join(discriminator_reasons)}\033[0m")

        return svt_detected

    def check_medtronic_wavelet_morphology(self, processed_signal, r_peaks, template_key):
        """Medtronic Wavelet Morphology Discrimination"""
        if len(r_peaks) < 3:
            print("Insufficient R-peaks for morphology analysis")
            return False, 0

        # Get baseline template using the provided key
        baseline_template = self.baseline_templates.get(template_key, None)

        if baseline_template is None:
            print(f"❌ No baseline template available for '{template_key}'")
            return False, 0

        print(f"✅ Using baseline template for '{template_key}'")

        # Extract recent beats for comparison
        match_scores = []
        mismatch_count = 0

        # Check last 3 beats (typical Medtronic approach)
        recent_peaks = r_peaks[-3:] if len(r_peaks) >= 3 else r_peaks

        print(f"Comparing {len(recent_peaks)} recent beats with baseline template")

        for idx, r_peak in enumerate(recent_peaks):
            try:
                current_coeffs = self.extract_wavelet_features(processed_signal, r_peak)
                if len(current_coeffs) == 0:
                    print(f"Beat {idx + 1}: No coefficients extracted - skipping")
                    continue

                match_score = self.compare_wavelet_templates(current_coeffs, baseline_template)
                match_scores.append(match_score)

                print(f"Beat {idx + 1}: Match score = {match_score:.1f}%")

                if match_score < self.icd_parameters.wavelet_threshold:
                    mismatch_count += 1
                    print(f"  -> MISMATCH (< {self.icd_parameters.wavelet_threshold}% threshold)")
                else:
                    print(f"  -> MATCH (≥ {self.icd_parameters.wavelet_threshold}% threshold)")

            except Exception as e:
                print(f"Error processing beat {idx + 1}: {e}")
                continue

        if not match_scores:
            print("❌ No valid beats for morphology comparison")
            return False, 0

        avg_match_score = np.mean(match_scores)

        # Medtronic logic: if majority of recent beats mismatch, morphology is different
        morphology_different = mismatch_count >= (len(match_scores) // 2 + 1)

        print(f"Morphology Analysis Results:")
        print(f"  - Valid beats analyzed: {len(match_scores)}")
        print(f"  - Mismatches: {mismatch_count}")
        print(f"  - Average match score: {avg_match_score:.1f}%")
        print(f"  - Morphology different: {morphology_different}")

        if morphology_different:
            print(
                f"\033[31mMorphology Criterion Met\033[0m - {mismatch_count}/{len(match_scores)} beats below {self.icd_parameters.wavelet_threshold}% threshold")
        else:
            print(f"\033[33mMorphology suggests SVT\033[0m - Average match: {avg_match_score:.1f}%")

        return morphology_different, avg_match_score

    def determine_lead_from_csv_row(self, row, daq_data):
        """Determine which lead to use based on CSV row and available data"""
        # Priority: RV Bipolar > RV Shock > LV Lead > BipECG
        lead_priority = [
            ('RVbip', row['RVbip'], 'RV Bipolar'),
            ('RVshock', row['RVshock'], 'RV Shock'),
            ('LVlead', row['LVlead'], 'LV Lead'),
            ('BipECG', row['BipECG'], 'BipECG')
        ]

        for csv_col, channel_names, lead_desc in lead_priority:
            csv_value = str(row.get(csv_col, '')).lower()

          # Try each channel name
            if hasattr(daq_data, csv_value) and channel_names in daq_data.sources:
                print(f"✅ Using {lead_desc} ({channel_names})")
                return channel_names, f'{lead_desc} ({channel_names})'

        print("❌ No suitable lead found")
        return None, None

    def analyse_detection_sequence(self, processed_signal, r_peaks, sampling_rate, template_key=None):
        """Analyse the complete detection sequence"""
        results = {
            'vf_detected': False,
            'vt_detected': False,
            'detection_time': None,
            'detection_rate': None,
            'rate_at_detection': None,
            'onset_met': False,
            'stability_met': False,
            'wavelet_match': 0,
            'morphology_different': False,
            'svt_discriminator_active': False,
            'therapy_indicated': False,
            'detection_intervals': '0/0',
            'withhold_reason': None,
            'vt_evidence': None,
            'noise_detected': False,
            'twave_oversensing': False,
            'episode_duration': 0,
            'total_episodes': 0
        }

        if len(r_peaks) < 2:
            print("Insufficient R-peaks for analysis")
            return results

        rr_intervals = self.calculate_rr_intervals(r_peaks, sampling_rate)

        # Check for T-wave oversensing
        twave_detected = self.detect_twave_oversensing(rr_intervals)
        if twave_detected:
            print("⚠️ T-wave oversensing detected")
            results['twave_oversensing'] = True

        # Check for noise
        noise_detected = self.detect_rv_noise(rr_intervals)
        if noise_detected:
            print("⚠️ RV noise detected")
            results['noise_detected'] = True

        vf_count = 0
        vt_count = 0
        rhythm_sequence = []

        print(f"Analysing {len(rr_intervals)} RR intervals...")

        # Process each RR interval
        for i, rr_interval in enumerate(rr_intervals):
            # Handle zero or invalid RR intervals
            if rr_interval <= 0:
                print(f"Beat {i + 1}: Invalid RR interval ({rr_interval}ms) - skipping")
                continue

            rhythm = self.classify_rhythm(rr_interval)
            rhythm_sequence.append(rhythm)
            self.icd_memory['rhythm_labels'].append(rhythm)
            self.icd_memory['rr_intervals'].append(rr_interval)
            self.icd_memory['onset_intervals'].append(rr_interval)
            self.icd_memory['stability_intervals'].append(rr_interval)

            # Safe rate calculation
            try:
                current_rate = 60000 / rr_interval
                print(f"Beat {i + 1}: RR={rr_interval:.0f}ms, Rate={current_rate:.0f}bpm, Zone={rhythm}")
            except ZeroDivisionError:
                print(f"Beat {i + 1}: RR={rr_interval:.0f}ms, Rate=Invalid, Zone={rhythm}")
                continue

            # Track episode timing
            if rhythm in ['FS', 'FT', 'TS']:  # Tachycardia zones
                if self.icd_memory['episode_start_time'] is None:
                    self.icd_memory['episode_start_time'] = (r_peaks[i] / sampling_rate) if i < len(r_peaks) else 0
                self.icd_memory['episode_intervals'].append(rr_interval)
            else:
                # End of tachycardia episode
                if self.icd_memory['episode_start_time'] is not None:
                    current_time = (r_peaks[i] / sampling_rate) if i < len(r_peaks) else 0
                    episode_duration = current_time - self.icd_memory['episode_start_time']

                    self.icd_memory['previous_episodes'].append({
                        'duration': episode_duration,
                        'intervals': list(self.icd_memory['episode_intervals'])
                    })

                    # Check if episode duration < 10 seconds - treat as same episode
                    if episode_duration < self.icd_parameters.vt_episode_min_duration:
                        print(f"Episode duration {episode_duration:.1f}s < 10s - treating as same episode")
                        # Don't reset counters - continue episode
                    else:
                        # Reset episode tracking
                        self.icd_memory['episode_start_time'] = None
                        self.icd_memory['episode_intervals'] = []
                        vf_count = 0
                        vt_count = 0

            # Count consecutive VF/VT intervals
            if rhythm == 'FS':  # VF
                vf_count += 1
                vt_count = 0
            elif rhythm in ['FT', 'TS']:  # VT/FVT
                vt_count += 1
                vf_count = 0
            else:
                # Only reset if not in same episode mode
                if (self.icd_memory['episode_start_time'] is None):
                    vf_count = 0
                    vt_count = 0

            # Check VF detection
            if vf_count >= self.icd_parameters.vf_min_nid and not results['vf_detected']:
                results['vf_detected'] = True
                results['detection_time'] = (r_peaks[i] / sampling_rate) if i < len(r_peaks) else 0

                # Safe rate calculation
                try:
                    results['detection_rate'] = 60000 / rr_interval if rr_interval > 0 else 'Invalid'
                    results['rate_at_detection'] = results['detection_rate']
                except ZeroDivisionError:
                    results['detection_rate'] = 'Invalid'
                    results['rate_at_detection'] = 'Invalid'

                results['detection_intervals'] = f"{vf_count}/{self.icd_parameters.vf_max_nid}"
                results['therapy_indicated'] = True  # VF always gets therapy

                # Calculate episode duration
                if self.icd_memory['episode_start_time'] is not None:
                    episode_duration = results['detection_time'] - self.icd_memory['episode_start_time']
                    results['episode_duration'] = episode_duration

                rate_display = f"{results['detection_rate']:.0f}" if isinstance(results['detection_rate'],
                                                                                (int, float)) else results[
                    'detection_rate']
                print(
                    f"\033[31m*** VF DETECTED *** at {results['detection_time']:.2f}s - Rate: {rate_display} bpm\033[0m")
                print(f"\033[31m*** THERAPY INDICATED *** (VF bypasses discriminators)\033[0m")

                # Run discriminators for analysis
                onset_met = self.check_onset_criterion()
                results['onset_met'] = onset_met if onset_met is not None else False

                stability_met = self.check_stability_criterion()
                results['stability_met'] = stability_met if stability_met is not None else False

                morphology_different, wavelet_score = self.check_medtronic_wavelet_morphology(
                    processed_signal, r_peaks[:i + 2], template_key
                )
                results['wavelet_match'] = wavelet_score
                results['morphology_different'] = morphology_different

                svt_discriminator_active = self.check_svt_discriminators(
                    list(self.icd_memory['rhythm_labels']),
                    list(self.icd_memory['rr_intervals'])
                )
                results['svt_discriminator_active'] = svt_discriminator_active

                break

            # Check VT detection
            elif vt_count >= self.icd_parameters.vt_nid and not results['vt_detected']:
                results['vt_detected'] = True
                results['detection_time'] = (r_peaks[i] / sampling_rate) if i < len(r_peaks) else 0

                # Safe rate calculation
                try:
                    results['detection_rate'] = 60000 / rr_interval if rr_interval > 0 else 'Invalid'
                    results['rate_at_detection'] = results['detection_rate']
                except ZeroDivisionError:
                    results['detection_rate'] = 'Invalid'
                    results['rate_at_detection'] = 'Invalid'

                results['detection_intervals'] = f"{vt_count}/{self.icd_parameters.vt_nid}"

                # Calculate episode duration
                if self.icd_memory['episode_start_time'] is not None:
                    episode_duration = results['detection_time'] - self.icd_memory['episode_start_time']
                    results['episode_duration'] = episode_duration

                print(f"\033[31m*** VT NID MET *** - Running Medtronic Discriminators...\033[0m")

                # Medtronic VT Discriminator Sequence
                onset_met = self.check_onset_criterion()
                results['onset_met'] = onset_met if onset_met is not None else False

                stability_met = self.check_stability_criterion()
                results['stability_met'] = stability_met if stability_met is not None else False

                morphology_different, wavelet_score = self.check_medtronic_wavelet_morphology(
                    processed_signal, r_peaks[:i + 2], template_key
                )
                results['wavelet_match'] = wavelet_score
                results['morphology_different'] = morphology_different

                svt_discriminator_active = self.check_svt_discriminators(
                    list(self.icd_memory['rhythm_labels']),
                    list(self.icd_memory['rr_intervals'])
                )
                results['svt_discriminator_active'] = svt_discriminator_active

                # Medtronic Therapy Decision Logic
                print(f"\n--- MEDTRONIC DISCRIMINATOR RESULTS ---")
                print(f"Onset Met: {results['onset_met']}")
                print(f"Stability Met: {results['stability_met']}")
                print(f"Morphology Different: {morphology_different} (Score: {wavelet_score:.1f}%)")
                print(f"SVT Discriminator: {svt_discriminator_active}")

                if svt_discriminator_active:
                    results['therapy_indicated'] = False
                    results['withhold_reason'] = "SVT Discriminator Active"
                    print(f"\033[33m*** THERAPY WITHHELD *** - SVT Discriminator Active\033[0m")

                elif not morphology_different and wavelet_score > 0:
                    results['therapy_indicated'] = False
                    results['withhold_reason'] = f"Morphology matches baseline ({wavelet_score:.1f}% similarity)"
                    print(f"\033[33m*** THERAPY WITHHELD *** - Morphology suggests SVT\033[0m")

                else:
                    # VT confirmed by discriminators
                    vt_evidence = []
                    if results['onset_met']:
                        vt_evidence.append("Sudden Onset")
                    if results['stability_met']:
                        vt_evidence.append("Stable Rate")
                    if morphology_different:
                        vt_evidence.append("Abnormal Morphology")

                    if vt_evidence:
                        results['therapy_indicated'] = True
                        results['vt_evidence'] = ", ".join(vt_evidence)
                        print(f"\033[31m*** VT CONFIRMED *** - Evidence: {results['vt_evidence']}\033[0m")
                        print(f"\033[31m*** THERAPY INDICATED ***\033[0m")
                    else:
                        results['therapy_indicated'] = False
                        results['withhold_reason'] = "Insufficient VT evidence from discriminators"
                        print(f"\033[33m*** THERAPY WITHHELD *** - Insufficient discriminator evidence\033[0m")

                break

        # Add episode information
        results['total_episodes'] = len(self.icd_memory['previous_episodes'])

        return results

    def process_baseline_period(self, signal, r_peaks, patient_id, label):
        """Process baseline period to create wavelet template"""
        if len(r_peaks) < 4:
            print(f"Insufficient beats in baseline for patient {patient_id} (need 4, got {len(r_peaks)})")
            return

        template_key = f"{patient_id}_{label}"
        print(f"Creating baseline template with key: '{template_key}' using {len(r_peaks)} beats")

        # Extract multiple baseline beats for template creation
        baseline_beats = []
        valid_beats = 0

        for i, r_peak in enumerate(r_peaks[:8]):  # Use up to 8 beats
            try:
                beat_coeffs = self.extract_wavelet_features(signal, r_peak)
                if len(beat_coeffs) > 0:
                    baseline_beats.append(beat_coeffs)
                    valid_beats += 1
                    print(f"  Baseline beat {i + 1}: {len(beat_coeffs)} coefficients extracted")
                else:
                    print(f"  Baseline beat {i + 1}: Failed to extract coefficients")
            except Exception as e:
                print(f"  Baseline beat {i + 1}: Error extracting features - {e}")
                continue

        # Create average template
        if baseline_beats and valid_beats >= 3:
            template = np.mean(baseline_beats, axis=0)
            self.baseline_templates[template_key] = template
            print(f"✅ Created baseline template for '{template_key}' with {valid_beats} valid beats")
        else:
            print(f"❌ Failed to create baseline template - insufficient valid beats ({valid_beats}/8)")


def process_patient_file(csv_row, data_dir, analyser):
    """Process a single patient file and run ICD algorithm"""

    patient = csv_row['Patient']
    file_name = csv_row['File']
    period = csv_row['Period']
    label = csv_row['Label']

    # Construct file path
    zip_path = os.path.join(data_dir, patient, 'Haem', file_name)

    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        return None

    print(f"Processing: {patient} - {period} - {label}")

    # Load DAQ data
    try:
        daq_data = DAQ_File("", zip_path)
    except Exception as e:
        print(f"Error loading {zip_path}: {e}")
        return None

    # Find suitable lead
    lead_attr, lead_name = analyser.determine_lead_from_csv_row(csv_row, daq_data)
    if lead_attr is None:
        print(f"No suitable lead found for {file_name}")
        return None

    # Get signal data
    signal_data = getattr(daq_data, lead_attr)
    sampling_rate = daq_data.sampling_rate

    # Extract data for the specified time period
    start_sample = int(csv_row['Begin'])
    end_sample = int(csv_row['End'])

    if end_sample > len(signal_data):
        print(f"End sample {end_sample} exceeds signal length {len(signal_data)}")
        end_sample = len(signal_data)

    period_signal = signal_data[start_sample:end_sample]

    # Process through sensing algorithm
    try:
        processed_signal, r_peaks = analyser.sensing_algorithm.process_signal_complete_pipeline(
            period_signal, lead_name
        )
    except Exception as e:
        print(f"Error processing signal: {e}")
        return None

    if len(r_peaks) < 2:
        print(f"Insufficient R-peaks detected in {file_name}")
        return None

    return {
        'patient': patient,
        'file_name': file_name,
        'period': period,
        'label': label,
        'lead_used': lead_name,
        'processed_signal': processed_signal,
        'r_peaks': r_peaks,
        'sampling_rate': analyser.sensing_algorithm.target_sampling_rate
    }


def main():
    """Main function to process the CSV database and run ICD algorithm"""

    # Configuration
    csv_file = "vt_csv_file2.csv"
    data_dir = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Research/02 Current Projects/02 AAD/DataGroups/Unnecessary/all_vt"
    output_dir = "icd_results"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV database
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Initialise analyser
    analyser = MedtronicICDAnalyser()

    # Results storage
    analysis_results = []

    # Group by patient and label for continuous traces
    grouped = df.groupby(['Patient', 'Label'])

    for (patient, label), group in grouped:
        print(f"\n{'=' * 50}")
        print(f"Processing Patient: {patient}, Label: {label}")
        print(f"{'=' * 50}")

        # Sort by period to process Baseline first
        group_sorted = group.sort_values('Period')

        # Reset analyser memory for this patient/label
        analyser.reset_memory()

        # Process baseline periods first
        baseline_processed = False
        for idx, row in group_sorted.iterrows():
            if row['Period'] == 'Baseline':
                print(f"Processing Baseline for Patient {row['Patient']}, File: {row['File']}")

                result = process_patient_file(row, data_dir, analyser)
                if result is not None:
                    # Process baseline template
                    analyser.process_baseline_period(
                        result['processed_signal'],
                        result['r_peaks'],
                        result['patient'],
                        result['label']
                    )
                    baseline_processed = True

        # Process non-baseline periods and combine into continuous trace
        continuous_data = {'signal': [], 'r_peaks': [], 'sampling_rate': 512}

        for idx, row in group_sorted.iterrows():
            if row['Period'] != 'Baseline' and row['Period'] != 'Recovery':
                print(f"\nProcessing {row['Period']} for Patient {row['Patient']}, File: {row['File']}")

                result = process_patient_file(row, data_dir, analyser)
                if result is not None:
                    # Append to continuous data
                    signal_start_idx = len(continuous_data['signal'])
                    continuous_data['signal'].extend(result['processed_signal'])

                    # Adjust R-peak indices for continuous signal
                    r_peaks_adjusted = result['r_peaks'] + signal_start_idx
                    continuous_data['r_peaks'].extend(r_peaks_adjusted)

                    lead_name = result['lead_used']  # Store for later use

        # Check if we have enough data for analysis
        if not continuous_data['signal'] or not continuous_data['r_peaks']:
            print(f"No valid data for {patient}/{label}")
            continue

        # Check episode duration and extend if necessary (treating as same episode)
        episode_duration_s = len(continuous_data['signal']) / continuous_data['sampling_rate']
        print(f"Episode duration: {episode_duration_s:.1f}s")

        if episode_duration_s < 10.0:
            print(f"⚠️ Episode too short ({episode_duration_s:.1f}s < 10s) - Extending to minimum 15s for analysis")

            # Calculate repetitions needed for 15s minimum
            target_duration = 15.0  # seconds
            target_samples = int(target_duration * continuous_data['sampling_rate'])
            repetitions_needed = int(target_samples / len(continuous_data['signal'])) + 1

            print(f"📈 Repeating episode {repetitions_needed} times (same episode logic)")

            # Store original data
            original_signal = continuous_data['signal'].copy()
            original_r_peaks = continuous_data['r_peaks'].copy()

            # Extend signal by repeating
            extended_signal = []
            extended_r_peaks = []

            for rep in range(repetitions_needed):
                # Add signal
                extended_signal.extend(original_signal)

                # Add R-peaks with offset
                offset = rep * len(original_signal)
                offset_r_peaks = [peak + offset for peak in original_r_peaks]
                extended_r_peaks.extend(offset_r_peaks)

            # Trim to exact target duration
            target_samples = int(target_duration * continuous_data['sampling_rate'])
            extended_signal = extended_signal[:target_samples]
            extended_r_peaks = [peak for peak in extended_r_peaks if peak < target_samples]

            # Update continuous data
            continuous_data['signal'] = extended_signal
            continuous_data['r_peaks'] = extended_r_peaks

            final_duration = len(continuous_data['signal']) / continuous_data['sampling_rate']
            print(f"✅ Extended episode to {final_duration:.1f}s using same episode logic")

        # Analyse the complete continuous trace
        print(f"\n*** ANALYSING COMPLETE TRACE FOR {patient}/{label} ***")
        print(f"Total signal length: {len(continuous_data['signal'])} samples")
        print(f"Total R-peaks detected: {len(continuous_data['r_peaks'])}")

        # Reset memory again for clean analysis
        analyser.reset_memory()

        # Create template key
        template_key = f"{patient}_{label}"
        print(f"Looking for baseline template with key: '{template_key}'")

        # Analyse detection
        try:
            results = analyser.analyse_detection_sequence(
                np.array(continuous_data['signal']),
                np.array(continuous_data['r_peaks']),
                continuous_data['sampling_rate'],
                template_key
            )

            # Add metadata
            results['Patient'] = patient
            results['Label'] = label
            results['Files_Processed'] = len(group_sorted[group_sorted['Period'] != 'Baseline'])
            results['Lead_Used'] = lead_name if 'lead_name' in locals() else 'Unknown'
            results['Baseline_Available'] = baseline_processed
            results['Original_Episode_Duration_s'] = episode_duration_s

            # Handle inf/nan values
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    if np.isinf(value):
                        results[key] = 'No Detection'
                    elif np.isnan(value):
                        results[key] = 'N/A'

            analysis_results.append(results)

            # Print summary
            print(f"\n*** ANALYSIS SUMMARY for {patient}/{label} ***")
            print(f"Lead Used: {results['Lead_Used']}")
            print(f"VF Detected: {results['vf_detected']}")
            print(f"VT Detected: {results['vt_detected']}")
            if results['detection_time']:
                print(f"Detection Time: {results['detection_time']:.2f}s")
                if isinstance(results['rate_at_detection'], (int, float)):
                    print(f"Rate at Detection: {results['rate_at_detection']:.0f} bpm")
                print(f"Detection Intervals: {results['detection_intervals']}")
            print(f"Therapy Indicated: {results['therapy_indicated']}")
            print(f"Episode Duration: {results.get('episode_duration', 0):.1f}s")
            print(f"Noise Detected: {results['noise_detected']}")
            print(f"T-wave Oversensing: {results['twave_oversensing']}")

        except Exception as e:
            print(f"❌ Error analysing trace for {patient}/{label}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if analysis_results:
        results_df = pd.DataFrame(analysis_results)

        # Define column order
        column_order = [
            'Patient', 'Label', 'Lead_Used', 'Files_Processed', 'Baseline_Available',
            'vf_detected', 'vt_detected', 'detection_time', 'detection_rate', 'rate_at_detection',
            'detection_intervals', 'therapy_indicated', 'onset_met', 'stability_met',
            'wavelet_match', 'morphology_different', 'svt_discriminator_active',
            'withhold_reason', 'vt_evidence', 'noise_detected', 'twave_oversensing',
            'episode_duration', 'total_episodes', 'Original_Episode_Duration_s'
        ]

        # Reorder columns
        available_columns = [col for col in column_order if col in results_df.columns]
        extra_columns = [col for col in results_df.columns if col not in column_order]
        final_column_order = available_columns + extra_columns

        results_df = results_df[final_column_order]

        # Replace inf/-inf values
        results_df = results_df.replace([np.inf, -np.inf], 'No Detection')
        results_df = results_df.replace([np.nan], 'N/A')

        output_file = os.path.join(output_dir, "medtronic_icd_analysis_results.csv")
        results_df.to_csv(output_file, index=False)

        print(f"\n{'=' * 60}")
        print(f"MEDTRONIC ICD ANALYSIS COMPLETE!")
        print(f"Results saved to: {output_file}")
        print(f"Total traces analysed: {len(analysis_results)}")
        print(f"VF detections: {sum(1 for r in analysis_results if r['vf_detected'])}")
        print(f"VT detections: {sum(1 for r in analysis_results if r['vt_detected'])}")
        print(f"Therapies indicated: {sum(1 for r in analysis_results if r['therapy_indicated'])}")
        print(f"Noise detections: {sum(1 for r in analysis_results if r['noise_detected'])}")
        print(f"T-wave oversensing: {sum(1 for r in analysis_results if r['twave_oversensing'])}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()