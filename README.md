# Monophonic-Pitch-Detection

Run the RealTimeAudioProcessing.py to detect the pitch detected by your device's microphones.

This script detects the pitch of a signal as follows. It calculates the normalized auto correction using the fast fourier transform of the signal, detects its peak value, and performs an interpolation to find the pitch. It also searches through integer submultiples of the detected frequency to account for octave errors that the naive autocorrelation algorithm is prone to.
