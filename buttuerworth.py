from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

b, a = signal.butter(10, 50, 'high', analog=True)
w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()

t = np.linspace(0, 1, 1000, False)  # 1 second
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
peaks, properties = signal.find_peaks(sig, height=0.3)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.plot(peaks/1000, sig[peaks], "x")
ax1.set_title('10 Hz and 20 Hz sinusoids')
ax1.axis([0, 1, -2, 2])

sos = signal.butter(10, 20, 'high', fs=1000, output='sos')
print(sos.shape)
sos2 = signal.firwin(11, cutoff=15, fs=1000, pass_zero='lowpass')
print(sos2.shape)

filtered = signal.sosfilt(sos, sig)
filtered2 = signal.lfilter(sos2,[1.0], sig)

peaks, properties = signal.find_peaks(filtered)   #height=0.5
print(peaks)
ax2.plot(t, filtered)
ax2.plot(peaks/1000, filtered[peaks], "x")
ax2.set_title('After 15 Hz high-pass filter')
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()



time_step = 1.0
sig_fft = fftpack.fft(sig)
power = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(sig.size, d=time_step)
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')
plt.show()

pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]




 # Check that it does indeed correspond to the frequency that we generate
 # the signal with
np.allclose(peak_freq, 1./5.)

 # An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])
 # scipy.signal.find_peaks_cwt can also be used for more advanced  peak detection


 #모든 high frequencies를 제거합니다.
high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(t, sig, label='Original signal')
plt.plot(t, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')

plt.show()
