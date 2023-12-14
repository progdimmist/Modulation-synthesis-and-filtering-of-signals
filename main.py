import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import hilbert

time = np.linspace(0, 4, 1000)
modulating_signal = np.piecewise(np.sin(2 * np.pi * 1 * time), [np.sin(2 * np.pi * 1 * time) < 0, np.sin(2 * np.pi * 1 * time) > 0], [0, 1])
frequency = 12

amplitude_signal = modulating_signal * np.sin(2 * np.pi * frequency * time)
frequency_signal = np.sin(2 * np.pi * (frequency + 10 * modulating_signal) * time)
phase_signal = np.sin(2 * np.pi * frequency * time + np.pi * modulating_signal)

amplitude_signal_spectrum = np.abs(fft(amplitude_signal - np.mean(amplitude_signal)))
frequency_signal_spectrum = np.abs(fft(frequency_signal - np.mean(frequency_signal)))
phase_signal_spectrum = np.abs(fft(phase_signal - np.mean(phase_signal)))

frequency_index = np.argmax(amplitude_signal_spectrum)
cutoff_frequency = 10

copy = amplitude_signal_spectrum.copy()
copy[:frequency_index - cutoff_frequency] = 0
copy[frequency_index + cutoff_frequency + 1:] = 0

synthesized_signal = ifft(copy)
synthesized_signal = np.real(synthesized_signal)

f_signal = np.abs(hilbert(synthesized_signal))

mean_frequency = np.mean(f_signal)
filtered_signal = np.piecewise(f_signal, [f_signal < mean_frequency, f_signal > mean_frequency], [0, 1])

plt.figure()

plt.subplot(4, 1, 1)
plt.plot(time, modulating_signal)
plt.title('Modulating signal (Meander)')

plt.subplot(4, 1, 2)
plt.plot(time, amplitude_signal)
plt.title('Amplitude modulated signal')

plt.subplot(4, 1, 3)
plt.plot(time, frequency_signal)
plt.title('Frequency modulated signal')

plt.subplot(4, 1, 4)
plt.plot(time, phase_signal)
plt.title('Phase modulated signal')

plt.tight_layout()
plt.figure()

plt.subplot(4, 1, 1)
plt.plot(np.fft.fftfreq(len(amplitude_signal_spectrum), 1 / 1000), amplitude_signal_spectrum)
plt.title('Amplitude modulation spectrum')
plt.xlim(0, 120)
plt.ylim(0, 400)

plt.subplot(4, 1, 2)
plt.plot(np.fft.fftfreq(len(frequency_signal_spectrum), 1 / 1000), frequency_signal_spectrum)
plt.title('Frequency modulation spectrum')
plt.xlim(0, 120)
plt.ylim(0, 400)

plt.subplot(4, 1, 3)
plt.plot(np.fft.fftfreq(len(phase_signal_spectrum), 1 / 1000), phase_signal_spectrum)
plt.title('Phase modulation spectrum')
plt.xlim(0, 120)
plt.ylim(0, 400)

plt.subplot(4, 1, 4)
plt.plot(np.fft.fftfreq(len(copy), 1 / 1000), copy)
plt.title('Spectrum without high and low frequencies')
plt.xlim(0, 120)
plt.ylim(0, 400)

plt.tight_layout()
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(time, synthesized_signal)
plt.title('Synthesized signal')

plt.subplot(2, 1, 2)
plt.plot(time, filtered_signal)
plt.title('Filtered signal')

plt.tight_layout()
plt.show()
