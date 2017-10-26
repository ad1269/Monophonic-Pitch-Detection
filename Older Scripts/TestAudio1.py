import math
import pyaudio
import numpy as np
from matplotlib import pyplot as plt

class Wave(object):
	"""docstring for Wave"""
	def __init__(self, samplingRate, numSamples, function):
		super(Wave, self).__init__()
		self.numSamples = numSamples
		self.samplingRate = samplingRate
		self.function = function
		self.samples = []
		self.times = []
		self.sampleFunction()

	def sampleFunction(self):
		for t in np.linspace(0, self.samplingRate*self.numSamples, self.numSamples):
			self.times.append(t)
			self.samples.append(self.function(t))

def getPeaks(auto):
	peaks = []
	peakIndices = []
	for i in range(1, len(auto) - 1):
		prev = auto[i - 1]
		current = auto[i]
		nxt = auto[i + 1]

		if prev < current and current < nxt:
			peaks.append(current)
			peakIndices.append(i)

	return peaks, peakIndices

CHUNKSIZE = 1024 # fixed chunk size
RATE = 44100
f = 400

#Test Audio data using my Wave class to generate random samples of a given sine function
myWave = Wave(1/RATE, CHUNKSIZE, lambda x: 30000*math.sin(2 * 3.14159 * f * x))
numpydata = np.array(myWave.samples)
t = myWave.times
print(myWave.samples)

"""
# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

# do this as long as you want fresh samples
data = stream.read(CHUNKSIZE)
numpydata = np.fromstring(data, dtype=np.int16).tolist()
t = np.linspace(0, CHUNKSIZE / RATE, CHUNKSIZE)[round(CHUNKSIZE/2):]

# close stream
stream.stop_stream()
stream.close()
p.terminate()
"""

# plot data
autocorrelated = np.correlate(numpydata, numpydata, "same")

# detect peaks
#peak, peakIndices = getPeaks(autocorrelated)
print(numpydata)
print(autocorrelated)

# Find peaks
top = np.argmax(autocorrelated)
bottom = np.argmin(autocorrelated[top:]) + top
second = np.argmax(autocorrelated[bottom:]) + bottom

print(1/(t[second] - t[top]))

print(t[top], t[bottom], t[second])

plt.plot(numpydata)
plt.figure()
plt.plot(autocorrelated)
plt.show()