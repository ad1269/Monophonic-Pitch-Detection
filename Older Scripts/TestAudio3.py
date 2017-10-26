import math
import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from math import pi

CHUNKSIZE = 4096 # fixed chunk size
RATE = 44100 # Sample rate

# Audio range we want to detect between
minF = 27.5
maxF = 4186
minP = int(RATE / (maxF - 1))
maxP = int(RATE / (minF + 1))

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

#Test Audio data using my Wave class to generate random samples of a given sine function
def generateSineWave(frequency):
	myWave = Wave(1/RATE, CHUNKSIZE, lambda x: math.sin(2 * math.pi * frequency * x))
	return np.array(myWave.samples)

def generateMiddleC():
	f = 440 * (2 ** (-0.75))
	p = RATE / f
	n = 2 * maxP
	signal = [0 for x in range(n)]
	for i in range(0, n):
		signal[i] += 1.0 * math.sin(2 * pi * 1 * i / p)
		signal[i] += 0.6 * math.sin(2 * pi * 2 * i / p)
		signal[i] += 0.3 * math.sin(2 * pi * 3 * i / p)
	return signal

def generateComplexWave(f):
	p = RATE / f
	n = 2 * maxP
	signal = [0 for x in range(n)]
	for i in range(0, n):
		signal[i] += 1.0 * math.sin(2 * pi * 1 * i / p)
		signal[i] += 0.6 * math.sin(2 * pi * 2 * i / p)
		signal[i] += 0.3 * math.sin(2 * pi * 3 * i / p)
	return signal

def normalizedAC(signal):
	nac = [0 for x in range(maxP + 2)]
	for p in range(minP - 1, maxP + 2):
		ac, sqSumStart, sqSumEnd = 0, 0, 0
		for i in range(0, len(signal) - p):
			ac += signal[i] * signal[i + p]
			sqSumStart += signal[i] * signal[i]
			sqSumEnd += signal[i + p] * signal[i + p]
		nac[p] = ac / math.sqrt(sqSumStart * sqSumEnd)
	return nac

def fastNAC(signal):
	x = np.asarray(signal)
	N = len(x)
	x = x-x.mean()
	s = np.fft.fft(x, N*2-1)
	result = np.real(np.fft.ifft(s * np.conjugate(s), N*2-1))
	result = result[:N]
	result /= result[0]
	return result

def getPeak(nac):
	peak = minP
	for p in range(minP, maxP + 1):
		if nac[p] > nac[peak]:
			peak = p
	return peak

def correctOctaveErrors(nac, bestP, pEst):
	kThreshold = 0.9
	maxMultiple = int(bestP / minP)
	found = False
	mul = maxMultiple
	while not found and mul >= 1:
		allStrong = True

		for k in range(1, mul):
			subMulPeriod = int(k * pEst / mul + 0.5)
			if nac[subMulPeriod] < kThreshold * nac[bestP]:
				allStrong = False

		if allStrong:
			found = True
			pEst = pEst / mul

		mul -= 1

	return pEst 

def estimatePeriod(signal):
	# Tracks the quality of the periodicity of the signal
	q = 0

	#Get the normalized autocorrelation of the signal
	nac = fastNAC(signal)

	# Get the highest peak of the NAC in the range of interest
	bestP = getPeak(nac)

	# If bestP is the highest value, but not the peak, we can't determine the period
	if nac[bestP] < nac[bestP - 1] and nac[bestP] < nac[bestP + 1]:
		return 0

	# Quality of the signal is the NAC at the highest peak
	q = nac[bestP]

	# Interpolate the right and left values to guess the real peak
	left, mid, right = nac[bestP - 1], nac[bestP], nac[bestP + 1]
	assert (2 * mid - left - right) > 0

	# Add the shift to the peak value to get the estimated period
	shift = 0.5 * (right - left) / (2 * mid - left - right)
	pEst = bestP + shift

	# Account for octave errors by looking through all integer submultiple periods
	pEst = correctOctaveErrors(nac, bestP, pEst)

	# Return the period and quality
	return pEst, q

def detectFundamentalFrequency(signal):
	# Estimate the period
	periodEstimate, quality = estimatePeriod(signal)
	frequencyEstimate = 0
	if periodEstimate > 0:
		frequencyEstimate = RATE / periodEstimate

	return frequencyEstimate

def getMicrophoneData():
	# initialize portaudio
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

	# do this as long as you want fresh samples
	data = stream.read(CHUNKSIZE)
	numpydata = np.fromstring(data, dtype=np.int16).tolist()

	# close stream
	stream.stop_stream()
	stream.close()
	p.terminate()
	return numpydata

def graphSignal(signal):
	plt.plot(signal)
	plt.figure()
	plt.plot(fastNAC(signal))
	plt.figure()
	plt.plot(normalizedAC(signal))
	plt.show(block=False)

signal = generateComplexWave(200)
graphSignal(signal)
print(detectFundamentalFrequency(signal))

plt.show()