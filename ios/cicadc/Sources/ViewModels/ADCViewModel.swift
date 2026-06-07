import SwiftUI
import Combine

@MainActor
final class ADCViewModel: ObservableObject {
    @Published var params = ADCParameters()
    @Published var isPlaying = true

    // Rendered data for views
    @Published var analogSamples: [(t: Double, v: Double)] = []
    @Published var analogSampleDots: [(t: Double, v: Double)] = []  // ADC sampling instants on the analog curve
    @Published var digitalSamples: [(t: Double, v: Double)] = []   // primary (white) output staircase
    @Published var coarseSamples: [(t: Double, v: Double)] = []    // ΣΔ/quantizer output (pale green), only when a decimator is active
    @Published var errorSamples: [(t: Double, v: Double)] = []
    @Published var errorFullScale: Double = 1.0                    // one LSB; the error strip spans ±this
    @Published var hasFilter = false                              // a decimator/averager is active
    @Published var groupDelay: Double = 0.0                       // decimator group delay, seconds
    @Published var fftMagnitudes: [Double] = []
    @Published var tNow: Double = 0.0

    private var displayLink: Timer?
    private var cancellables = Set<AnyCancellable>()
    private var sigmaDelta = SigmaDelta(order: .first, bits: 1, dither: false)
    private var lastSignature = ""    // ΣΔ cache is reset when this changes
    private var stfSignature = ""     // modulator STF is re-measured when this changes
    private var stfMag = 1.0
    private var stfPhase = 0.0
    private let visibleDuration = 4.0   // seconds of signal visible on screen
    private let continuousSamples = 256 // points for analog curve
    private let fftSize = 512
    private let frameRate = 30.0

    init() {
        // Re-render on any parameter change, paused or not (mirrors the Python
        // app's refresh-on-change). Deferred to the next runloop tick so the new
        // parameter value is committed before recompute() reads it.
        $params
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.recompute() }
            .store(in: &cancellables)

        recompute()
        start()
    }

    func start() {
        guard displayLink == nil else { return }
        displayLink = Timer.scheduledTimer(withTimeInterval: 1.0 / frameRate, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.tick() }
        }
    }

    func stop() {
        displayLink?.invalidate()
        displayLink = nil
    }

    func togglePlayPause() {
        isPlaying.toggle()
        if isPlaying { start() } else { stop() }
    }

    // MARK: - Tick

    private func tick() {
        guard isPlaying else { return }
        let dt = (1.0 / frameRate) * params.speed
        tNow += dt
        recompute()
    }

    /// Reset the ΣΔ modulator cache when any parameter that changes its output
    /// sequence has changed.
    private func syncModulator() {
        let sig = "\(params.frequency)|\(params.amplitude)|\(params.samplePeriod)|\(params.bits)|\(params.dither)|\(params.adcType.rawValue)|\(params.noiseAmplitude)"
        if sig != lastSignature {
            lastSignature = sig
            sigmaDelta.reset()
        }
    }

    func recompute() {
        syncModulator()
        let src = SignalSource(
            frequency: params.frequency,
            amplitude: params.amplitude,
            noiseAmplitude: params.noiseAmplitude
        )
        let quant = Quantizer(bits: params.bits)
        configureModulator()

        let Ts = params.samplePeriod
        let half = visibleDuration / 2.0
        let tStart = tNow - half
        let tEnd = tNow + half

        // Analog curve (the noisy ADC input).
        analogSamples = (0..<continuousSamples).map { i in
            let t = tStart + (tEnd - tStart) * Double(i) / Double(continuousSamples - 1)
            return (t: t, v: src.noisyValue(at: t))
        }

        // Decimation filter (sinc^M) and its in-band gain / group delay.
        let M = filterOrder()
        let K = max(1, params.avgTaps)
        let decim = Decimator(taps: K, order: M)
        let w = 2.0 * .pi * params.frequency * Ts
        let stf = modulatorSTF(src: src, quant: quant)
        let gain = decim.filterGain(w: w, stfMag: stf.mag)
        let delay = decim.groupDelay(samplePeriod: Ts, w: w, stfPhase: stf.phase)
        hasFilter = K > 1
        groupDelay = hasFilter ? delay : 0.0

        // Index ranges touched this frame.
        let kNow = Int(floor(tNow / Ts))
        let kStart = Int(floor(tStart / Ts)) - 1
        let kEnd = Int(ceil(tEnd / Ts)) + 1
        let fftStart = kNow - fftSize + 1

        // Precompute the raw (unfiltered) level for every index we will read,
        // into a flat array — the decimator reads up to `decim.length` past
        // samples per output, so doing this once avoids tens of thousands of
        // per-frame modulator lookups (which otherwise stall the main thread).
        let minK = min(kStart, fftStart) - (decim.length - 1)
        let maxK = max(kEnd, kNow)
        let raw = rawLevels(from: minK, to: maxK, src: src, quant: quant)
        let rawAt: (Int) -> Double = { k in raw[k - minK] }

        // Sample-and-hold staircases over the visible window (one sample of slack
        // each side so the steps reach the panel edges).
        var coarse: [(t: Double, v: Double)] = []
        var filtered: [(t: Double, v: Double)] = []
        var errors: [(t: Double, v: Double)] = []
        var dots: [(t: Double, v: Double)] = []
        for k in kStart...kEnd {
            let tk = Double(k) * Ts
            let rawV = rawAt(k)
            coarse.append((t: tk, v: rawV))
            dots.append((t: tk, v: src.noisyValue(at: tk)))
            if hasFilter {
                // Filtered output is delay-compensated so it lines up in time
                // with the analog signal it represents.
                let f = decim.filtered(at: k, gain: gain, rawLevel: rawAt)
                filtered.append((t: tk - delay, v: f))
                let ref = src.cleanValue(at: tk - delay)
                errors.append((t: tk - delay, v: ref - f))
            } else {
                let ref = src.cleanValue(at: tk)
                errors.append((t: tk, v: ref - rawV))
            }
        }

        if hasFilter {
            coarseSamples = coarse        // pale-green coarse modulator/quantizer output
            digitalSamples = filtered     // white decimated digital output
        } else {
            coarseSamples = []
            digitalSamples = coarse       // the quantizer/modulator output is the digital output
        }
        errorSamples = errors
        analogSampleDots = dots
        errorFullScale = max(quant.stepSize, 1e-6)

        // FFT of the digital output (decimated when a filter is active). The
        // signal is defined for all time, so negative sample indices are
        // evaluated normally rather than zero-filled.
        var fftInput = [Double](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            let k = fftStart + i
            fftInput[i] = hasFilter ? decim.filtered(at: k, gain: gain, rawLevel: rawAt) : rawAt(k)
        }
        fftMagnitudes = FFTHelper.magnitudeSpectrum(samples: fftInput)
    }

    /// Raw (unfiltered) digital levels for the contiguous index range
    /// `[k0, k1]`. Evaluating ascending lets the ΣΔ modulator extend its cache in
    /// O(1) per sample, and the flat array removes per-read dictionary lookups.
    private func rawLevels(from k0: Int, to k1: Int, src: SignalSource, quant: Quantizer) -> [Double] {
        guard k1 >= k0 else { return [] }
        var out = [Double](repeating: 0, count: k1 - k0 + 1)
        switch params.adcType {
        case .nyquist:
            for k in k0...k1 {
                out[k - k0] = quant.quantize(src.noisyValue(at: Double(k) * params.samplePeriod)).level
            }
        case .sigmaDelta1, .sigmaDelta2:
            for k in k0...k1 {
                out[k - k0] = sigmaDelta.output(at: k, signal: src, samplePeriod: params.samplePeriod)
            }
        }
        return out
    }

    // MARK: - Digital chain helpers

    /// Decimator order M for the active ADC mode (modulator order + 1).
    private func filterOrder() -> Int {
        switch params.adcType {
        case .nyquist: return 1
        case .sigmaDelta1: return 2
        case .sigmaDelta2: return 3
        }
    }

    /// Apply the live ADC settings to the modulator before it is evaluated.
    private func configureModulator() {
        switch params.adcType {
        case .sigmaDelta1: sigmaDelta.order = .first
        case .sigmaDelta2: sigmaDelta.order = .second
        case .nyquist: break
        }
        sigmaDelta.bits = params.bits
        sigmaDelta.dither = params.dither
    }

    /// Unfiltered digital level at sample `k`: the coarse modulator output for a
    /// ΣΔ ADC, otherwise the memoryless uniform-quantizer level.
    private func rawLevel(_ k: Int, src: SignalSource, quant: Quantizer) -> Double {
        switch params.adcType {
        case .nyquist:
            let t = Double(k) * params.samplePeriod
            return quant.quantize(src.noisyValue(at: t)).level
        case .sigmaDelta1, .sigmaDelta2:
            return sigmaDelta.output(at: k, signal: src, samplePeriod: params.samplePeriod)
        }
    }

    /// Measured signal transfer STF(e^{jw}) of the active modulator, via a
    /// lock-in of the modulator output against the clean input at the signal
    /// frequency. Unity for Nyquist; cached per parameter set (it is independent
    /// of time). Mirrors `manim_scene._modulator_stf`.
    private func modulatorSTF(src: SignalSource, quant: Quantizer) -> (mag: Double, phase: Double) {
        guard params.adcType != .nyquist else { return (1.0, 0.0) }
        let f = params.frequency, Ts = params.samplePeriod
        let w = 2.0 * .pi * f * Ts
        guard abs(w) > 1e-6 else { return (1.0, 0.0) }

        let sig = "\(params.adcType.rawValue)|\(params.bits)|\(params.dither)|\(f)|\(Ts)|\(params.amplitude)"
        if sig == stfSignature { return (stfMag, stfPhase) }

        let sampPerCycle = 1.0 / max(f * Ts, 1e-9)
        let n = Int(min(max(sampPerCycle * 24.0, 64.0), 4000.0))
        let k0 = Int(floor(tNow / Ts))
        var numRe = 0.0, numIm = 0.0, denRe = 0.0, denIm = 0.0
        for idx in 0..<n {
            let k = k0 - n + 1 + idx
            let t = Double(k) * Ts
            let y = rawLevel(k, src: src, quant: quant)
            let ref = params.amplitude * sin(2.0 * .pi * f * t)
            let win = 0.5 - 0.5 * cos(2.0 * .pi * Double(idx) / Double(n - 1))  // Hann
            let ang = -2.0 * .pi * f * t
            let pr = cos(ang), pim = sin(ang)
            numRe += y * win * pr;  numIm += y * win * pim
            denRe += ref * win * pr; denIm += ref * win * pim
        }
        let denMag2 = denRe * denRe + denIm * denIm
        var hMag = 1.0, hPhase = 0.0
        if denMag2 > 1e-24 {
            let hRe = (numRe * denRe + numIm * denIm) / denMag2
            let hIm = (numIm * denRe - numRe * denIm) / denMag2
            hMag = (hRe * hRe + hIm * hIm).squareRoot()
            hPhase = atan2(hIm, hRe)
        }
        stfSignature = sig; stfMag = hMag; stfPhase = hPhase
        return (hMag, hPhase)
    }
}
