import SwiftUI
import Combine

@MainActor
final class ADCViewModel: ObservableObject {
    @Published var params = ADCParameters()
    @Published var isPlaying = true

    // Rendered data for views
    @Published var analogSamples: [(t: Double, v: Double)] = []
    @Published var digitalSamples: [(t: Double, v: Double)] = []
    @Published var errorSamples: [(t: Double, v: Double)] = []
    @Published var fftMagnitudes: [Double] = []
    @Published var tNow: Double = 0.0

    private var displayLink: Timer?
    private var sigmaDelta = SigmaDelta(order: .first, bits: 1, dither: false)
    private let visibleDuration = 4.0   // seconds of signal visible on screen
    private let continuousSamples = 256 // points for analog curve
    private let fftSize = 512

    init() {
        start()
    }

    func start() {
        displayLink = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.tick() }
        }
    }

    func stop() {
        displayLink?.invalidate()
        displayLink = nil
    }

    func togglePlayPause() {
        isPlaying.toggle()
    }

    // MARK: - Tick

    private func tick() {
        guard isPlaying else { return }
        let dt = (1.0 / 30.0) * params.speed
        tNow += dt
        recompute()
    }

    func recompute() {
        let src = SignalSource(
            frequency: params.frequency,
            amplitude: params.amplitude,
            noiseAmplitude: params.noiseAmplitude
        )
        let quant = Quantizer(bits: params.bits)

        let half = visibleDuration / 2.0
        let tStart = tNow - half
        let tEnd = tNow + half

        // Analog curve
        analogSamples = (0..<continuousSamples).map { i in
            let t = tStart + (tEnd - tStart) * Double(i) / Double(continuousSamples - 1)
            return (t: t, v: src.noisyValue(at: t))
        }

        // Digital samples (quantized)
        let kStart = Int(floor(tStart / params.samplePeriod))
        let kEnd = Int(ceil(tEnd / params.samplePeriod))
        var digital: [(t: Double, v: Double)] = []
        var errors: [(t: Double, v: Double)] = []

        for k in kStart...kEnd {
            let t = Double(k) * params.samplePeriod
            guard t >= tStart && t <= tEnd else { continue }
            let input = src.noisyValue(at: t)
            let dv: Double
            let err: Double
            switch params.adcType {
            case .nyquist:
                let r = quant.quantize(input)
                dv = r.level
                err = r.error
            case .sigmaDelta1:
                sigmaDelta.order = .first
                sigmaDelta.bits = 1
                sigmaDelta.dither = params.dither
                if params.avgTaps > 1 {
                    dv = sigmaDelta.decimatedOutput(at: k, signal: src, samplePeriod: params.samplePeriod, avgTaps: params.avgTaps)
                } else {
                    dv = sigmaDelta.output(at: k, signal: src, samplePeriod: params.samplePeriod)
                }
                err = dv - input
            case .sigmaDelta2:
                sigmaDelta.order = .second
                sigmaDelta.bits = 1
                sigmaDelta.dither = params.dither
                if params.avgTaps > 1 {
                    dv = sigmaDelta.decimatedOutput(at: k, signal: src, samplePeriod: params.samplePeriod, avgTaps: params.avgTaps)
                } else {
                    dv = sigmaDelta.output(at: k, signal: src, samplePeriod: params.samplePeriod)
                }
                err = dv - input
            }
            digital.append((t: t, v: dv))
            errors.append((t: t, v: err))
        }
        digitalSamples = digital
        errorSamples = errors

        // FFT: collect fftSize digital samples ending at kNow
        let kNow = Int(floor(tNow / params.samplePeriod))
        var fftInput = [Double](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            let k = kNow - fftSize + 1 + i
            if k < 0 { continue }
            let t = Double(k) * params.samplePeriod
            let input = src.noisyValue(at: t)
            switch params.adcType {
            case .nyquist:
                fftInput[i] = quant.quantize(input).level
            case .sigmaDelta1, .sigmaDelta2:
                fftInput[i] = sigmaDelta.output(at: k, signal: src, samplePeriod: params.samplePeriod)
            }
        }
        fftMagnitudes = FFTHelper.magnitudeSpectrum(samples: fftInput)
    }
}
