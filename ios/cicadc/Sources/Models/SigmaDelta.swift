import Foundation

enum SigmaDeltaOrder: Int, CaseIterable {
    case first = 1
    case second = 2
}

/// Stateless sigma-delta modulator — computes output by running from a warm-up index.
struct SigmaDelta {
    var order: SigmaDeltaOrder
    var bits: Int                // output quantizer bits (usually 1)
    var dither: Bool

    // Cache: sample index → modulator state snapshot after processing that sample.
    private var cache: [Int: ModulatorState] = [:]
    private let warmupLength = 64

    init(order: SigmaDeltaOrder, bits: Int, dither: Bool) {
        self.order = order
        self.bits = bits
        self.dither = dither
    }

    /// Drop all cached state. Call whenever the input signal or quantizer
    /// parameters change so stale modulator states are not reused.
    mutating func reset() {
        cache.removeAll(keepingCapacity: true)
    }

    /// Coarse modulator output (a reconstructed quantizer level) at sample index
    /// k. Decimation/averaging is handled separately by `Decimator`.
    mutating func output(at k: Int, signal: SignalSource, samplePeriod: Double) -> Double {
        let state = runModulator(upTo: k, signal: signal, samplePeriod: samplePeriod)
        return state.lastOutput
    }

    // MARK: - Private

    private struct ModulatorState {
        var integrator1: Double = 0
        var integrator2: Double = 0
        var lastOutput: Double = 0
    }

    private mutating func runModulator(upTo k: Int, signal: SignalSource, samplePeriod: Double) -> ModulatorState {
        // Cold start warms up from a settled-enough past (works for any k,
        // including negative sample indices). A nearer cached ancestor, if one
        // exists, lets us extend it in O(1) instead.
        var startK = k - warmupLength
        var state = ModulatorState()
        for idx in stride(from: k - 1, through: k - warmupLength * 4, by: -1) {
            if let cached = cache[idx] {
                startK = idx + 1
                state = cached
                break
            }
        }

        for idx in startK...k {
            let t = Double(idx) * samplePeriod
            let input = signal.noisyValue(at: t)
            state = step(state: state, input: input, k: idx)
            if idx == k || idx % 8 == 0 {
                cache[idx] = state
            }
        }
        // Trim cache to avoid unbounded growth
        if cache.count > 512 {
            let sorted = cache.keys.sorted()
            sorted.prefix(128).forEach { cache.removeValue(forKey: $0) }
        }
        return state
    }

    private func step(state: ModulatorState, input: Double, k: Int) -> ModulatorState {
        var s = state
        switch order {
        case .first:
            s.integrator1 += input - s.lastOutput
            s.lastOutput = quantize(s.integrator1, k: k)
        case .second:
            // Standard stable single-bit CIFB design: a1 = 1, a2 = 2, b1 = g1 = 1
            // (delaying integrators), realising a (1 - z^-1)^2 noise transfer.
            // The second integrator uses the just-updated first integrator.
            let y = s.lastOutput
            s.integrator1 += input - y
            s.integrator2 += s.integrator1 - 2.0 * y
            s.lastOutput = quantize(s.integrator2, k: k)
        }
        return s
    }

    /// True `bits`-bit mid-rise quantizer: 2^bits levels symmetric in [-1, 1]
    /// (a 1-bit quantizer returns only ±1). Dither, when enabled, is added at the
    /// quantizer input, mirroring the Python reference.
    private func quantize(_ value: Double, k: Int) -> Double {
        var v = value
        if dither { v += 0.5 * deterministicDither(at: k) }
        let levels = 1 << bits
        if levels <= 2 { return v >= 0.0 ? 1.0 : -1.0 }
        let step = 2.0 / Double(levels - 1)
        var idx = Int((v + 1.0) / step + 0.5)
        idx = max(0, min(levels - 1, idx))
        return -1.0 + Double(idx) * step
    }

    private func deterministicDither(at k: Int) -> Double {
        let x = sin(Double(k) * 78.233 + 1.0) * 43758.5453
        return 2.0 * (x - floor(x)) - 1.0
    }
}
