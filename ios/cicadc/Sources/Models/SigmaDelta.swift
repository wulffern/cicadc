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

    /// 1-bit output bitstream value (+1 or -1) at sample index k.
    mutating func output(at k: Int, signal: SignalSource, samplePeriod: Double) -> Double {
        let state = runModulator(upTo: k, signal: signal, samplePeriod: samplePeriod)
        return state.lastOutput
    }

    /// Reconstructed (decimated) output at sample index k with averaging length avgTaps.
    mutating func decimatedOutput(
        at k: Int,
        signal: SignalSource,
        samplePeriod: Double,
        avgTaps: Int
    ) -> Double {
        guard avgTaps > 1 else {
            return output(at: k, signal: signal, samplePeriod: samplePeriod)
        }
        var sum = 0.0
        for i in max(0, k - avgTaps + 1)...k {
            sum += output(at: i, signal: signal, samplePeriod: samplePeriod)
        }
        return sum / Double(avgTaps)
    }

    // MARK: - Private

    private struct ModulatorState {
        var integrator1: Double = 0
        var integrator2: Double = 0
        var lastOutput: Double = 0
    }

    mutating func runModulator(upTo k: Int, signal: SignalSource, samplePeriod: Double) -> ModulatorState {
        // Find nearest cached ancestor
        var startK = 0
        var state = ModulatorState()
        for idx in stride(from: k - 1, through: max(0, k - warmupLength * 4), by: -1) {
            if let cached = cache[idx] {
                startK = idx + 1
                state = cached
                break
            }
        }

        for idx in startK...k {
            let t = Double(idx) * samplePeriod
            let input = signal.noisyValue(at: t)
            let d = dither ? deterministicDither(at: idx) : 0.0
            state = step(state: state, input: input + d)
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

    private func step(state: ModulatorState, input: Double) -> ModulatorState {
        var s = state
        switch order {
        case .first:
            let error = input - s.lastOutput
            s.integrator1 += error
            s.lastOutput = s.integrator1 >= 0 ? 1.0 : -1.0
        case .second:
            let fb = s.lastOutput
            let e1 = input - fb
            let e2 = s.integrator1 - fb
            s.integrator1 += e1
            s.integrator2 += e2
            s.lastOutput = s.integrator2 >= 0 ? 1.0 : -1.0
        }
        return s
    }

    private func deterministicDither(at k: Int) -> Double {
        let x = UInt32(bitPattern: Int32(k &* 1664525 &+ 1013904223))
        return (Double(x) / Double(UInt32.max) - 0.5) * 0.5
    }
}
