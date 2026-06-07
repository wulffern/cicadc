import Foundation

/// Normalized sinc^M decimation filter — M cascaded K-tap boxcars, matching the
/// Python reference (`manim_scene._decimation_taps` / `_filt_level`).
///
/// `M` is the decimator order: Nyquist uses a plain moving average (sinc^1), a
/// 1st-order ΣΔ is matched by sinc^2 and a 2nd-order ΣΔ by sinc^3 — the textbook
/// "decimator order = modulator order + 1" rule. The composite FIR has
/// `sum(h) == K^M`, so dividing by it gives unity gain at DC.
struct Decimator {
    let taps: Int      // K
    let order: Int     // M

    private let fir: [Double]
    private let firSum: Double

    init(taps: Int, order: Int) {
        let K = max(1, taps)
        let M = max(1, order)
        self.taps = K
        self.order = M
        var h = [1.0]
        let box = [Double](repeating: 1.0, count: K)
        for _ in 0..<M { h = Decimator.convolve(h, box) }
        self.fir = h
        self.firSum = h.reduce(0, +)
    }

    /// Length of the composite impulse response, in samples.
    var length: Int { fir.count }

    /// Decimated digital level at sample `k`: the sinc^M cascade applied to the
    /// raw levels (unity DC gain), then divided by the in-band filter gain so the
    /// recovered sinusoid keeps its amplitude.
    func filtered(at k: Int, gain: Double, rawLevel: (Int) -> Double) -> Double {
        guard taps > 1 else { return rawLevel(k) }
        var acc = 0.0
        for (j, hj) in fir.enumerated() { acc += hj * rawLevel(k - j) }
        return (acc / firSum) / gain
    }

    /// In-band magnitude of the sinc^M filter at digital frequency `w`
    /// (rad/sample), multiplied by the modulator signal-transfer magnitude.
    /// Clamped to avoid a blow-up near a filter null.
    func filterGain(w: Double, stfMag: Double) -> Double {
        guard taps > 1 else { return 1.0 }
        let s = sin(w / 2.0)
        guard abs(s) > 1e-9 else { return 1.0 }
        let mag = pow(abs(sin(Double(taps) * w / 2.0) / s) / Double(taps), Double(order))
        return max(mag * stfMag, 0.05)
    }

    /// Group delay in seconds: the linear-phase FIR delay `M*(K-1)/2` samples,
    /// plus the modulator STF phase delay `-arg(STF)/w` when a decimator is
    /// active, so the reconstructed output lines up in time with the analog
    /// signal it represents.
    func groupDelay(samplePeriod: Double, w: Double, stfPhase: Double) -> Double {
        var delay = Double(order) * Double(taps - 1) / 2.0 * samplePeriod
        if taps > 1, abs(w) > 1e-9 {
            delay += (-stfPhase / w) * samplePeriod
        }
        return delay
    }

    private static func convolve(_ a: [Double], _ b: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: a.count + b.count - 1)
        for i in 0..<a.count {
            for j in 0..<b.count { out[i + j] += a[i] * b[j] }
        }
        return out
    }
}
