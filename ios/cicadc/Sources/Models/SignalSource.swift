import Foundation

/// Generates a sinusoidal analog signal with optional smooth noise.
struct SignalSource {
    var frequency: Double   // Hz
    var amplitude: Double   // fraction of full scale [0, 1]
    var noiseAmplitude: Double  // fraction of full scale [0, 1]

    /// Clean sinusoid value at absolute time t.
    func cleanValue(at t: Double) -> Double {
        amplitude * sin(2 * .pi * frequency * t)
    }

    /// Noisy signal value at absolute time t (smooth value-noise added).
    func noisyValue(at t: Double) -> Double {
        let clean = cleanValue(at: t)
        guard noiseAmplitude > 0 else { return clean }
        let noise = smoothNoise(t: t) * noiseAmplitude
        return (clean + noise).clamped(to: -1...1)
    }

    // Smooth pseudo-random noise that scrolls with time (not per-frame random).
    private func smoothNoise(t: Double) -> Double {
        let scale = 3.0
        let x = t * frequency * scale
        let xi = Int(floor(x))
        let xf = x - Double(xi)
        let fade = xf * xf * (3 - 2 * xf)
        return lerp(a: hash(xi), b: hash(xi + 1), t: fade)
    }

    private func hash(_ n: Int) -> Double {
        var x = UInt64(bitPattern: Int64(n &* 1234567891 &+ 987654321))
        x = x ^ (x >> 30)
        x = x &* 0xbf58476d1ce4e5b9
        x = x ^ (x >> 27)
        x = x &* 0x94d049bb133111eb
        x = x ^ (x >> 31)
        return Double(x) / Double(UInt64.max) * 2.0 - 1.0
    }

    private func lerp(a: Double, b: Double, t: Double) -> Double {
        a + (b - a) * t
    }
}

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
