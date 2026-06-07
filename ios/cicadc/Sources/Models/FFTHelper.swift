import Accelerate

/// Computes a real-valued FFT magnitude spectrum in dBFS.
struct FFTHelper {
    /// Returns magnitude spectrum in dBFS for the given samples. A Hann window is
    /// applied and the result is normalised by the window's coherent gain so a
    /// full-scale sine (amplitude 1.0) peaks at 0 dBFS.
    /// Output length = samples.count / 2.
    static func magnitudeSpectrum(samples: [Double]) -> [Double] {
        let n = samples.count
        guard n > 0, n.isPowerOfTwo else { return [] }

        // Hann window + its coherent gain (sum(win) / 2) for 0 dBFS calibration.
        var windowSum = 0.0
        let windowed: [Double] = samples.enumerated().map { i, s in
            let w = 0.5 - 0.5 * cos(2.0 * .pi * Double(i) / Double(n - 1))
            windowSum += w
            return s * w
        }
        let coherentGain = max(windowSum / 2.0, 1e-9)

        var real = windowed.map { Float($0) }
        var imag = [Float](repeating: 0, count: n)

        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var splitComplex = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                let log2n = vDSP_Length(log2(Double(n)))
                guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return }
                defer { vDSP_destroy_fftsetup(setup) }
                vDSP_fft_zip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
            }
        }

        let halfN = n / 2
        var magnitudes = [Float](repeating: 0, count: halfN)
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var splitComplex = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_zvabs(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfN))
            }
        }

        // Normalize by the window's coherent gain (single-sided) and convert to dBFS.
        let scale = 1.0 / coherentGain
        var scaled = magnitudes.map { Double($0) * scale }
        if !scaled.isEmpty { scaled[0] /= 2 }  // DC is single-sided already

        return scaled.map { m -> Double in
            let db = 20.0 * log10(max(m, 1e-10))
            return db.clamped(to: -120...0)
        }
    }
}

extension Int {
    var isPowerOfTwo: Bool { self > 0 && (self & (self - 1)) == 0 }
}
