import Accelerate

/// Computes a real-valued FFT magnitude spectrum in dBFS.
struct FFTHelper {
    /// Returns magnitude spectrum in dBFS for the given samples.
    /// Output length = samples.count / 2.
    static func magnitudeSpectrum(samples: [Double]) -> [Double] {
        let n = samples.count
        guard n > 0, n.isPowerOfTwo else { return [] }

        var real = samples.map { Float($0) }
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

        // Normalize and convert to dBFS
        let scale = Float(1.0 / Double(n))
        var scaled = magnitudes.map { $0 * scale * 2 }  // *2 for single-sided
        scaled[0] /= 2  // DC correction

        return scaled.map { m -> Double in
            let db = 20.0 * log10(Double(max(m, 1e-10)))
            return db.clamped(to: -120...0)
        }
    }
}

extension Int {
    var isPowerOfTwo: Bool { self > 0 && (self & (self - 1)) == 0 }
}
