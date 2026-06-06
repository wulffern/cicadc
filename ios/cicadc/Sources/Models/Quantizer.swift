import Foundation

/// Result of quantizing a sample.
struct QuantizerResult {
    let code: Int         // integer code 0..<2^bits
    let level: Double     // reconstructed analog level in [-1, 1]
    let error: Double     // quantization error = level - input
}

/// N-bit mid-tread uniform quantizer over [-vRef, +vRef].
struct Quantizer {
    var bits: Int
    var vRef: Double = 1.0

    var levels: Int { 1 << bits }
    var stepSize: Double { 2.0 * vRef / Double(levels) }

    func quantize(_ input: Double) -> QuantizerResult {
        let clipped = input.clamped(to: -vRef...vRef)
        // Map to [0, levels)
        let normalized = (clipped + vRef) / stepSize
        let code = min(Int(floor(normalized)), levels - 1)
        let level = (Double(code) + 0.5) * stepSize - vRef
        return QuantizerResult(code: code, level: level, error: level - clipped)
    }
}
