import Foundation

enum ADCType: String, CaseIterable, Identifiable {
    case nyquist = "Nyquist"
    case sigmaDelta1 = "ΣΔ 1st"
    case sigmaDelta2 = "ΣΔ 2nd"
    var id: String { rawValue }
}

/// All user-adjustable parameters for the ADC simulation.
struct ADCParameters {
    var frequency: Double = 1.0     // Hz
    var amplitude: Double = 0.8     // fraction of FS
    var noiseAmplitude: Double = 0.0
    var samplePeriod: Double = 0.1  // seconds
    var bits: Int = 4
    var adcType: ADCType = .nyquist
    var dither: Bool = false
    var avgTaps: Int = 1            // decimation filter length
    var speed: Double = 1.0         // simulation speed multiplier

    var sampleRate: Double { 1.0 / samplePeriod }
}
