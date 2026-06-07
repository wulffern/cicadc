import SwiftUI

/// Frequency spectrum view (dBFS bar chart).
struct FFTView: View {
    let magnitudes: [Double]    // dBFS values, length = fftSize/2
    let sampleRate: Double
    let dbMin: Double = -100
    let dbMax: Double = 0

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottomLeading) {
                Color.black

                // Grid lines at -20, -40, -60, -80 dB
                ForEach([-20, -40, -60, -80], id: \.self) { db in
                    let frac = CGFloat((Double(db) - dbMin) / (dbMax - dbMin))
                    let y = geo.size.height * (1 - frac)
                    HStack {
                        Text("\(db)dB")
                            .font(.system(size: 8))
                            .foregroundColor(.gray)
                            .frame(width: 36, alignment: .trailing)
                        Spacer()
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .offset(y: y - 6)
                    Path { path in
                        path.move(to: CGPoint(x: 38, y: y))
                        path.addLine(to: CGPoint(x: geo.size.width, y: y))
                    }
                    .stroke(Color.gray.opacity(0.3), style: StrokeStyle(lineWidth: 0.5, dash: [4, 4]))
                }

                // Spectrum as a single filled curve on a log-frequency axis.
                if magnitudes.count > 1 {
                    let plotWidth = geo.size.width - 38
                    Path { path in
                        let n = magnitudes.count
                        let lo = log(1.0 / Double(2 * n))   // lowest bin: f/fs = 1/N
                        let hi = log(0.5)                    // Nyquist
                        func x(_ i: Int) -> CGFloat {
                            let frac = (log(Double(i) / Double(2 * n)) - lo) / (hi - lo)
                            return 38 + CGFloat(frac.clamped(to: 0...1)) * plotWidth
                        }
                        func y(_ db: Double) -> CGFloat {
                            let frac = ((db - dbMin) / (dbMax - dbMin)).clamped(to: 0...1)
                            return geo.size.height * (1 - CGFloat(frac))
                        }
                        path.move(to: CGPoint(x: x(1), y: geo.size.height))
                        for i in 1..<n {
                            path.addLine(to: CGPoint(x: x(i), y: y(magnitudes[i])))
                        }
                        path.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height))
                        path.closeSubpath()
                    }
                    .fill(Color.cyan.opacity(0.35))
                    .overlay(
                        Path { path in
                            let n = magnitudes.count
                            let lo = log(1.0 / Double(2 * n))
                            let hi = log(0.5)
                            func x(_ i: Int) -> CGFloat {
                                let frac = (log(Double(i) / Double(2 * n)) - lo) / (hi - lo)
                                return 38 + CGFloat(frac.clamped(to: 0...1)) * plotWidth
                            }
                            func y(_ db: Double) -> CGFloat {
                                let frac = ((db - dbMin) / (dbMax - dbMin)).clamped(to: 0...1)
                                return geo.size.height * (1 - CGFloat(frac))
                            }
                            path.move(to: CGPoint(x: x(1), y: y(magnitudes[1])))
                            for i in 2..<n {
                                path.addLine(to: CGPoint(x: x(i), y: y(magnitudes[i])))
                            }
                        }
                        .stroke(Color.cyan, lineWidth: 1)
                    )
                }

                Text("Spectrum (dBFS)")
                    .font(.caption.bold())
                    .foregroundColor(.cyan)
                    .padding(4)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .offset(x: 38)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}

extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
