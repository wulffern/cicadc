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

                // Spectrum bars
                if !magnitudes.isEmpty {
                    let barWidth = (geo.size.width - 38) / CGFloat(magnitudes.count)
                    ForEach(0..<magnitudes.count, id: \.self) { i in
                        let db = magnitudes[i]
                        let frac = CGFloat((db - dbMin) / (dbMax - dbMin)).clamped(to: 0...1)
                        let barHeight = frac * geo.size.height
                        Rectangle()
                            .fill(Color.cyan.opacity(0.8))
                            .frame(width: max(barWidth - 1, 1), height: barHeight)
                            .position(
                                x: 38 + barWidth * CGFloat(i) + barWidth / 2,
                                y: geo.size.height - barHeight / 2
                            )
                    }
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
