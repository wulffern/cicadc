import SwiftUI

/// Quantization error vs. time strip.
struct ErrorView: View {
    let samples: [(t: Double, v: Double)]
    let tNow: Double
    let visibleDuration: Double

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                Color.black

                // Zero line
                Path { path in
                    path.move(to: CGPoint(x: 0, y: geo.size.height / 2))
                    path.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height / 2))
                }
                .stroke(Color.gray.opacity(0.4), lineWidth: 0.5)

                // Error bars
                Path { path in
                    let tStart = tNow - visibleDuration / 2
                    for s in samples {
                        let x = CGFloat((s.t - tStart) / visibleDuration) * geo.size.width
                        let midY = geo.size.height / 2
                        let errY = midY - CGFloat(s.v) * geo.size.height * 0.45
                        path.move(to: CGPoint(x: x, y: midY))
                        path.addLine(to: CGPoint(x: x, y: errY))
                    }
                }
                .stroke(Color.orange, lineWidth: 2)

                Text("Quant. Error")
                    .font(.caption.bold())
                    .foregroundColor(.orange)
                    .padding(4)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}
