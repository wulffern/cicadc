import SwiftUI

/// Quantization error vs. time strip. The band spans ±1 LSB (`fullScale`), so
/// the error keeps the same relative height whatever the bit depth; the trace is
/// a sample-and-hold staircase like the digital panel.
struct ErrorView: View {
    let samples: [(t: Double, v: Double)]
    let tNow: Double
    let visibleDuration: Double
    var fullScale: Double = 1.0   // value (in FS) at the top/bottom of the band (one LSB)

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

                // Error staircase, normalised to ±1 LSB.
                Path { path in
                    buildStaircase(path: &path, size: geo.size)
                }
                .stroke(Color.orange, lineWidth: 2)

                Text("Quant. Error (±1 LSB)")
                    .font(.caption.bold())
                    .foregroundColor(.orange)
                    .padding(4)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func buildStaircase(path: inout Path, size: CGSize) {
        let tStart = tNow - visibleDuration / 2
        let mid = size.height / 2
        let half = size.height * 0.45
        func x(_ t: Double) -> CGFloat { CGFloat((t - tStart) / visibleDuration) * size.width }
        func y(_ v: Double) -> CGFloat {
            let n = (v / max(fullScale, 1e-9)).clamped(to: -1...1)
            return mid - CGFloat(n) * half
        }
        var prev: (t: Double, v: Double)?
        for s in samples {
            if let p = prev {
                path.addLine(to: CGPoint(x: x(s.t), y: y(p.v)))
                path.addLine(to: CGPoint(x: x(s.t), y: y(s.v)))
            } else {
                path.move(to: CGPoint(x: x(s.t), y: y(s.v)))
            }
            prev = s
        }
    }
}
