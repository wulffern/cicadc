import SwiftUI

/// Renders a scrolling analog or digital signal panel.
struct SignalView: View {
    let samples: [(t: Double, v: Double)]
    let tNow: Double
    let visibleDuration: Double
    let color: Color
    let title: String
    let isStaircase: Bool  // draw sample-and-hold steps (digital) vs smooth curve

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                // Background grid
                GridOverlay(visibleDuration: visibleDuration, tNow: tNow)

                // Signal path
                if !samples.isEmpty {
                    Path { path in
                        buildPath(path: &path, size: geo.size)
                    }
                    .stroke(color, lineWidth: 2)

                    // "Now" marker dot
                    if let nowY = valueAtNow(size: geo.size) {
                        Circle()
                            .fill(color)
                            .frame(width: 10, height: 10)
                            .position(x: xForT(tNow, width: geo.size.width),
                                      y: nowY)
                    }
                }

                // "Now" horizontal line
                Rectangle()
                    .fill(Color.white.opacity(0.25))
                    .frame(height: 1)
                    .offset(y: geo.size.height / 2)

                Text(title)
                    .font(.caption.bold())
                    .foregroundColor(color)
                    .padding(4)
            }
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func xForT(_ t: Double, width: CGFloat) -> CGFloat {
        let tStart = tNow - visibleDuration / 2
        let fraction = (t - tStart) / visibleDuration
        return CGFloat(fraction) * width
    }

    private func yForV(_ v: Double, height: CGFloat) -> CGFloat {
        CGFloat((1 - (v + 1) / 2)) * height
    }

    private func buildPath(path: inout Path, size: CGSize) {
        if isStaircase {
            buildStaircase(path: &path, size: size)
        } else {
            buildCurve(path: &path, size: size)
        }
    }

    private func buildCurve(path: inout Path, size: CGSize) {
        var started = false
        for s in samples {
            let x = xForT(s.t, width: size.width)
            let y = yForV(s.v, height: size.height)
            if !started {
                path.move(to: CGPoint(x: x, y: y))
                started = true
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
    }

    private func buildStaircase(path: inout Path, size: CGSize) {
        var prev: (t: Double, v: Double)?
        for s in samples {
            let x = xForT(s.t, width: size.width)
            let y = yForV(s.v, height: size.height)
            if let p = prev {
                let prevX = xForT(p.t, width: size.width)
                let prevY = yForV(p.v, height: size.height)
                // Horizontal then vertical (sample-and-hold)
                path.addLine(to: CGPoint(x: x, y: prevY))
                path.addLine(to: CGPoint(x: x, y: y))
            } else {
                path.move(to: CGPoint(x: x, y: y))
            }
            prev = s
        }
    }

    private func valueAtNow(size: CGSize) -> CGFloat? {
        // Interpolate to find value at tNow
        guard samples.count >= 2 else { return nil }
        for i in 1..<samples.count {
            let a = samples[i - 1], b = samples[i]
            if a.t <= tNow && tNow <= b.t {
                let frac = (tNow - a.t) / (b.t - a.t)
                let v = a.v + (b.v - a.v) * frac
                return yForV(v, height: size.height)
            }
        }
        return nil
    }
}

struct GridOverlay: View {
    let visibleDuration: Double
    let tNow: Double
    private let gridLines = 5  // horizontal amplitude lines

    var body: some View {
        GeometryReader { geo in
            // Horizontal amplitude grid
            ForEach(0..<gridLines, id: \.self) { i in
                let y = CGFloat(i) / CGFloat(gridLines - 1) * geo.size.height
                Path { path in
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: geo.size.width, y: y))
                }
                .stroke(Color.gray.opacity(0.3), style: StrokeStyle(lineWidth: 0.5, dash: [4, 4]))
            }
        }
    }
}
