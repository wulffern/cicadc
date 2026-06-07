import SwiftUI

/// The little car that rides the "now" point of a signal trace, rotated to
/// follow the curve's local slope — mirrors the cars in the Python/manim app.
struct CarMarker: View {
    let angle: Angle      // heading along the trace tangent
    let height: CGFloat   // rendered car height in points
    let tint: Color       // fallback / accent colour
    var desaturate: Bool = false  // grayscale car for the digital trace

    var body: some View {
        Group {
            if let ui = CarMarker.image {
                Image(uiImage: ui)
                    .resizable()
                    .scaledToFit()
                    .saturation(desaturate ? 0 : 1)
            } else {
                // Fallback when the asset is missing: a simple triangle marker.
                Triangle()
                    .fill(tint)
            }
        }
        .frame(height: height)
        .rotationEffect(angle)
    }

    /// Loaded once; the car sprite is a chroma-keyed PNG bundled with the app.
    static let image: UIImage? = UIImage(named: "car")
}

private struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.midX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        p.closeSubpath()
        return p
    }
}
