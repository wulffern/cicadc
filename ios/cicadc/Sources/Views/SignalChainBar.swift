import SwiftUI

/// Shows the ADC signal chain as a row of labeled blocks.
struct SignalChainBar: View {
    let params: ADCParameters

    private var blocks: [(label: String, active: Bool, color: Color)] {
        let noiseActive = params.noiseAmplitude > 0
        let filterActive = params.avgTaps > 1
        let sdActive = params.adcType != .nyquist
        return [
            ("Analog", true, .green),
            ("Noise", noiseActive, .yellow),
            (params.adcType.rawValue, true, .blue),
            ("Filter ×\(params.avgTaps)", filterActive, .purple),
            ("Digital", true, .cyan),
        ]
    }

    var body: some View {
        HStack(spacing: 4) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { idx, block in
                if idx > 0 {
                    Image(systemName: "arrow.right")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
                Text(block.label)
                    .font(.caption2.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(block.active ? block.color.opacity(0.25) : Color.gray.opacity(0.1))
                    .foregroundColor(block.active ? block.color : .gray)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity)
        .background(Color(.systemBackground).opacity(0.8))
    }
}
