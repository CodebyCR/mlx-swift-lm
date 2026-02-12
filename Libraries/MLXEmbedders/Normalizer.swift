// Copyright © 2026 Apple Inc.

import Foundation
import MLX

public enum Normalizer {

    @inlinable
    public static func l2Normalisation(
        from array: MLXArray,
        axis: Int = -1,
        eps: Float = 1e-8
    ) -> MLXArray {
        let norm = MLXLinalg.norm(array, ord: 2, axis: axis, keepDims: true)
        
        return array / (norm + eps)
    }
}
