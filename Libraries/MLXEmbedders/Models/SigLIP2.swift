// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Configurations

/// Configuration for the SigLIP text encoder.
public struct SiglipTextConfiguration: Codable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let maxPositionEmbeddings: Int
    public let hiddenAct: String
    public let layerNormEps: Float

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32000
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 64
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}

/// Configuration for the SigLIP vision encoder.
public struct SiglipVisionConfiguration: Codable, Sendable {
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let hiddenAct: String
    public let layerNormEps: Float

    enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 224
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}

/// Shared configuration for the SigLIP 2 model.
public struct SiglipConfiguration: Codable, Sendable {
    public let textConfig: SiglipTextConfiguration
    public let visionConfig: SiglipVisionConfiguration
    public let modelType: String

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try container.decode(SiglipTextConfiguration.self, forKey: .textConfig)
        visionConfig = try container.decode(SiglipVisionConfiguration.self, forKey: .visionConfig)
        modelType = try container.decode(String.self, forKey: .modelType)
    }
}

// MARK: - Components

private class SiglipMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

private class SiglipTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MultiHeadAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SiglipMLP

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int, eps: Float) {
        self._attention.wrappedValue = MultiHeadAttention(dimensions: hiddenSize, numHeads: numHeads, bias: false)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: eps)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: eps)
        self._mlp.wrappedValue = SiglipMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x + attention(layerNorm1(x), keys: layerNorm1(x), values: layerNorm1(x), mask: mask)
        h = h + mlp(layerNorm2(h))
        return h
    }
}

private class SiglipTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [SiglipTransformerLayer]

    init(numLayers: Int, hiddenSize: Int, numHeads: Int, intermediateSize: Int, eps: Float) {
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            SiglipTransformerLayer(hiddenSize: hiddenSize, numHeads: numHeads, intermediateSize: intermediateSize, eps: eps)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return h
    }
}

// MARK: - Vision Model

private class SiglipVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(_ config: SiglipVisionConfiguration) {
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize)
        )
        let numPatches = (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
        self._positionEmbedding.wrappedValue = Embedding(embeddingCount: numPatches, dimensions: config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var p = patchEmbedding(x)
        let B = p.dim(0)
        let H = p.dim(1)
        let W = p.dim(2)
        let D = p.dim(3)
        p = p.reshaped(B, H * W, D)
        
        let posIds = MLXArray.arange(H * W)
        return p + positionEmbedding(posIds)
    }
}

private class SiglipAttentionMapHead: Module {
    @ModuleInfo(key: "attention") var attention: MultiHeadAttention
    @ParameterInfo(key: "latent") var latent: MLXArray

    init(hiddenSize: Int, numHeads: Int) {
        self._attention.wrappedValue = MultiHeadAttention(dimensions: hiddenSize, numHeads: numHeads, bias: false)
        self._latent.wrappedValue = MLXArray.zeros([1, 1, hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let q = broadcast(latent, to: [B, 1, latent.dim(2)])
        return attention(q, keys: x, values: x)
    }
}

public class SiglipVisionModel: Module {
    @ModuleInfo(key: "embeddings") fileprivate var embeddings: SiglipVisionEmbeddings
    @ModuleInfo(key: "encoder") fileprivate var encoder: SiglipTransformer
    @ModuleInfo(key: "post_layernorm") fileprivate var postLayerNorm: LayerNorm
    @ModuleInfo(key: "head") fileprivate var head: SiglipAttentionMapHead?

    init(_ config: SiglipVisionConfiguration) {
        self._embeddings.wrappedValue = SiglipVisionEmbeddings(config)
        self._encoder.wrappedValue = SiglipTransformer(
            numLayers: config.numHiddenLayers,
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            intermediateSize: config.intermediateSize,
            eps: config.layerNormEps
        )
        self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._head.wrappedValue = SiglipAttentionMapHead(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = encoder(embeddings(x))
        h = postLayerNorm(h)
        if let head {
            h = head(h)
        }
        return h
    }
}

// MARK: - Text Model

private class SiglipTextEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(_ config: SiglipTextConfiguration) {
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._positionEmbedding.wrappedValue = Embedding(embeddingCount: config.maxPositionEmbeddings, dimensions: config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, positionIds: MLXArray? = nil) -> MLXArray {
        let seqLen = inputIds.dim(1)
        let posIds = positionIds ?? MLXArray.arange(seqLen)
        return tokenEmbedding(inputIds) + positionEmbedding(posIds)
    }
}

public class SiglipTextModel: Module {
    @ModuleInfo(key: "embeddings") fileprivate var embeddings: SiglipTextEmbeddings
    @ModuleInfo(key: "encoder") fileprivate var encoder: SiglipTransformer
    @ModuleInfo(key: "final_layer_norm") fileprivate var finalLayerNorm: LayerNorm

    init(_ config: SiglipTextConfiguration) {
        self._embeddings.wrappedValue = SiglipTextEmbeddings(config)
        self._encoder.wrappedValue = SiglipTransformer(
            numLayers: config.numHiddenLayers,
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            intermediateSize: config.intermediateSize,
            eps: config.layerNormEps
        )
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let h = encoder(embeddings(inputIds), mask: attentionMask)
        return finalLayerNorm(h)
    }
}

// MARK: - SigLIP2 Model

/// The SigLIP 2 model implementation for MLX.
public class SigLIP2: Module, EmbeddingModel {
    @ModuleInfo(key: "vision_model") public var visionModel: SiglipVisionModel
    @ModuleInfo(key: "text_model") public var textModel: SiglipTextModel

    public var vocabularySize: Int

    public init(_ config: SiglipConfiguration) {
        self.vocabularySize = config.textConfig.vocabSize
        self._visionModel.wrappedValue = SiglipVisionModel(config.visionConfig)
        self._textModel.wrappedValue = SiglipTextModel(config.textConfig)
        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        if inputs.ndim == 4 {
            // vision input: [Batch, Height, Width, Channels]
            let visionOutput = visionModel(inputs)
            return EmbeddingModelOutput(hiddenStates: visionOutput, pooledOutput: visionOutput.squeezed(axis: 1))
        } else {
            // text input: [Batch, SequenceLength]
            var mask = attentionMask
            if let m = mask, m.ndim == 2 {
                mask = m.expandedDimensions(axes: [1, 2]).log()
            }
            
            let textOutput = textModel(inputs, attentionMask: mask)
            return EmbeddingModelOutput(hiddenStates: textOutput, pooledOutput: textOutput.mean(axis: 1))
        }
    }

    public func sanitize(weights: [String : MLXArray]) -> [String : MLXArray] {
        return weights.reduce(into: [:]) { result, item in
            var key = item.key
            
            // Remove model prefix if present
            if key.hasPrefix("model.") {
                key = String(key.dropFirst(6))
            }
            
            // Map self-attention keys to MLX MultiHeadAttention format
            key = key.replacingOccurrences(of: ".self_attn.q_proj.", with: ".self_attn.query_proj.")
            key = key.replacingOccurrences(of: ".self_attn.k_proj.", with: ".self_attn.key_proj.")
            key = key.replacingOccurrences(of: ".self_attn.v_proj.", with: ".self_attn.value_proj.")
            key = key.replacingOccurrences(of: ".self_attn.out_proj.", with: ".self_attn.out_proj.")
            
            // Map attention map head keys
            key = key.replacingOccurrences(of: ".head.attention.q_proj.", with: ".head.attention.query_proj.")
            key = key.replacingOccurrences(of: ".head.attention.k_proj.", with: ".head.attention.key_proj.")
            key = key.replacingOccurrences(of: ".head.attention.v_proj.", with: ".head.attention.value_proj.")
            key = key.replacingOccurrences(of: ".head.attention.out_proj.", with: ".head.attention.out_proj.")
            
            result[key] = item.value
        }
    }
}
