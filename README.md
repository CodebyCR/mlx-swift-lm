# mlx-swift-lm

`mlx-swift-lm` is a Swift package for can be used to build
tools and applications using large language models (LLMs) and visual language 
models (VLMs) implemented with MLX.

Some key features include:

- Integration with the Hugging Face Hub to easily use thousands of LLMs with a single command.
- Low-rank and full model fine-tuning with support for quantized models.
- Many model architectures for both LLM and VLMs.

- [MLX Swift](https://github.com/ml-explore/mlx-swift) -- Swift version of MLX
- [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples) -- Example applications and tools that use `mlx-swift-lm`

# Using mlx-swift-lm

The MLXLLM, MLXVLM, MLXLMCommon, and MLXEmbedders libraries are available
as Swift Packages.

Add the following dependency to your Package.swift

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", branch: "main"),
```

or use the latest release:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", .upToNextMinor(from: "2.29.1")),
```

Then add one or more libraries to the target as a dependency:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm")
    ]),
```

Alternatively, add `https://github.com/ml-explore/mlx-swift-lm/` to the `Project Dependencies` and set the `Dependency Rule` to `Branch` and `main` in Xcode.

# Quick Start

See also [MLXLMCommon](Libraries/MLXLMCommon).  You can easily use
a wide variety of open weight LLM and VLMs in your code.  You can use
this simplified API:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?")
print(try await session.respond(to: "How about a great place to eat?")
```

Or use the underlying API to control every aspect of the evaluation.

# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations


# MLX Swift Examples

See also [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples) for
a variety of command line tools and applications that make use of these libraries.
