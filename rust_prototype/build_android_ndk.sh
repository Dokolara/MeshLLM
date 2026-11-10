#!/bin/bash
set -e

# Target Android ABI architectures
TARGETS=("aarch64-linux-android" "x86_64-linux-android")

export ANDROID_NDK_HOME="/home/foor/Android/Sdk/ndk/26.1.10909125"
export TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
export PATH="$TOOLCHAIN/bin:$PATH"
export API_LEVEL=28

echo "=== Android NDK Cross-Compilation Configuration ==="
echo "NDK: $ANDROID_NDK_HOME"
echo "Toolchain: $TOOLCHAIN"

# Compilers
export CC_aarch64_linux_android="$TOOLCHAIN/bin/aarch64-linux-android${API_LEVEL}-clang"
export CXX_aarch64_linux_android="$TOOLCHAIN/bin/aarch64-linux-android${API_LEVEL}-clang++"
export AR_aarch64_linux_android="$TOOLCHAIN/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"

export CC_x86_64_linux_android="$TOOLCHAIN/bin/x86_64-linux-android${API_LEVEL}-clang"
export CXX_x86_64_linux_android="$TOOLCHAIN/bin/x86_64-linux-android${API_LEVEL}-clang++"
export AR_x86_64_linux_android="$TOOLCHAIN/bin/llvm-ar"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$CC_x86_64_linux_android"

echo "Environment ready for cargo build --target aarch64-linux-android"
