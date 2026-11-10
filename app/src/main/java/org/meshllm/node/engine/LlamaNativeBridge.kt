package org.meshllm.node.engine

import android.util.Log
import java.io.File

/**
 * JNI bridge to the standard compiled llama.cpp C++ native engine.
 * Handles model loading into RAM, tokenization, layer forward passes,
 * and memory tracking.
 */
object LlamaNativeBridge {
    private const val TAG = "LlamaNativeBridge"

    var isNativeLoaded: Boolean = false
        private set

    init {
        try {
            System.loadLibrary("llama-mesh")
            isNativeLoaded = nativeInit()
            Log.i(TAG, "Successfully loaded native libllama-mesh.so (isNativeLoaded=$isNativeLoaded)")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "Native library libllama-mesh.so not found or running in JVM unit test: ${e.message}")
            isNativeLoaded = false
        } catch (e: Throwable) {
            Log.e(TAG, "Error initializing native llama.cpp: ${e.message}", e)
            isNativeLoaded = false
        }
    }

    fun loadModel(file: File, startLayer: Int = 0, endLayer: Int = -1, nThreads: Int = 0): Int {
        if (!isNativeLoaded) return -99
        return nativeLoadModel(file.absolutePath, startLayer, endLayer, nThreads)
    }

    fun tokenize(text: String): IntArray {
        if (!isNativeLoaded) return intArrayOf()
        return nativeTokenize(text)
    }

    fun decodeToken(tokenId: Int): String {
        if (!isNativeLoaded) return ""
        return nativeDecodeToken(tokenId)
    }

    fun evalToken(tokenId: Int, pos: Int, isOutput: Boolean): Int {
        if (!isNativeLoaded) return -99
        return nativeEvalToken(tokenId, pos, isOutput)
    }

    fun evalTokensBatch(tokens: IntArray, startPos: Int, sampleLast: Boolean): Int {
        if (!isNativeLoaded) return -99
        return nativeEvalTokensBatch(tokens, startPos, sampleLast)
    }

    fun prefillPipeline(
        tokens: IntArray?,
        incomingHidden: FloatArray?,
        isInputBlock: Boolean,
        isOutputBlock: Boolean
    ): NativePipelineResult? {
        if (!isNativeLoaded) return null
        return nativePrefillPipeline(tokens, incomingHidden, isInputBlock, isOutputBlock)
    }

    fun forwardPipeline(
        token: Int,
        pos: Int,
        incomingHidden: FloatArray?,
        isInputBlock: Boolean,
        isOutputBlock: Boolean
    ): NativePipelineResult? {
        if (!isNativeLoaded) return null
        return nativeForwardPipeline(token, pos, incomingHidden, isInputBlock, isOutputBlock)
    }

    fun unloadModel(): Boolean {
        if (!isNativeLoaded) return false
        return nativeUnloadModel()
    }

    fun getAllocatedRamBytes(): Long {
        if (!isNativeLoaded) return 0L
        return nativeGetAllocatedRamBytes()
    }

    fun isModelLoaded(): Boolean {
        if (!isNativeLoaded) return false
        return nativeIsModelLoaded()
    }

    fun resetContext() {
        if (!isNativeLoaded) return
        nativeResetContext()
    }

    fun isGpuAccelerated(): Boolean {
        if (!isNativeLoaded) return false
        return nativeIsGpuAccelerated()
    }

    fun isEog(tokenId: Int): Boolean {
        if (!isNativeLoaded) return tokenId == 2 || tokenId == 0
        return nativeIsEog(tokenId)
    }

    // Native JNI declarations
    private external fun nativeInit(): Boolean
    private external fun nativeLoadModel(modelPath: String, startLayer: Int, endLayer: Int, nThreads: Int): Int
    private external fun nativeTokenize(text: String): IntArray
    private external fun nativeDecodeToken(tokenId: Int): String
    private external fun nativeEvalToken(tokenId: Int, pos: Int, isOutput: Boolean): Int
    private external fun nativeEvalTokensBatch(tokens: IntArray, startPos: Int, sampleLast: Boolean): Int
    private external fun nativePrefillPipeline(tokens: IntArray?, incomingHidden: FloatArray?, isInputBlock: Boolean, isOutputBlock: Boolean): NativePipelineResult?
    private external fun nativeForwardPipeline(token: Int, pos: Int, incomingHidden: FloatArray?, isInputBlock: Boolean, isOutputBlock: Boolean): NativePipelineResult?
    private external fun nativeUnloadModel(): Boolean
    private external fun nativeGetAllocatedRamBytes(): Long
    private external fun nativeIsModelLoaded(): Boolean
    private external fun nativeResetContext()
    private external fun nativeIsGpuAccelerated(): Boolean
    private external fun nativeIsEog(tokenId: Int): Boolean
}
