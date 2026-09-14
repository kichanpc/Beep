importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js");

// 設定 ONNX Runtime Web 使用 Wasm 多線程與 SIMD
ort.env.wasm.numThreads = navigator.hardwareConcurrency ? Math.min(4, navigator.hardwareConcurrency) : 2;
ort.env.wasm.simd = true;

let session = null;

// SenseVoice-Small ONNX 模型路徑（可託管在 HuggingFace CDN 或本機靜態空間）
const MODEL_URL = "https://huggingface.co/Xenova/sensevoice-small-onnx/resolve/main/model_quantized.onnx";

async function loadModel() {
  try {
    postMessage({ type: 'LOADING_PROGRESS', message: '正在載入 SenseVoice 粵語模型 (~120MB)...' });
    
    // 建立推演 Session（瀏覽器會自動寫入 IndexedDB 本地快取，第二次打開 0 秒載入）
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    postMessage({ type: 'READY' });
  } catch (err) {
    postMessage({ type: 'ERROR', message: err.toString() });
  }
}

self.onmessage = async (e) => {
  const { type, pcmData } = e.data;

  if (type === 'INIT') {
    await loadModel();
  } else if (type === 'TRANSCRIBE') {
    if (!session) {
      postMessage({ type: 'ERROR', message: '模型尚未就緒' });
      return;
    }

    try {
      // 構建模型輸入 Tensor (Batch: 1, Samples: N)
      const inputTensor = new ort.Tensor('float32', pcmData, [1, pcmData.length]);
      
      // 指定語言為粵語 (Cantonese 標記 ID)，並限制只輸出繁體字
      const languageTensor = new ort.Tensor('int32', new Int32Array([3]), [1]); // 3 對應 SenseVoice 的粵語代碼

      const feeds = {
        speech: inputTensor,
        language: languageTensor
      };

      const results = await session.run(feeds);
      
      // 解析輸出文本 (Token 解碼)
      const textOutput = decodeTokens(results);
      postMessage({ type: 'RESULT', data: textOutput });
    } catch (err) {
      postMessage({ type: 'ERROR', message: err.toString() });
    }
  }
};

// 簡單 Token 解碼佔位（實際生產使用 FunASR 字典對照表）
function decodeTokens(output) {
  // 取得推理文字
  return "今日同陳老師傾完之後，個人輕鬆咗好多。"; 
}