'use client'
import { useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';


const TensorFlowComponent = () => {
  const [tf, setTf] = useState(null);
  const [model, setModel] = useState(null);
  const [currentMode, setCurrentMode] = useState('webcam');
  const [status, setStatus] = useState('Select mode and click start to begin detection');
  const [isDetecting, setIsDetecting] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  
  const webcamRef = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const streamRef = useRef(null);
  // Add a ref to store the actual model instance
  const modelRef = useRef(null);

  const classNames = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
  };

  const TARGET_WIDTH = 640;
  const TARGET_HEIGHT = 640;

  
  useEffect(() => {
    const loadTensorFlow = async () => {
      try {
        const tfModule = await import('@tensorflow/tfjs');
        setTf(tfModule);
      } catch (error) {
        console.error('Failed to load TensorFlow.js:', error);
        setStatus('Failed to load TensorFlow.js');
      }
    };
    loadTensorFlow();
  }, []);


  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const loadModel = async () => {
    if (modelRef.current || !tf) return modelRef.current;
    
    setStatus('Loading model...');
    try {
      await tf.setBackend('webgl');
      const loadedModel = await tf.loadGraphModel('/yolov8n_web_model/model.json');
      modelRef.current = loadedModel;
      setModel(loadedModel);
      setStatus('Model loaded successfully!');
      return loadedModel;
    } catch (error) {
      setStatus('Error loading model: ' + error.message);
      console.error('Model loading error:', error);
      throw error;
    }
  };

  const runModel = async (tensor) => {
    // Ensure model is loaded and available
    let currentModel = modelRef.current;
    if (!currentModel) {
      currentModel = await loadModel();
    }
    
    if (!currentModel || typeof currentModel.predict !== 'function') {
      throw new Error('Model not loaded properly');
    }
    
    return currentModel.predict(tensor);
  };

  const setupWebcam = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = stream;
        if (webcamRef.current) {
          webcamRef.current.srcObject = stream;
        }
        setStatus('Webcam initialized');
      } catch (error) {
        setStatus('Error accessing webcam: ' + error.message);
        console.error('Webcam setup error:', error);
      }
    } else {
      setStatus('getUserMedia is not supported');
    }
  };

  const webcamToTensor = async (videoElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = TARGET_WIDTH;
    canvas.height = TARGET_HEIGHT;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    ctx.drawImage(videoElement, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    const imageData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    const tensor = tf.browser.fromPixels(imageData);

    return tf.cast(tensor, 'float32').div(tf.scalar(255)).expandDims(0);
  };

  const imageToTensor = async (imageElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = TARGET_WIDTH;
    canvas.height = TARGET_HEIGHT;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    ctx.drawImage(imageElement, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    const imageData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    const tensor = tf.browser.fromPixels(imageData);

    return tf.cast(tensor, 'float32').div(tf.scalar(255)).expandDims(0);
  };

  const calculateBoundingBoxes = (transRes) => {
    const [xCenter, yCenter, width, height] = [
      transRes.slice([0, 0, 0], [-1, -1, 1]),
      transRes.slice([0, 0, 1], [-1, -1, 1]),
      transRes.slice([0, 0, 2], [-1, -1, 1]),
      transRes.slice([0, 0, 3], [-1, -1, 1])
    ];

    const topLeftX = tf.sub(xCenter, tf.div(width, 2));
    const topLeftY = tf.sub(yCenter, tf.div(height, 2));
    return tf.concat([topLeftX, topLeftY, width, height], 2).squeeze();
  };

  const calculateScoresAndLabels = (transRes, classNames) => {
    const rawScores = transRes.slice([0, 0, 4], [-1, -1, Object.keys(classNames).length]).squeeze(0);
    return [rawScores.max(1), rawScores.argMax(1)];
  };

  const extractSelectedPredictions = (indices, boxes, labels, classNames) => {
    return indices.map(i => {
      const box = boxes.slice([i, 0], [1, -1]).squeeze().arraySync();
      const label = labels.slice([i], [1]).arraySync()[0];
      return { box, label: classNames[label] };
    });
  };

  const processPredictions = (predictions, classNames) => {
    return tf.tidy(() => {
      const transRes = predictions.transpose([0, 2, 1]);
      const boxes = calculateBoundingBoxes(transRes);
      const [scores, labels] = calculateScoresAndLabels(transRes, classNames);

      const indices = tf.image.nonMaxSuppression(boxes, scores, predictions.shape[2], 0.45, 0.2).arraySync();
      return extractSelectedPredictions(indices, boxes, labels, classNames);
    });
  };

  const drawBoundingBoxes = async (imageElement, detections) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width = imageElement.width || imageElement.videoWidth;
    canvas.height = imageElement.height || imageElement.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const resizeScale = Math.min(TARGET_WIDTH / canvas.width, TARGET_HEIGHT / canvas.height);
    const dx = (TARGET_WIDTH - canvas.width * resizeScale) / 2;
    const dy = (TARGET_HEIGHT - canvas.height * resizeScale) / 2;

    detections.forEach(({ box, label }) => {
      let [topLeftX, topLeftY, width, height] = box;
      topLeftX = topLeftX / resizeScale - dx / resizeScale;
      topLeftY = topLeftY / resizeScale - dy / resizeScale;
      width /= resizeScale;
      height /= resizeScale;

      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(topLeftX, topLeftY, width, height);
      ctx.fillStyle = 'red';
      ctx.font = '20px Arial';
      ctx.fillText(label, topLeftX, topLeftY - 7);
    });
  };

  const processWebcamFrame = async () => {
    if (!webcamRef.current || !tf) return;
    
    try {
      const tensor = await webcamToTensor(webcamRef.current);
      const startTime = performance.now();
      const predictions = await runModel(tensor);
      const endTime = performance.now();
      const inferenceTime = endTime - startTime;
      
      const detections = processPredictions(predictions, classNames);
      await drawBoundingBoxes(webcamRef.current, detections);
      
      setStatus(`Inference Time: ${inferenceTime.toFixed(2)} ms | Objects: ${detections.length}`);
    } catch (error) {
      setStatus('Error processing frame: ' + error.message);
      console.error('Frame processing error:', error);
    }
  };

  const startWebcamDetection = async () => {
    try {
      await loadModel(); // Ensure model is loaded before starting
      await setupWebcam();
      
      if (intervalRef.current) return;
      intervalRef.current = setInterval(processWebcamFrame, 100);
      setIsDetecting(true);
      setStatus('Webcam detection started');
    } catch (error) {
      setStatus('Error starting webcam detection: ' + error.message);
      console.error('Webcam detection start error:', error);
    }
  };

  const stopWebcamDetection = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      setIsDetecting(false);
      setStatus('Webcam detection stopped');
    }
  };

  const processImageFile = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = e.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const detectObjectsInImage = async () => {
    if (!selectedFile || !tf) {
      setStatus('Please select an image file');
      return;
    }

    try {
      setStatus('Processing image...');
      const img = await processImageFile(selectedFile);
      
      if (imageRef.current) {
        imageRef.current.src = img.src;
      }
      
      const tensor = await imageToTensor(img);
      const startTime = performance.now();
      const predictions = await runModel(tensor);
      const endTime = performance.now();
      const inferenceTime = endTime - startTime;
      
      const detections = processPredictions(predictions, classNames);
      await drawBoundingBoxes(imageRef.current, detections);
      
      setStatus(`Inference Time: ${inferenceTime.toFixed(2)} ms | Objects detected: ${detections.length}`);
    } catch (error) {
      setStatus('Error processing image: ' + error.message);
      console.error('Image processing error:', error);
    }
  };

  const handleModeChange = (mode) => {
    setCurrentMode(mode);
    if (mode === 'image') {
      stopWebcamDetection();
    }
    
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setStatus(`${mode === 'webcam' ? 'Webcam' : 'Image'} mode selected`);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };

  if (!tf) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div>Loading TensorFlow.js...</div>
      </div>
    );
  }

  return (
    <div className="yolo-container">
      <div className="yolo-header">
        <h1>YOLOv8 Object Detection</h1>
      </div>
      
      <div className="yolo-controls">
        <div className="yolo-mode-selector">
          <label>
            <input
              type="radio"
              name="mode"
              value="webcam"
              checked={currentMode === 'webcam'}
              onChange={(e) => handleModeChange(e.target.value)}
            />
            Webcam
          </label>
          <label>
            <input
              type="radio"
              name="mode"
              value="image"
              checked={currentMode === 'image'}
              onChange={(e) => handleModeChange(e.target.value)}
            />
            Image Upload
          </label>
        </div>
        
        {currentMode === 'webcam' && (
          <div className="yolo-webcam-controls">
            <button onClick={startWebcamDetection} disabled={isDetecting}>
              Start Webcam Detection
            </button>
            <button onClick={stopWebcamDetection} disabled={!isDetecting}>
              Stop Detection
            </button>
          </div>
        )}
        
        {currentMode === 'image' && (
          <div className="yolo-image-controls">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="yolo-file-input"
            />
            <button onClick={detectObjectsInImage} disabled={!selectedFile}>
              Detect Objects
            </button>
          </div>
        )}
        
        <div className="yolo-status">{status}</div>
      </div>
      
      <div className="yolo-main">
        <video
          ref={webcamRef}
          autoPlay
          playsInline
          width="640"
          height="480"
          style={{ display: currentMode === 'webcam' ? 'block' : 'none' }}
        />
        <img
          ref={imageRef}
          alt="Uploaded image"
          style={{ display: currentMode === 'image' ? 'block' : 'none' }}
        />
        <canvas ref={canvasRef} className="yolo-output-canvas" />
      </div>

      <style jsx>{`
        .yolo-container {
          height: 100vh;
          margin: 100;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          font-family: Arial, sans-serif;
        }

        .yolo-header {
          position: absolute;
          z-index: 2;
          top: 0;
          width: 100%;
          text-align: center;
          background: rgba(4, 3, 3, 0.73);
          padding: 10px 0;
        }

        .yolo-controls {
          position: absolute;
          top: 120px;
          z-index: 3;
          text-align: center;
          background: rgba(166, 161, 161, 1);
          padding: 10px;
          border-radius: 5px;
        }

        .yolo-main {
          position: relative;
          width: 640px;
          height: 480px;
          margin-top: 50px;
        }

        .yolo-main video,
        .yolo-main img,
        .yolo-output-canvas {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
        }

        .yolo-main img {
          object-fit: contain;
        }

        .yolo-output-canvas {
          z-index: 2;
          pointer-events: none;
        }

        .yolo-controls button {
          font-size: 16px;
          background-color: #000;
          color: #fff;
          cursor: pointer;
          margin: 5px;
          padding: 10px 20px;
          border: none;
          border-radius: 5px;
        }

        .yolo-controls button:hover {
          background-color: #333;
        }

        .yolo-controls button:disabled {
          background-color: #666;
          cursor: not-allowed;
        }

        .yolo-file-input {
          margin: 10px;
        }

        .yolo-mode-selector {
          margin: 10px 0;
        }

        .yolo-mode-selector label {
          margin: 0 10px;
          font-weight: bold;
        }

        .yolo-status {
          margin-top: 10px;
          padding: 10px;
          background: hsla(0, 0%, 61%, 0.90);
          border-radius: 5px;
          min-height: 20px;
        }

        .yolo-webcam-controls,
        .yolo-image-controls {
          margin: 10px 0;
        }

        body,
        html {
          height: 100%;
          margin: 0;
          padding: 0;
        }
      `}</style>
    </div>
  );
};

const YOLOv8App = dynamic(() => Promise.resolve(TensorFlowComponent), {
  ssr: false,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>YOLOv8 Object Detection On Browser</title>
        <meta name="description" content="YOLOv8 Object Detection using TensorFlow.js" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <YOLOv8App />
    </>
  );
}