
"use client"
import { useEffect, useRef, useState } from 'react';
import Script from 'next/script';
import Head from 'next/head';

export default function Page() {
  const webcamRef = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [mode, setMode] = useState('webcam');
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState('Select mode and click start to begin detection');
  const [intervalId, setIntervalId] = useState(null);

  useEffect(() => {
    switchMode('webcam');
  }, []);

  const updateStatus = (msg) => setStatus(msg);

  const loadModel = async () => {
    if (model) return;
    updateStatus('Loading model...');
    try {
      await window.tf.setBackend('webgl');
      const loadedModel = await window.tf.loadGraphModel('/yolov8n_web_model/model.json');
      setModel(loadedModel);
      updateStatus('Model loaded successfully!');
    } catch (err) {
      updateStatus('Error loading model: ' + err.message);
    }
  };

  const handleStart = async () => {
    await loadModel();
    await setupWebcam();
    startDetection();
  };

  const handleStop = () => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
      updateStatus('Webcam detection stopped');
    }
  };

  const switchMode = (newMode) => {
    setMode(newMode);
    updateStatus(`${newMode === 'webcam' ? 'Webcam' : 'Image'} mode selected`);
    clearCanvas();
    handleStop();
  };

  const setupWebcam = async () => {
    if (navigator.mediaDevices?.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamRef.current.srcObject = stream;
        updateStatus('Webcam initialized');
      } catch (err) {
        updateStatus('Webcam error: ' + err.message);
      }
    }
  };

  const startDetection = () => {
    const id = setInterval(() => {
      
    }, 100);
    setIntervalId(id);
    updateStatus('Webcam detection started');
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  return (
    <>
      <Head>
        <title>YOLOv8 Object Detection</title>
        <Script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js" strategy="beforeInteractive" />
      </Head>
      <div id="header">
        <h1>YOLOv8 Object Detection</h1>
      </div>

      <div id="controls">
        <div className="mode-selector">
          <label>
            <input type="radio" name="mode" value="webcam" checked={mode === 'webcam'} onChange={() => switchMode('webcam')} /> Webcam
          </label>
          <label>
            <input type="radio" name="mode" value="image" checked={mode === 'image'} onChange={() => switchMode('image')} /> Image Upload
          </label>
        </div>

        {mode === 'webcam' ? (
          <>
            <button onClick={handleStart}>Start Webcam Detection</button>
            <button onClick={handleStop}>Stop Detection</button>
          </>
        ) : (
          <>
            <input type="file" ref={fileInputRef} accept="image/*" />
            <button>Detect Objects</button>
          </>
        )}

        <div id="status">{status}</div>
      </div>

      <div id="main">
        <video ref={webcamRef} autoPlay playsInline width="640" height="480" style={{ display: mode === 'webcam' ? 'block' : 'none' }} />
        <img ref={imageRef} alt="Uploaded image" style={{ display: mode === 'image' ? 'block' : 'none' }} />
        <canvas ref={canvasRef} id="outputCanvas"></canvas>
      </div>

      <style jsx>{`
        body,
        html {
          height: 100%;
          margin: 0;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          font-family: Arial, sans-serif;
          background: rgba(255, 255, 255, 0.9);
        }

        #header {
          position: absolute;
          z-index: 2;
          top: 0px;
          width: 100%;
          text-align: center;
          background: rgba(255, 255, 255, 0.9);
          padding: 10px 0;
        }

        #controls {
          position: absolute;
          top: 120px;
          z-index: 3;
          text-align: center;
          background: rgba(255, 255, 255, 0.9);
          padding: 10px;
          border-radius: 5px;
        }

        #main {
          position: relative;
          width: 640px;
          height: 480px;
          margin-top: 50px;
        }

        #outputCanvas {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: 2;
          pointer-events: none;
        }
      `}</style>
    </>
  );
}
