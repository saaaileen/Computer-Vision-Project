'use client';

import { useRef, useCallback, useState, useEffect } from 'react';
import Webcam from 'react-webcam';

interface DetectionResult {
  labels: string[];
  message?: string;
  error?: string;
}

export default function SignLanguageDetector() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [predictions, setPredictions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [history, setHistory] = useState<string[]>([]);

  // Check API connection on mount
  useEffect(() => {
    checkApiConnection();
  }, []);

  const checkApiConnection = async () => {
    try {
      setApiStatus('checking');
      const response = await fetch('http://localhost:5000/predict', {
        method: 'OPTIONS',
      }).catch(() => null);
      setApiStatus(response ? 'connected' : 'disconnected');
    } catch {
      setApiStatus('disconnected');
    }
  };

  const capture = useCallback(async () => {
    if (!webcamRef.current) return;

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      setIsLoading(true);

      // Convert base64 to blob
      const response = await fetch(imageSrc);
      const blob = await response.blob();

      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');

      const result = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data: DetectionResult = await result.json();

      if (data.labels && data.labels.length > 0) {
        setPredictions(data.labels);
        setHistory((prev) => {
          const updated = [data.labels[0], ...prev];
          return updated.slice(0, 10); // Keep last 10
        });
      } else if (data.message) {
        setPredictions([]);
      }

      if (data.error) {
        console.error('API Error:', data.error);
      }
    } catch (error) {
      console.error('Error:', error);
      setApiStatus('disconnected');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleStart = () => {
    setIsRunning(true);
    setPredictions([]);
    setHistory([]);
    // Capture every 500ms for real-time detection
    intervalRef.current = setInterval(capture, 500);
  };

  const handleStop = () => {
    setIsRunning(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user',
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-3 tracking-tight">
            Sign Language Detector
          </h1>
          <p className="text-slate-400 text-lg">
            Real-time ASL detection using AI and computer vision
          </p>
          <div className="mt-4 flex justify-center gap-2">
            <div
              className={`px-4 py-2 rounded-full text-sm font-semibold flex items-center gap-2 ${
                apiStatus === 'connected'
                  ? 'bg-green-900 text-green-200'
                  : apiStatus === 'checking'
                    ? 'bg-yellow-900 text-yellow-200'
                    : 'bg-red-900 text-red-200'
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  apiStatus === 'connected'
                    ? 'bg-green-400'
                    : apiStatus === 'checking'
                      ? 'bg-yellow-400'
                      : 'bg-red-400'
                }`}
              />
              API Status:{' '}
              {apiStatus === 'checking'
                ? 'Checking...'
                : apiStatus === 'connected'
                  ? 'Connected'
                  : 'Disconnected'}
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Webcam Section */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800 rounded-2xl overflow-hidden shadow-2xl border border-slate-700">
              <div className="relative bg-black aspect-video">
                <Webcam
                  ref={webcamRef}
                  videoConstraints={videoConstraints}
                  screenshotFormat="image/jpeg"
                  className="w-full h-full object-cover"
                />

                {/* Overlay Info */}
                {isRunning && (
                  <div className="absolute top-4 left-4 flex items-center gap-2">
                    <div className="flex items-center gap-2 bg-red-900/80 px-4 py-2 rounded-lg backdrop-blur">
                      <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
                      <span className="text-white font-semibold text-sm">
                        Live Detection
                      </span>
                    </div>
                  </div>
                )}

                {/* Current Prediction */}
                {predictions.length > 0 && (
                  <div className="absolute bottom-4 left-4 right-4">
                    <div className="bg-blue-600/90 backdrop-blur rounded-lg p-4 border-2 border-blue-400">
                      <p className="text-white text-xs font-semibold tracking-widest uppercase">
                        Detected Signs
                      </p>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {predictions.map((pred, idx) => (
                          <span
                            key={idx}
                            className="bg-blue-500 text-white px-4 py-2 rounded-full font-bold text-lg"
                          >
                            {pred.toUpperCase()}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {isLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/30 backdrop-blur-sm">
                    <div className="flex flex-col items-center">
                      <div className="w-12 h-12 rounded-full border-4 border-slate-600 border-t-blue-500 animate-spin mb-3" />
                      <span className="text-white font-semibold">Processing...</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="bg-gradient-to-r from-slate-700 to-slate-800 p-6">
                <div className="flex gap-4">
                  <button
                    onClick={handleStart}
                    disabled={isRunning || apiStatus === 'disconnected'}
                    className={`flex-1 py-3 px-6 rounded-xl font-bold text-lg transition-all duration-200 ${
                      isRunning
                        ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                        : apiStatus === 'disconnected'
                          ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                          : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-green-900/50'
                    }`}
                  >
                    ▶ Start Detection
                  </button>
                  <button
                    onClick={handleStop}
                    disabled={!isRunning}
                    className={`flex-1 py-3 px-6 rounded-xl font-bold text-lg transition-all duration-200 ${
                      !isRunning
                        ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                        : 'bg-red-600 hover:bg-red-700 text-white shadow-lg hover:shadow-red-900/50'
                    }`}
                  >
                    ⏹ Stop
                  </button>
                  <button
                    onClick={checkApiConnection}
                    className="py-3 px-6 rounded-xl font-bold text-lg bg-slate-600 hover:bg-slate-700 text-white transition-all duration-200"
                  >
                    🔄 Retry
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar Panel */}
          <div className="space-y-6">
            {/* Instructions */}
            <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                📋 How to Use
              </h3>
              <ol className="space-y-3 text-slate-300 text-sm">
                <li className="flex gap-3">
                  <span className="font-bold text-blue-400 flex-shrink-0">1</span>
                  <span>Make sure Flask API is running on port 5000</span>
                </li>
                <li className="flex gap-3">
                  <span className="font-bold text-blue-400 flex-shrink-0">2</span>
                  <span>Click "Start Detection" to begin</span>
                </li>
                <li className="flex gap-3">
                  <span className="font-bold text-blue-400 flex-shrink-0">3</span>
                  <span>Show hand signs to the camera</span>
                </li>
                <li className="flex gap-3">
                  <span className="font-bold text-blue-400 flex-shrink-0">4</span>
                  <span>See predictions in real-time</span>
                </li>
                <li className="flex gap-3">
                  <span className="font-bold text-blue-400 flex-shrink-0">5</span>
                  <span>Click "Stop" to end detection</span>
                </li>
              </ol>
            </div>

            {/* Detection History */}
            <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                📜 Detection History
              </h3>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {history.length === 0 ? (
                  <p className="text-slate-500 text-sm italic">
                    No detections yet. Start detection to see history.
                  </p>
                ) : (
                  history.map((item, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-700/50 hover:bg-slate-700 rounded-lg px-4 py-3 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-xs">
                          #{history.length - idx}
                        </span>
                        <span className="bg-blue-600 text-white px-3 py-1 rounded-full font-bold text-sm">
                          {item.toUpperCase()}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Info Card */}
            <div className="bg-gradient-to-br from-blue-900 to-blue-800 rounded-2xl p-6 border border-blue-700 shadow-xl">
              <p className="text-blue-100 text-sm leading-relaxed">
                💡 This detector uses MediaPipe for hand landmark tracking and an SVM classifier for sign recognition.
              </p>
            </div>
          </div>
        </div>
      </div>
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}
