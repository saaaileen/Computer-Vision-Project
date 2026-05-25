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

  const statusCopy = {
    checking: {
      label: 'Checking backend',
      tone: 'border-amber-500/30 bg-amber-500/10 text-amber-100',
      dot: 'bg-amber-300',
    },
    connected: {
      label: 'API connected',
      tone: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100',
      dot: 'bg-emerald-300',
    },
    disconnected: {
      label: 'API offline',
      tone: 'border-rose-500/30 bg-rose-500/10 text-rose-100',
      dot: 'bg-rose-300',
    },
  }[apiStatus];

  const quickStats = [
    { label: 'Stream', value: isRunning ? 'Live' : 'Idle' },
    { label: 'Latest', value: predictions[0]?.toUpperCase() ?? '—' },
    { label: 'History', value: `${history.length} saved` },
  ];

  const steps = [
    'Start the Flask API on port 5000.',
    'Allow camera access when prompted.',
    'Press Start Detection and show a sign.',
    'Watch the live result and recent history update.',
  ];

  // Check API connection on mount
  useEffect(() => {
    checkApiConnection();
  }, []);

  const checkApiConnection = async () => {
    try {
      setApiStatus('checking');
      const response = await fetch('https://computer-vision-project-nhm5.onrender.com/predict', {
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
    <main className="w-full h-full px-4 py-6 overflow-scroll">
      <div className="mx-auto flex h-[calc(100vh-3rem)] w-full max-w-7xl flex-col gap-6 ">
        <section className='grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(320px,0.7fr)] flex-1'>
          <div className="flex flex-col">
            <div className="px-6 py-8 sm:px-8 sm:py-9 lg:px-10">

              <div className="gap-8 lg:grid-cols-[1.1fr_0.9fr] lg:items-end">
                <div className="max-w-3xl">
                  <div className="mb-5 flex flex-wrap items-center gap-3 text-[0.68rem] uppercase tracking-[0.26em] text-(--muted)">
                    <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono">
                      Live ASL recognition
                    </span>
                    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono ${statusCopy.tone}`}>
                      <span className={`h-2 w-2 rounded-full ${statusCopy.dot} animate-pulse`} />
                      {statusCopy.label}
                    </span>
                  </div>

                  <h1 className="max-w-2xl text-3xl font-semibold tracking-tight text-white sm:text-4xl lg:text-[3.5rem]">
                    Computer Vision Project
                  </h1>
                  <p className="mt-4 max-w-2xl text-base leading-7 text-(--muted) sm:text-[1.05rem]">
                    Keep the webcam central, surface the latest prediction without noise, and make the backend state obvious at a glance.
                  </p>

                  <div className="mt-7 flex flex-wrap gap-3">
                    <button
                      onClick={handleStart}
                      disabled={isRunning || apiStatus === 'disconnected'}
                      className={`inline-flex items-center justify-center rounded-full px-5 py-3 text-sm font-semibold transition-all duration-200 ${isRunning || apiStatus === 'disconnected'
                        ? 'cursor-not-allowed border border-white/10 bg-white/5 text-slate-500'
                        : 'border border-[#7fb7ff]/30 bg-[#8dd6ff] text-slate-950 shadow-[0_12px_30px_rgba(99,168,255,0.28)] hover:-translate-y-0.5 hover:bg-white'
                        }`}
                    >
                      Start detection
                    </button>
                    <button
                      onClick={handleStop}
                      disabled={!isRunning}
                      className={`inline-flex items-center justify-center rounded-full px-5 py-3 text-sm font-semibold transition-all duration-200 ${!isRunning
                        ? 'cursor-not-allowed border border-white/10 bg-white/5 text-slate-500'
                        : 'border border-white/10 bg-white/10 text-white hover:-translate-y-0.5 hover:bg-white/15'
                        }`}
                    >
                      Stop
                    </button>
                    <button
                      onClick={checkApiConnection}
                      className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition-all duration-200 hover:-translate-y-0.5 hover:bg-white/10"
                    >
                      Retry backend
                    </button>
                  </div>
                </div>

              </div>
            </div>

            <div className="overflow-hidden rounded-3xl border border-white/10 bg-(--surface) shadow-[0_18px_50px_rgba(2,8,23,0.24)] flex flex-col flex-1">
              <div className="border-b border-white/10 px-5 py-4 sm:px-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-[0.68rem] uppercase tracking-[0.28em] text-(--muted)">
                      Camera feed
                    </p>
                    <h2 className="mt-1 text-lg font-semibold text-white">
                      Web camera stage
                    </h2>
                  </div>
                  <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-mono uppercase tracking-[0.22em] text-(--muted)">
                    500 ms capture interval
                  </div>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-center px-4">
                <div className="w-full max-w-4xl rounded-2xl border border-white/10 bg-[rgba(8,17,28,0.92)] px-4 py-4">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-[0.68rem] uppercase tracking-[0.18em] text-(--muted)">Detected</div>
                      <div className="mt-2 text-3xl sm:text-4xl font-bold text-white leading-tight">
                        {predictions[0]?.toUpperCase() ?? '—'}
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {predictions.length > 1
                          ? predictions.slice(1, 6).map((pred, idx) => (
                            <span
                              key={`${pred}-${idx}`}
                              className="rounded-full border border-[#8dd6ff]/20 bg-[#8dd6ff]/80 px-3 py-1 text-xs font-semibold tracking-[0.08em] text-slate-950"
                            >
                              {pred.toUpperCase()}
                            </span>
                          ))
                          : (
                            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-(--muted)">
                              No additional predictions
                            </span>
                          )}
                      </div>
                    </div>

                    <div className="hidden sm:flex flex-col items-end">
                      <div className="text-sm text-(--muted)">Latest</div>
                      <div className="mt-1 text-lg font-semibold text-white">{predictions[0]?.toUpperCase() ?? '—'}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-4 sm:p-6 flex flex-col flex-1">
                <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-[#050b12] flex-1">
                  <div className="relative h-full">
                    <Webcam
                      ref={webcamRef}
                      videoConstraints={videoConstraints}
                      screenshotFormat="image/jpeg"
                      className="h-full w-full object-cover"
                      mirrored={true}
                    />

                    <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(5,11,18,0.02),rgba(5,11,18,0.1)_52%,rgba(5,11,18,0.3))]" />

                    {isRunning && (
                      <div className="absolute left-4 top-4 rounded-full border border-rose-400/30 bg-rose-500/15 px-3 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-rose-100 backdrop-blur-md">
                        <span className="mr-2 inline-block h-2 w-2 rounded-full bg-rose-300 animate-pulse" />
                        Live detection
                      </div>
                    )}

                  </div>



                  {isLoading && (
                    <div className="absolute inset-0 grid place-items-center bg-slate-950/45 backdrop-blur-sm">
                      <div className="rounded-2xl border border-white/10 bg-[rgba(8,17,28,0.9)] px-5 py-4 text-center">
                        <div className="mx-auto mb-3 h-10 w-10 rounded-full border-4 border-white/15 border-t-[#8dd6ff] animate-spin" />
                        <p className="text-sm font-medium text-white">Analyzing frame</p>
                        <p className="mt-1 text-xs text-slate-300">Sending the current image to the classifier.</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="border-t border-white/10 bg-(--surface-strong) px-5 py-4 sm:px-6">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="text-sm text-slate-300">
                      Real-time capture is only active while detection is running.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-mono uppercase tracking-[0.22em] text-slate-300">
                        MediaPipe landmarks
                      </span>
                      <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-mono uppercase tracking-[0.22em] text-slate-300">
                        SVM classifier
                      </span>
                      <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-mono uppercase tracking-[0.22em] text-slate-300">
                        Webcam input
                      </span>
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </div>




          <div className="grid gap-6">
            <aside className="rounded-3xl border border-white/10 bg-(--surface) p-6 shadow-[0_18px_50px_rgba(2,8,23,0.22)]">
              <p className="text-[0.68rem] uppercase tracking-[0.28em] text-(--muted)">
                Workflow
              </p>
              <h3 className="mt-2 text-xl font-semibold text-white">
                How to run it
              </h3>
              <div className="mt-5 space-y-3">
                {steps.map((step, index) => (
                  <div
                    key={step}
                    className="flex items-start gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3"
                  >
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-white/10 bg-white/10 text-sm font-semibold text-white">
                      {index + 1}
                    </div>
                    <p className="pt-1 text-sm leading-6 text-slate-200">{step}</p>
                  </div>
                ))}
              </div>
            </aside>

            <aside className="rounded-3xl border border-white/10 bg-(--surface) p-6 shadow-[0_18px_50px_rgba(2,8,23,0.22)]">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-[0.68rem] uppercase tracking-[0.28em] text-(--muted)">
                    Prediction trail
                  </p>
                  <h3 className="mt-2 text-xl font-semibold text-white">
                    Recent detections
                  </h3>
                </div>
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-mono uppercase tracking-[0.22em] text-(--muted)">
                  Latest 10
                </span>
              </div>

              <div className="mt-5 max-h-96 space-y-3 overflow-y-auto pr-1">
                {history.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-white/5 px-4 py-6 text-sm text-(--muted)">
                    No predictions yet. Start the stream to fill this panel.
                  </div>
                ) : (
                  history.map((item, idx) => (
                    <div
                      key={`${item}-${idx}`}
                      className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-4 py-3"
                    >
                      <span className="text-xs uppercase tracking-[0.26em] text-(--muted)">
                        #{history.length - idx}
                      </span>
                      <span className="rounded-full border border-[#8dd6ff]/20 bg-[#8dd6ff] px-3 py-1 text-sm font-semibold text-slate-950">
                        {item.toUpperCase()}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </aside>

            <aside className="rounded-3xl border border-white/10 bg-(--surface) p-6 shadow-[0_18px_50px_rgba(2,8,23,0.22)]">
              <p className="text-[0.7rem] uppercase tracking-[0.28em] text-white/70">
                Implementation note
              </p>
              <p className="mt-3 text-sm leading-7 text-white/88">
                The detector combines MediaPipe hand landmarks with an SVM model, so the interface stays focused on the camera stream and the most recent output instead of decorative noise.
              </p>
            </aside>
          </div>


        </section>

      </div >
    </main >
  );
}
