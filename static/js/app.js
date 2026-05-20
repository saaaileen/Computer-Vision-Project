/**
 * SignSense AI - JavaScript Orchestrator
 * Controls Webcam, MediaPipe Tracking, WebGL Drawing, REST Prediction, Spelling Engine, and TTS
 */

// --- ASL REFERENCE DATA ---
const ASL_LETTERS = {
    'a': 'Fist, thumb on side',
    'b': 'Flat open hand, thumb folded',
    'c': 'Cupped hand shape like a C',
    'd': 'Index points up, others touch thumb',
    'e': 'Clawed fist, fingernails touch thumb',
    'f': 'Index/thumb touch, others spread up',
    'g': 'Index/thumb pinch pointing sideways',
    'h': 'Index & middle extended sideways',
    'i': 'Pinky finger points straight up',
    'j': 'Pinky sweeps upward in a curve',
    'k': 'Index/middle up, thumb on middle knuckle',
    'l': 'Thumb and index form L shape',
    'm': 'Thumb tucked under index, middle, ring',
    'n': 'Thumb tucked under index and middle',
    'o': 'All fingers touch thumb forming O',
    'p': 'K shape pointing straight down',
    'q': 'G shape pointing straight down',
    'r': 'Index and middle fingers crossed',
    's': 'Fist, thumb wrapped across front',
    't': 'Thumb tucked under index finger only',
    'u': 'Index & middle held closed together',
    'v': 'Index & middle spread wide (V sign)',
    'w': 'Index, middle, ring spread wide',
    'x': 'Fist, index curved like a hook',
    'y': 'Thumb and pinky spread wide',
    'z': 'Index finger traces a Z in the air'
};

const ASL_NUMBERS = {
    '0': 'All fingers touch thumb in circle',
    '1': 'Index finger points straight up',
    '2': 'Index and middle fingers up',
    '3': 'Thumb, index, and middle extended',
    '4': 'Four fingers spread up, thumb tucked',
    '5': 'Five fingers spread wide open',
    '6': 'Pinky touches thumb tip, others up',
    '7': 'Ring finger touches thumb, others up',
    '8': 'Middle finger touches thumb, others up',
    '9': 'Index finger touches thumb, others up'
};

// --- DOM ELEMENTS ---
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('landmark-canvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loading-overlay');
const cameraPromptOverlay = document.getElementById('camera-prompt-overlay');

const btnEnableCamera = document.getElementById('btn-enable-camera');
const btnToggleCamera = document.getElementById('btn-toggle-camera');
const btnToggleMirror = document.getElementById('btn-toggle-mirror');
const btnToggleLandmarks = document.getElementById('btn-toggle-landmarks');

const fpsCounter = document.getElementById('fps-counter');
const latencyCounter = document.getElementById('latency-counter');
const backendStatus = document.getElementById('backend-status');

const stabilityFill = document.getElementById('stability-fill');
const predictedCharacter = document.getElementById('predicted-character');
const gestureInstruction = document.getElementById('gesture-instruction');

const composedTextDisplay = document.getElementById('composed-text-display');
const btnBackspace = document.getElementById('btn-backspace');
const btnSpace = document.getElementById('btn-space');
const btnClear = document.getElementById('btn-clear');
const btnSpeak = document.getElementById('btn-speak');
const btnCopy = document.getElementById('btn-copy');

const refHeader = document.getElementById('ref-header');
const refChevron = document.getElementById('ref-chevron');
const refBody = document.getElementById('ref-body');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// --- STATE MANAGEMENT ---
let isCameraRunning = false;
let isMirrored = true;
let showLandmarks = true;
let mediaStream = null;
let cameraHelper = null;

let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

let lastPredictTime = 0;
const PREDICT_INTERVAL = 150; // Milliseconds between predictions to protect backend

let activePrediction = "";
let stablePrediction = "";
let stableCount = 0;
const REQUIRED_STABLE_FRAMES = 6; // At 150ms prediction rate, this is ~0.9s hold
let composedText = "";
let isBackendOnline = false;

// --- INITIALIZE THE APPLICATION ---
document.addEventListener('DOMContentLoaded', () => {
    // Populate Reference Lists
    populateReferenceGrid('tab-letters', ASL_LETTERS);
    populateReferenceGrid('tab-numbers', ASL_NUMBERS);
    
    // Wire UI Event Listeners
    setupEventListeners();
    
    // Pre-check Backend
    checkBackendHealth();
    
    // Setup Canvas Sizes dynamically
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Initialize MediaPipe Hands
    initializeMediaPipe();
});

// --- TELEMETRY & SYSTEM HEALTH ---
function setBackendStatus(online) {
    isBackendOnline = online;
    const indicator = backendStatus.querySelector('.indicator');
    const text = backendStatus.querySelector('.text');
    
    if (online) {
        indicator.className = 'indicator green';
        text.textContent = 'Server: Connected';
        backendStatus.classList.add('model-badge'); // style update
    } else {
        indicator.className = 'indicator red';
        text.textContent = 'Server: Offline';
        backendStatus.classList.remove('model-badge');
    }
}

async function checkBackendHealth() {
    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: Array(21).fill({ x: 0, y: 0, z: 0 }) })
        });
        if (res.status === 200 || res.status === 500) {
            // Even if model fails with dummy inputs, server responded
            setBackendStatus(true);
        } else {
            setBackendStatus(false);
        }
    } catch (e) {
        setBackendStatus(false);
    }
}

// --- MEDIAPIPE INITIALIZATION ---
let handsTracker = null;

function initializeMediaPipe() {
    handsTracker = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    
    handsTracker.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55
    });
    
    handsTracker.onResults(onMediaPipeResults);
    
    // Start Camera immediately
    startWebcamStream();
}

// --- WEBCAM STREAM CONTROLS ---
async function startWebcamStream() {
    loadingOverlay.classList.remove('hidden');
    cameraPromptOverlay.classList.add('hidden');
    
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        };
        
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamElement.srcObject = mediaStream;
        isCameraRunning = true;
        
        // Setup MediaPipe camera helper
        cameraHelper = new Camera(webcamElement, {
            onFrame: async () => {
                if (isCameraRunning) {
                    await handsTracker.send({ image: webcamElement });
                }
            },
            width: 640,
            height: 480
        });
        
        await cameraHelper.start();
        loadingOverlay.classList.add('hidden');
        btnToggleCamera.innerHTML = '<i class="fa-solid fa-pause"></i> <span>Pause Stream</span>';
        btnToggleCamera.classList.add('active');
        
    } catch (err) {
        console.error("Camera access error:", err);
        loadingOverlay.classList.add('hidden');
        cameraPromptOverlay.classList.remove('hidden');
        isCameraRunning = false;
    }
}

function stopWebcamStream() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (cameraHelper) {
        cameraHelper.stop();
    }
    webcamElement.srcObject = null;
    isCameraRunning = false;
    btnToggleCamera.innerHTML = '<i class="fa-solid fa-play"></i> <span>Start Stream</span>';
    btnToggleCamera.classList.remove('active');
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Reset prediction card
    resetPredictionState();
}

function resizeCanvas() {
    const rect = webcamElement.getBoundingClientRect();
    canvasElement.width = webcamElement.videoWidth || 640;
    canvasElement.height = webcamElement.videoHeight || 480;
}

// --- MEDIAPIPE RESULTS CALLBACK ---
function onMediaPipeResults(results) {
    if (!isCameraRunning) return;
    
    // Track client FPS
    calculateFPS();
    
    // Clear Canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Resize dynamically if needed
    if (canvasElement.width !== webcamElement.videoWidth) {
        resizeCanvas();
    }
    
    // Draw landmarks if hand detected and overlay is toggled
    const handDetected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
    
    if (handDetected) {
        const landmarks = results.multiHandLandmarks[0];
        
        if (showLandmarks) {
            drawSkeleton(landmarks);
        }
        
        // Throttled prediction call
        const now = performance.now();
        if (now - lastPredictTime >= PREDICT_INTERVAL) {
            lastPredictTime = now;
            sendPredictionData(landmarks);
        }
    } else {
        // No hand in view
        decayPredictionState();
    }
}

// --- DRAWING SKELETON WITH GLOW EFFECTS ---
function drawSkeleton(landmarks) {
    const w = canvasElement.width;
    const h = canvasElement.height;
    
    canvasCtx.save();
    
    // Neon Connection Styling
    canvasCtx.lineWidth = 4;
    canvasCtx.strokeStyle = 'rgba(0, 240, 255, 0.8)';
    canvasCtx.shadowBlur = 12;
    canvasCtx.shadowColor = 'rgba(0, 240, 255, 1)';
    canvasCtx.lineCap = 'round';
    canvasCtx.lineJoin = 'round';
    
    // Draw Connections
    HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];
        
        canvasCtx.beginPath();
        canvasCtx.moveTo(start.x * w, start.y * h);
        canvasCtx.lineTo(end.x * w, end.y * h);
        canvasCtx.stroke();
    });
    
    // Neon Joint Node Styling
    canvasCtx.shadowBlur = 15;
    canvasCtx.shadowColor = 'rgba(210, 0, 255, 1)';
    canvasCtx.fillStyle = 'rgba(255, 255, 255, 1)';
    
    // Draw Nodes
    landmarks.forEach((lm) => {
        canvasCtx.beginPath();
        canvasCtx.arc(lm.x * w, lm.y * h, 5, 0, 2 * Math.PI);
        canvasCtx.fill();
        
        // Draw a colored border around node
        canvasCtx.strokeStyle = 'rgba(210, 0, 255, 0.95)';
        canvasCtx.lineWidth = 2;
        canvasCtx.stroke();
    });
    
    canvasCtx.restore();
}

// --- FPS METRIC CALCULATOR ---
function calculateFPS() {
    frameCount++;
    const now = performance.now();
    const elapsed = now - lastFrameTime;
    
    if (elapsed >= 1000) {
        fps = Math.round((frameCount * 1000) / elapsed);
        fpsCounter.textContent = fps;
        frameCount = 0;
        lastFrameTime = now;
    }
}

// --- SEND LANDMARKS TO FASTAPI ---
async function sendPredictionData(landmarks) {
    const startTime = performance.now();
    
    // Format input exactly as expected by Pydantic: List of {x, y, z}
    const payload = {
        landmarks: landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }))
    };
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);
        latencyCounter.textContent = latency;
        
        if (data.status === 'success') {
            setBackendStatus(true);
            handlePredictionResult(data.prediction);
        } else {
            console.error("Predict endpoint error:", data.message);
        }
    } catch (err) {
        console.error("Connection to FastAPI failed:", err);
        setBackendStatus(false);
        latencyCounter.textContent = "—";
    }
}

// --- SPELLING ENGINE & STABILITY CONTROLLER ---
function handlePredictionResult(char) {
    activePrediction = char;
    
    // Display raw character on UI
    predictedCharacter.textContent = char;
    predictedCharacter.classList.remove('empty-char');
    predictedCharacter.parentElement.classList.add('active');
    
    // Stability calculation
    if (char === stablePrediction) {
        stableCount++;
        
        // Cap progress and trigger addition
        const progress = Math.min((stableCount / REQUIRED_STABLE_FRAMES) * 100, 100);
        stabilityFill.style.width = `${progress}%`;
        
        if (stableCount >= REQUIRED_STABLE_FRAMES) {
            // Character is stable! Add to composer once
            triggerStableLetter(char);
        } else {
            gestureInstruction.innerHTML = `Holding gesture... <span class="stable-text">${Math.round(progress)}%</span>`;
        }
    } else {
        // Character changed, reset stability countdown
        stablePrediction = char;
        stableCount = 1;
        stabilityFill.style.width = `15%`;
        gestureInstruction.innerHTML = `Detecting sign... <span class="highlight">${char.toUpperCase()}</span>`;
    }
}

function triggerStableLetter(char) {
    // Only add if it's different from the last added character, OR if there was a break in gesture
    // To allow natural typing, we add the stable character once and lock it
    // until the hand goes away or changes to another sign.
    if (activePrediction !== this.lastComposedChar) {
        this.lastComposedChar = char;
        
        // Append text
        const displayChar = char.toUpperCase();
        
        if (composedText.length > 0 && !composedText.endsWith(" ")) {
            composedText += displayChar;
        } else {
            composedText += displayChar;
        }
        
        updateComposerDisplay();
        
        // Micro animation on text area
        composedTextDisplay.style.borderColor = 'var(--accent-cyan)';
        setTimeout(() => composedTextDisplay.style.borderColor = 'var(--border-glass)', 250);
        
        // SpeechSynthesis feedback for the individual letter (optional: clean sound click)
        playSoftTone(600, 0.05); // Play a cute synthesizer click!
    }
    
    gestureInstruction.innerHTML = `Composed: <span class="stable-text font-bold">${char.toUpperCase()}</span> (Lock)`;
    stabilityFill.style.width = `100%`;
}

function decayPredictionState() {
    // When no hand is detected, we decay the stability progress bar and raw text
    activePrediction = "";
    stablePrediction = "";
    stableCount = 0;
    this.lastComposedChar = ""; // Reset composer lock so same character can be repeated
    
    stabilityFill.style.width = `0%`;
    predictedCharacter.textContent = "—";
    predictedCharacter.classList.add('empty-char');
    predictedCharacter.parentElement.classList.remove('active');
    
    gestureInstruction.textContent = "Position your hand in front of the camera";
}

function resetPredictionState() {
    decayPredictionState();
    latencyCounter.textContent = "0";
}

// --- COMPOSE AREA UPDATES ---
function updateComposerDisplay() {
    // Format display for readability (add spaces between letters for a nice spelling view)
    composedTextDisplay.textContent = composedText;
}

// --- EVENT HANDLERS & WIRING ---
function setupEventListeners() {
    // Webcam overlay button
    btnEnableCamera.addEventListener('click', startWebcamStream);
    
    // Viewport control buttons
    btnToggleCamera.addEventListener('click', () => {
        if (isCameraRunning) {
            stopWebcamStream();
        } else {
            startWebcamStream();
        }
    });
    
    btnToggleMirror.addEventListener('click', () => {
        isMirrored = !isMirrored;
        if (isMirrored) {
            webcamElement.classList.add('mirrored');
            canvasElement.classList.add('mirrored');
            btnToggleMirror.classList.add('active');
        } else {
            webcamElement.classList.remove('mirrored');
            canvasElement.classList.remove('mirrored');
            btnToggleMirror.classList.remove('active');
        }
    });
    
    btnToggleLandmarks.addEventListener('click', () => {
        showLandmarks = !showLandmarks;
        if (showLandmarks) {
            btnToggleLandmarks.classList.add('active');
            btnToggleLandmarks.querySelector('span').textContent = "Show Landmarks";
        } else {
            btnToggleLandmarks.classList.remove('active');
            btnToggleLandmarks.querySelector('span').textContent = "Hide Landmarks";
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }
    });
    
    // Spelling controls
    btnBackspace.addEventListener('click', () => {
        if (composedText.length > 0) {
            composedText = composedText.slice(0, -1);
            this.lastComposedChar = ""; // Reset composer lock to prevent blockage
            updateComposerDisplay();
            playSoftTone(400, 0.08);
        }
    });
    
    btnSpace.addEventListener('click', () => {
        composedText += " ";
        this.lastComposedChar = "";
        updateComposerDisplay();
        playSoftTone(500, 0.06);
    });
    
    btnClear.addEventListener('click', () => {
        composedText = "";
        this.lastComposedChar = "";
        updateComposerDisplay();
        playSoftTone(300, 0.15);
    });
    
    btnSpeak.addEventListener('click', () => {
        if (composedText.trim().length > 0) {
            speakText(composedText);
        }
    });
    
    btnCopy.addEventListener('click', () => {
        if (composedText.length > 0) {
            navigator.clipboard.writeText(composedText)
                .then(() => {
                    const originalText = btnCopy.innerHTML;
                    btnCopy.innerHTML = '<i class="fa-solid fa-circle-check"></i> Copied!';
                    btnCopy.classList.add('active');
                    setTimeout(() => {
                        btnCopy.innerHTML = originalText;
                        btnCopy.classList.remove('active');
                    }, 1500);
                })
                .catch(err => console.error("Clipboard copy failed:", err));
        }
    });
    
    // Reference Panel Accordion Toggle
    refHeader.addEventListener('click', () => {
        refBody.classList.toggle('collapsed');
        refChevron.classList.toggle('open');
    });
    
    // Reference Tabs Switching
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const targetTabId = `tab-${btn.dataset.tab}`;
            document.getElementById(targetTabId).classList.add('active');
        });
    });
}

// --- POPULATE REFERENCE GUIDE ---
function populateReferenceGrid(containerId, dataset) {
    const container = document.getElementById(containerId).querySelector('.reference-grid');
    container.innerHTML = '';
    
    Object.entries(dataset).forEach(([key, desc]) => {
        const item = document.createElement('div');
        item.className = 'ref-item';
        item.title = desc; // Tooltip on hover
        
        item.innerHTML = `
            <span class="ref-key">${key}</span>
            <span class="ref-desc">${desc}</span>
        `;
        
        // Add click behavior: teach the user by putting instructions in the prediction helper
        item.addEventListener('click', () => {
            gestureInstruction.innerHTML = `Sign Guide for <span class="highlight">${key.toUpperCase()}</span>: <b>${desc}</b>`;
            playSoftTone(700, 0.04);
        });
        
        container.appendChild(item);
    });
}

// --- SPEECH UTILITY (TTS) ---
function speakText(text) {
    if ('speechSynthesis' in window) {
        // Cancel ongoing speak requests
        window.speechSynthesis.cancel();
        
        // Speak spelled text out loud with standard English accent
        const utterance = new SpeechSynthesisUtterance(text.toLowerCase());
        utterance.rate = 0.9; // Slightly slower for clear spelling speech
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    } else {
        alert("Speech synthesis is not supported on this browser.");
    }
}

// --- AUDIO UTILITY (Synthesizer Click Clicks) ---
let audioCtx = null;
function playSoftTone(frequency, duration) {
    try {
        // Lazy initialize AudioContext on user interaction
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        osc.frequency.value = frequency;
        osc.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
        
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        // AudioContext blocks are fine, ignore gracefully
    }
}
