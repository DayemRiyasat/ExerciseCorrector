// FormCheck frontend controller
// Handles exercise selection, media input, results, demo mode, and session history. 

let selectedExercise = null;
let selectedExerciseMeta = null;
let selectedMode = null;
let webcamStream = null;
let isProcessing = false;
let processingAnimation = null;
let frameCount = 0;
let stats = { good: 0, bad: 0, total: 0 };
let repCount = 0;
let lastPrediction = "";
let lastRepCount = 0;
let lastUpdateTime = 0;
let currentSquatState = "standing";

let isVideoProcessing = false;
let videoPaused = false;
let videoResults = [];
let currentVideoTime = 0;
let videoProcessedFrames = 0;
let videoRepCount = 0;
let videoFrameQueue = [];
let isProcessingQueue = false;
let maxQueueSize = 30;

const qs = (selector) => document.querySelector(selector);
const qsa = (selector) => Array.from(document.querySelectorAll(selector));

const exerciseCards = qsa('.exercise-card');
const modeSection = qs('#input-mode-selection');
const analysisSection = qs('#analysis-section');
const exerciseRecommendation = qs('#exercise-recommendation');
const demoAnalysisBtn = qs('#demo-analysis-btn');
const refreshHistoryBtn = qs('#refresh-history-btn');

const webcamBtn = qs('#webcam-btn');
const videoBtn = qs('#video-btn');
const imageBtn = qs('#image-btn');
const webcamMode = qs('#webcam-mode');
const videoMode = qs('#video-mode');
const imageMode = qs('#image-mode');

const startWebcamBtn = qs('#start-webcam');
const stopWebcamBtn = qs('#stop-webcam');
const resetRepsBtn = qs('#reset-reps');
const webcamElement = qs('#webcam');
const outputCanvas = qs('#output-canvas');
const showLandmarksCheckbox = qs('#show-landmarks');

const uploadArea = qs('#upload-area');
const videoInput = qs('#video-input');
const videoPreview = qs('#video-preview');
const previewVideo = qs('#preview-video');
const analyzeVideoBtn = qs('#analyze-video');
const pauseVideoBtn = qs('#pause-video');
const resumeVideoBtn = qs('#resume-video');
const stopVideoBtn = qs('#stop-video');
const uploadAnotherVideoBtn = qs('#upload-another-video');
const videoProgress = qs('#video-progress');
const progressText = qs('#progress-text');
const progressFrames = qs('#progress-frames');
const videoProgressFill = qs('#video-progress-fill');
const videoOutputCanvas = qs('#video-output-canvas');
const videoResultsDiv = qs('#video-results');

const imageUploadArea = qs('#image-upload-area');
const imageInput = qs('#image-input');
const imagePreview = qs('#image-preview');
const originalImage = qs('#original-image');
const analyzedImage = qs('#analyzed-image');
const analyzeImageBtn = qs('#analyze-image');
const uploadAnotherBtn = qs('#upload-another');
const imageResults = qs('#image-results');

const statusBadge = qs('#status-badge');
const statusIcon = qs('#status-icon');
const statusMessage = qs('#status-message');
const confidenceFill = qs('#confidence-fill');
const confidenceValue = qs('#confidence-value');
const tipsList = qs('#tips-list');
const summaryScore = qs('#summary-score');
const summaryStatus = qs('#summary-status');
const summaryIssue = qs('#summary-issue');
const explainabilityText = qs('#explainability-text');
const modelVersionText = qs('#model-version-text');

const statGood = qs('#stat-good');
const statBad = qs('#stat-bad');
const statTotal = qs('#stat-total');
const statAccuracy = qs('#stat-accuracy');
const statReps = qs('#stat-reps');
const repCounterOverlay = qs('#rep-counter-overlay');
const videoRepCounterOverlay = qs('#video-rep-counter-overlay');

const metricsGrid = qs('#metrics-grid');
const bodyFeedbackList = qs('#body-feedback-list');
const repTimelineList = qs('#rep-timeline-list');
const cameraGuidanceList = qs('#camera-guidance-list');
const readinessStatus = qs('#readiness-status');
const historyList = qs('#history-list');
const loadingOverlay = qs('#loading-overlay');
const loadingMessage = qs('#loading-message');

const stateLabels = {
    standing: 'READY',
    descending: 'DOWN',
    bottom: 'HOLD',
    ascending: 'UP',
    lifting: 'LIFT',
    up: 'READY',
    down: 'ARMS DOWN',
    curling: 'CURLING',
    lowering: 'LOWERING'
};

function setNavActive() {
    const links = qsa('.nav-link');
    links.forEach((link) => {
        link.addEventListener('click', () => {
            links.forEach((item) => item.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function safeText(value, fallback = '-') {
    if (value === null || value === undefined || value === '') return fallback;
    return String(value);
}

function prettifyLabel(value) {
    if (!value) return 'Good form';
    if (value === 'none') return 'Good form';
    return String(value).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function setActiveModeButton(activeButton) {
    [webcamBtn, videoBtn, imageBtn].forEach((button) => button.classList.remove('active'));
    activeButton.classList.add('active');
}

function showAnalysisSection() {
    analysisSection.classList.remove('hidden');
    analysisSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    resetStats();
    resetRepCount();
    clearRepTimeline();
}

exerciseCards.forEach((card) => {
    card.addEventListener('click', async () => {
        const exercise = card.dataset.exercise;
        if (!exercise) return;

        exerciseCards.forEach((item) => item.classList.remove('selected'));
        card.classList.add('selected');
        selectedExercise = exercise;

        showLoading('Loading model and exercise profile...');

        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exercise_type: exercise })
            });
            const data = await response.json();
            hideLoading();

            if (!data.success) {
                throw new Error(data.error || 'Model could not be loaded');
            }

            selectedExerciseMeta = data.exercise_meta || null;
            if (selectedExerciseMeta && exerciseRecommendation) {
                exerciseRecommendation.textContent = selectedExerciseMeta.recommended_view || 'Keep the full body visible throughout the movement.';
            }

            modeSection.classList.remove('hidden');
            modeSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            renderModelLoadedState(data);
            showNotification(`${selectedExerciseMeta?.label || prettifyLabel(exercise)} model loaded`, 'success');
        } catch (error) {
            hideLoading();
            card.classList.remove('selected');
            selectedExercise = null;
            showNotification(error.message, 'error');
        }
    });
});

function renderModelLoadedState(data) {
    modelVersionText.textContent = `Model version: ${data.model_version || 'prototype-v1'}`;
    renderCameraGuidance([
        { label: 'Full body visible', status: 'Check' },
        { label: 'Lighting', status: 'Check' },
        { label: 'Camera stability', status: 'Check' }
    ]);
}

webcamBtn.addEventListener('click', () => {
    selectedMode = 'webcam';
    setActiveModeButton(webcamBtn);
    showAnalysisSection();
    webcamMode.classList.remove('hidden');
    videoMode.classList.add('hidden');
    imageMode.classList.add('hidden');
});

videoBtn.addEventListener('click', () => {
    selectedMode = 'video';
    setActiveModeButton(videoBtn);
    showAnalysisSection();
    webcamMode.classList.add('hidden');
    videoMode.classList.remove('hidden');
    imageMode.classList.add('hidden');
});

imageBtn.addEventListener('click', () => {
    selectedMode = 'image';
    setActiveModeButton(imageBtn);
    showAnalysisSection();
    webcamMode.classList.add('hidden');
    videoMode.classList.add('hidden');
    imageMode.classList.remove('hidden');
});

startWebcamBtn.addEventListener('click', async () => {
    if (!selectedExercise) {
        showNotification('Select an exercise first', 'error');
        return;
    }

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 },
            audio: false
        });

        webcamElement.srcObject = webcamStream;
        webcamElement.onloadedmetadata = () => {
            webcamElement.play();
            outputCanvas.width = webcamElement.videoWidth;
            outputCanvas.height = webcamElement.videoHeight;
            startWebcamBtn.classList.add('hidden');
            stopWebcamBtn.classList.remove('hidden');
            resetRepsBtn.classList.remove('hidden');
            isProcessing = true;
            processWebcamFrame();
        };
    } catch (error) {
        showNotification(`Camera error: ${error.message}`, 'error');
    }
});

stopWebcamBtn.addEventListener('click', () => {
    stopWebcam();
});

resetRepsBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/reset_counter', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            resetRepCount();
            clearRepTimeline();
            showNotification('Rep counter reset', 'success');
        }
    } catch (error) {
        showNotification('Could not reset rep counter', 'error');
    }
});

async function stopWebcam() {
    isProcessing = false;

    if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        webcamStream = null;
    }

    webcamElement.srcObject = null;
    startWebcamBtn.classList.remove('hidden');
    stopWebcamBtn.classList.add('hidden');
    resetRepsBtn.classList.add('hidden');

    if (processingAnimation) {
        cancelAnimationFrame(processingAnimation);
    }

    frameCount = 0;

    if (stats.total > 0) {
        await saveSession({
            exercise: selectedExercise,
            mode: 'webcam',
            score: Number(summaryScore.textContent) || Math.round((stats.good / stats.total) * 100),
            confidence: Number(confidenceValue.textContent.replace('%', '')) / 100 || 0,
            reps: repCount,
            main_issue: summaryIssue.textContent,
            summary: explainabilityText.textContent,
            payload: { stats }
        });
    }
}

async function processWebcamFrame() {
    if (!isProcessing) return;

    frameCount += 1;
    if (frameCount % 3 !== 0) {
        processingAnimation = requestAnimationFrame(processWebcamFrame);
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcamElement, 0, 0);
    const frameData = canvas.toDataURL('image/jpeg', 0.55);

    try {
        const response = await fetch('/api/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frameData })
        });
        const data = await response.json();
        const now = Date.now();

        if (data.success) {
            drawProcessedFrame(data.processed_frame, outputCanvas);

            if (data.rep_info) {
                currentSquatState = data.rep_info.state;
                repCount = data.rep_info.rep_count || repCount;
                if (repCount !== lastRepCount) {
                    updateRepCount();
                    lastRepCount = repCount;
                }
                if (data.rep_info.rep_counted) {
                    flashRepCounter(repCounterOverlay);
                    showNotification(`Rep ${repCount} counted. Score ${data.rep_event?.score || data.form_score}/100`, 'success');
                }
                updateStateDisplay(repCounterOverlay, data.rep_info);
            }

            const shouldUpdate = data.prediction !== lastPrediction || now - lastUpdateTime > 900;
            if (shouldUpdate) {
                applyAnalysisResult(data);
                lastPrediction = data.prediction;
                lastUpdateTime = now;
            }

            updateFrameStats(data.prediction);
            if (data.rep_event) appendRepEvent(data.rep_event);
        } else {
            applyAnalysisResult(data);
        }
    } catch (error) {
        console.error(error);
    }

    processingAnimation = requestAnimationFrame(processWebcamFrame);
}

function drawProcessedFrame(imageData, canvasElement) {
    if (!imageData || !canvasElement || (showLandmarksCheckbox && !showLandmarksCheckbox.checked)) return;
    const img = new Image();
    img.onload = () => {
        const ctx = canvasElement.getContext('2d');
        ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
    };
    img.src = imageData;
}

function updateStateDisplay(overlay, repInfo) {
    if (!overlay || !repInfo) return;
    const repLabel = overlay.querySelector('.rep-label');
    if (repLabel) {
        repLabel.textContent = stateLabels[repInfo.state] || 'REPS';
    }
}

uploadArea.addEventListener('click', () => videoInput.click());
videoInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) handleVideoFile(file);
});

['dragover', 'dragleave', 'drop'].forEach((eventName) => {
    uploadArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadArea.style.borderColor = eventName === 'dragover' ? 'var(--cyan)' : 'rgba(34, 211, 238, 0.42)';
        if (eventName === 'drop') {
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) handleVideoFile(file);
        }
    });
});

function handleVideoFile(file) {
    if (!file.type.startsWith('video/')) {
        showNotification('Please upload a video file', 'error');
        return;
    }
    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    previewVideo.onloadedmetadata = () => {
        videoOutputCanvas.width = previewVideo.videoWidth || 640;
        videoOutputCanvas.height = previewVideo.videoHeight || 480;
    };

    uploadArea.classList.add('hidden');
    videoPreview.classList.remove('hidden');
    videoResultsDiv.classList.add('hidden');
    videoProgress.classList.add('hidden');
    analyzeVideoBtn.classList.remove('hidden');
    pauseVideoBtn.classList.add('hidden');
    resumeVideoBtn.classList.add('hidden');
    stopVideoBtn.classList.add('hidden');
    uploadAnotherVideoBtn.classList.add('hidden');
    previewVideo.videoFile = file;
    clearRepTimeline();
}

uploadAnotherVideoBtn.addEventListener('click', resetVideoUpload);

function resetVideoUpload() {
    isVideoProcessing = false;
    videoPaused = false;
    videoResults = [];
    currentVideoTime = 0;
    videoProcessedFrames = 0;
    videoRepCount = 0;
    videoFrameQueue = [];
    isProcessingQueue = false;
    previewVideo.src = '';
    previewVideo.videoFile = null;
    videoInput.value = '';
    videoPreview.classList.add('hidden');
    videoResultsDiv.classList.add('hidden');
    videoProgress.classList.add('hidden');
    uploadArea.classList.remove('hidden');
    setOverlayRepCount(videoRepCounterOverlay, 0);
    updateStatusPanel('unknown', 0, {
        status: 'WAITING',
        message: 'Upload a video to analyze',
        tips: ['Use a clear side view', 'Keep the full movement visible', 'Use steady lighting'],
        color: 'secondary'
    });
    resetStats();
    clearRepTimeline();
}

analyzeVideoBtn.addEventListener('click', () => {
    if (!previewVideo.videoFile) {
        showNotification('No video file selected', 'error');
        return;
    }
    startVideoAnalysis();
});

function startVideoAnalysis() {
    isVideoProcessing = true;
    videoPaused = false;
    videoResults = [];
    currentVideoTime = 0;
    videoProcessedFrames = 0;
    videoRepCount = 0;
    videoFrameQueue = [];
    isProcessingQueue = false;

    setOverlayRepCount(videoRepCounterOverlay, 0);
    statReps.textContent = '0';
    previewVideo.currentTime = 0;
    videoOutputCanvas.width = previewVideo.videoWidth || 640;
    videoOutputCanvas.height = previewVideo.videoHeight || 480;

    analyzeVideoBtn.classList.add('hidden');
    pauseVideoBtn.classList.remove('hidden');
    stopVideoBtn.classList.remove('hidden');
    uploadAnotherVideoBtn.classList.add('hidden');
    videoProgress.classList.remove('hidden');
    videoResultsDiv.classList.add('hidden');

    applyAnalysisResult({
        success: false,
        form_score: 0,
        score_status: 'Analyzing',
        main_issue: 'Processing video',
        feedback: {
            status: 'ANALYZING',
            message: 'Processing video frames',
            tips: ['Sampling frames throughout the video', 'Results will appear when processing is complete'],
            color: 'secondary'
        },
        camera_guidance: [
            { label: 'Full body visible', status: 'Check' },
            { label: 'Lighting', status: 'Check' },
            { label: 'Camera stability', status: 'Check' }
        ]
    });

    captureVideoFrames();
    processVideoQueue();
}

pauseVideoBtn.addEventListener('click', () => {
    videoPaused = true;
    pauseVideoBtn.classList.add('hidden');
    resumeVideoBtn.classList.remove('hidden');
});

resumeVideoBtn.addEventListener('click', () => {
    videoPaused = false;
    resumeVideoBtn.classList.add('hidden');
    pauseVideoBtn.classList.remove('hidden');
    if (!isProcessingQueue) processVideoQueue();
    captureVideoFrames();
});

stopVideoBtn.addEventListener('click', () => {
    finishVideoAnalysis();
});

async function captureVideoFrames() {
    if (!isVideoProcessing) return;
    if (videoPaused) {
        setTimeout(captureVideoFrames, 120);
        return;
    }

    if (!Number.isFinite(previewVideo.duration) || previewVideo.duration <= 0) {
        showNotification('Video metadata is not available yet', 'error');
        finishVideoAnalysis();
        return;
    }

    if (currentVideoTime >= previewVideo.duration) {
        isVideoProcessing = false;
        waitForQueueAndFinish();
        return;
    }

    previewVideo.currentTime = currentVideoTime;
    await new Promise((resolve) => {
        previewVideo.onseeked = resolve;
    });

    const frameNumber = Math.round(currentVideoTime * 30);
    const canvas = document.createElement('canvas');
    canvas.width = previewVideo.videoWidth || 640;
    canvas.height = previewVideo.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(previewVideo, 0, 0, canvas.width, canvas.height);

    if (videoFrameQueue.length < maxQueueSize) {
        videoFrameQueue.push({
            data: canvas.toDataURL('image/jpeg', 0.55),
            number: frameNumber,
            time: currentVideoTime.toFixed(2)
        });
    }

    currentVideoTime += 0.20;
    setTimeout(captureVideoFrames, 10);
}

async function processVideoQueue() {
    if (isProcessingQueue) return;
    isProcessingQueue = true;

    while ((isVideoProcessing || videoFrameQueue.length > 0) && !videoPaused) {
        if (videoFrameQueue.length === 0) {
            await sleep(50);
            continue;
        }

        const frameInfo = videoFrameQueue.shift();
        try {
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: frameInfo.data })
            });
            const data = await response.json();

            if (data.success) {
                videoProcessedFrames += 1;
                videoResults.push({
                    frame: frameInfo.number,
                    time: frameInfo.time,
                    prediction: data.prediction,
                    confidence: data.confidence,
                    score: data.form_score,
                    issue: data.main_issue
                });

                drawProcessedFrame(data.processed_frame, videoOutputCanvas);
                applyAnalysisResult(data);
                updateFrameStats(data.prediction);

                if (data.rep_info) {
                    currentSquatState = data.rep_info.state || currentSquatState;
                    updateStateDisplay(videoRepCounterOverlay, data.rep_info);
                    if (data.rep_info.rep_counted) {
                        videoRepCount += 1;
                        setOverlayRepCount(videoRepCounterOverlay, videoRepCount);
                        statReps.textContent = videoRepCount;
                    }
                }
                if (data.rep_event) appendRepEvent({ ...data.rep_event, rep: videoRepCount || data.rep_event.rep });
            }

            const progress = Math.min(Math.round((Number(frameInfo.time) / previewVideo.duration) * 100), 100);
            progressText.textContent = `Processing: ${progress}%`;
            progressFrames.textContent = `Frames: ${videoProcessedFrames} | Reps: ${videoRepCount} | State: ${stateLabels[currentSquatState] || 'READY'}`;
            videoProgressFill.style.width = `${progress}%`;
        } catch (error) {
            console.error(error);
        }
    }

    isProcessingQueue = false;
}

function waitForQueueAndFinish() {
    const interval = setInterval(() => {
        if (videoFrameQueue.length === 0 && !isProcessingQueue) {
            clearInterval(interval);
            finishVideoAnalysis();
        }
    }, 120);
}

async function finishVideoAnalysis() {
    isVideoProcessing = false;
    videoPaused = false;
    videoFrameQueue = [];
    pauseVideoBtn.classList.add('hidden');
    resumeVideoBtn.classList.add('hidden');
    stopVideoBtn.classList.add('hidden');
    analyzeVideoBtn.classList.remove('hidden');
    uploadAnotherVideoBtn.classList.remove('hidden');
    videoProgress.classList.add('hidden');

    if (videoResults.length === 0) return;

    const correctFrames = videoResults.filter((item) => item.prediction === 'none').length;
    const totalFrames = videoResults.length;
    const accuracy = totalFrames > 0 ? (correctFrames / totalFrames) * 100 : 0;
    const avgScore = Math.round(videoResults.reduce((sum, item) => sum + (item.score || 0), 0) / totalFrames);
    const mainIssue = mostCommonIssue(videoResults);

    displayVideoResults({
        total_frames: totalFrames,
        correct_frames: correctFrames,
        accuracy,
        results: videoResults,
        reps: videoRepCount,
        score: avgScore,
        main_issue: mainIssue
    });

    await saveSession({
        exercise: selectedExercise,
        mode: 'video',
        score: avgScore,
        confidence: accuracy / 100,
        reps: videoRepCount,
        main_issue: mainIssue,
        summary: `Video analysis completed across ${totalFrames} sampled frames.`,
        payload: { results: videoResults.slice(0, 30) }
    });
}

function displayVideoResults(data) {
    const incorrect = data.total_frames - data.correct_frames;
    videoResultsDiv.classList.remove('hidden');
    videoResultsDiv.innerHTML = `
        <h3>Video Analysis Complete</h3>
        <div class="result-summary">
            <div class="result-item"><div class="result-item-value">${data.score}</div><div class="result-item-label">Average Score</div></div>
            <div class="result-item"><div class="result-item-value">${data.reps}</div><div class="result-item-label">Total Reps</div></div>
            <div class="result-item"><div class="result-item-value">${data.total_frames}</div><div class="result-item-label">Frames</div></div>
            <div class="result-item"><div class="result-item-value result-good">${data.correct_frames}</div><div class="result-item-label">Correct</div></div>
            <div class="result-item"><div class="result-item-value result-bad">${incorrect}</div><div class="result-item-label">Review</div></div>
            <div class="result-item"><div class="result-item-value">${data.accuracy.toFixed(1)}%</div><div class="result-item-label">Accuracy</div></div>
        </div>
        <div class="result-details">
            <h4>Frame breakdown</h4>
            <div class="compact-results">
                ${data.results.slice(0, 60).map((item) => `
                    <div class="result-row">
                        <span class="result-frame">Frame ${item.frame} at ${item.time}s</span>
                        <span class="result-prediction ${item.prediction === 'none' ? 'correct' : 'incorrect'}">${prettifyLabel(item.prediction)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    videoResultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

imageUploadArea.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) handleImageFile(file);
});

['dragover', 'dragleave', 'drop'].forEach((eventName) => {
    imageUploadArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        imageUploadArea.style.borderColor = eventName === 'dragover' ? 'var(--cyan)' : 'rgba(34, 211, 238, 0.42)';
        if (eventName === 'drop') {
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) handleImageFile(file);
        }
    });
});

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please upload an image file', 'error');
        return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
        originalImage.src = event.target.result;
        originalImage.imageData = event.target.result;
        imageUploadArea.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        imageResults.classList.add('hidden');
        analyzedImage.width = 0;
        analyzedImage.height = 0;
    };
    reader.readAsDataURL(file);
}

analyzeImageBtn.addEventListener('click', async () => {
    if (!originalImage.imageData) {
        showNotification('No image loaded', 'error');
        return;
    }

    showLoading('Analyzing posture...');
    try {
        const response = await fetch('/api/process_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: originalImage.imageData })
        });
        const data = await response.json();
        hideLoading();

        if (!data.success) {
            throw new Error(data.message || data.error || 'Image could not be analyzed');
        }

        const img = new Image();
        img.onload = () => {
            analyzedImage.width = img.width;
            analyzedImage.height = img.height;
            analyzedImage.getContext('2d').drawImage(img, 0, 0);
        };
        img.src = data.processed_frame;

        applyAnalysisResult(data);
        displayImageAnalysis(data);
        imageResults.classList.remove('hidden');
        imageResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
        await loadHistory();
    } catch (error) {
        hideLoading();
        showNotification(error.message, 'error');
    }
});

uploadAnotherBtn.addEventListener('click', () => {
    imagePreview.classList.add('hidden');
    imageResults.classList.add('hidden');
    imageUploadArea.classList.remove('hidden');
    originalImage.src = '';
    originalImage.imageData = null;
    analyzedImage.width = 0;
    analyzedImage.height = 0;
    imageInput.value = '';
});

function displayImageAnalysis(data) {
    const details = data.analysis_details || {};
    setText('#knee-angle-value', `${safeText(details.knee_angle || details.front_knee_angle)}°`);
    setText('#left-knee', safeText(details.left_knee_angle || details.front_knee_angle));
    setText('#right-knee', safeText(details.right_knee_angle || details.back_knee_angle));
    setText('#hip-angle-value', `${safeText(details.hip_angle || details.front_hip_angle)}°`);
    setText('#depth-value', details.depth_achieved === true ? 'Good' : details.depth_achieved === false ? 'Needs work' : '-');
    setText('#stance-value', safeText(details.stance_width));
    setText('#back-lean-value', safeText(details.back_lean));
    setText('#lean-amount', safeText(details.lean_amount));
    setText('#overall-form', data.prediction === 'none' ? 'Strong' : prettifyLabel(data.prediction));
    setText('#form-confidence', Math.round((data.confidence || 0) * 100));
}

function setText(selector, value) {
    const element = qs(selector);
    if (element) element.textContent = value;
}

function applyAnalysisResult(data) {
    const feedback = data.feedback || {
        status: data.score_status || 'WAITING',
        message: data.message || data.error || 'Position yourself in frame',
        tips: ['Ensure the full body is visible', 'Use steady lighting'],
        color: 'secondary'
    };

    updateStatusPanel(data.prediction || 'unknown', data.confidence || 0, feedback);
    renderSummary(data);
    renderMetrics(data.metrics || []);
    renderBodyFeedback(data.body_feedback || []);
    renderCameraGuidance(data.camera_guidance || []);

    if (data.rep_timeline) {
        renderRepTimeline(data.rep_timeline);
    }
    if (data.rep_event) {
        appendRepEvent(data.rep_event);
    }
}

function updateStatusPanel(prediction, confidence, feedback) {
    statusBadge.textContent = feedback.status || prettifyLabel(prediction);
    statusBadge.className = 'status-badge';

    if (feedback.color === 'success') statusBadge.classList.add('correct');
    if (feedback.color === 'danger') statusBadge.classList.add('incorrect');
    if (feedback.color === 'warning') statusBadge.classList.add('warning');

    const statusLabels = {
        success: 'GOOD',
        danger: 'FIX',
        warning: 'CHECK',
        secondary: 'WAIT'
    };
    statusIcon.textContent = statusLabels[feedback.color] || 'WAIT';
    statusMessage.textContent = feedback.message || 'Position yourself in frame';

    const confidencePercent = Math.round((confidence || 0) * 100);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;

    tipsList.innerHTML = '';
    (feedback.tips || ['Use steady lighting', 'Keep the full body visible']).forEach((tip) => {
        const li = document.createElement('li');
        li.textContent = tip;
        tipsList.appendChild(li);
    });
}

function renderSummary(data) {
    if (data.form_score !== undefined && data.form_score !== null) {
        summaryScore.textContent = data.form_score;
    }
    summaryStatus.textContent = data.score_status || 'Analysis running';
    summaryIssue.textContent = data.main_issue || prettifyLabel(data.prediction);

    if (data.explainability) {
        explainabilityText.textContent = `${data.explainability.reason} ${data.explainability.score_note || ''}`.trim();
    } else if (data.message || data.error) {
        explainabilityText.textContent = data.message || data.error;
    }

    if (data.model_version) {
        modelVersionText.textContent = `Model version: ${data.model_version}`;
    }
}

function renderMetrics(metrics) {
    if (!metrics.length) {
        metricsGrid.innerHTML = '<div class="empty-state">Metrics will appear when pose landmarks are detected.</div>';
        return;
    }
    metricsGrid.innerHTML = metrics.map((metric) => `
        <div class="metric-item">
            <span>${metric.label}</span>
            <div class="metric-value">${metric.value}${metric.suffix || ''}</div>
        </div>
    `).join('');
}

function renderBodyFeedback(items) {
    if (!items.length) {
        bodyFeedbackList.innerHTML = '<div class="empty-state">Analyze a movement to see area-specific feedback.</div>';
        return;
    }
    bodyFeedbackList.innerHTML = items.map((item) => `
        <div class="body-feedback-item">
            <div>
                <strong>${item.area}</strong>
                <span>${item.note}</span>
            </div>
            <span>${item.status}</span>
        </div>
    `).join('');
}

function renderCameraGuidance(items) {
    if (!items.length) return;
    cameraGuidanceList.innerHTML = items.map((item) => `
        <div><span>${item.label}</span><strong>${item.status}</strong></div>
    `).join('');

    const allGood = items.every((item) => String(item.status).toLowerCase() === 'good');
    readinessStatus.textContent = allGood ? 'Ready' : 'Check setup';
}

function appendRepEvent(event) {
    if (!event || !event.rep) return;
    const existing = qsa('.rep-item').some((item) => item.dataset.rep === String(event.rep));
    if (existing) return;

    if (repTimelineList.querySelector('.empty-state')) {
        repTimelineList.innerHTML = '';
    }

    const row = document.createElement('div');
    row.className = 'rep-item';
    row.dataset.rep = String(event.rep);
    row.innerHTML = `
        <div>
            <strong>Rep ${event.rep}</strong>
            <span>${event.issue || 'Good form'}</span>
        </div>
        <span>${event.status || 'Review'} · ${event.score || 0}/100</span>
    `;
    repTimelineList.prepend(row);
}

function renderRepTimeline(items) {
    if (!items.length) {
        clearRepTimeline();
        return;
    }
    repTimelineList.innerHTML = items.map((item) => `
        <div class="rep-item" data-rep="${item.rep}">
            <div><strong>Rep ${item.rep}</strong><span>${item.issue}</span></div>
            <span>${item.status} · ${item.score}/100</span>
        </div>
    `).join('');
}

function clearRepTimeline() {
    repTimelineList.innerHTML = '<div class="empty-state">Completed reps will appear here.</div>';
}

function updateFrameStats(prediction) {
    if (prediction === 'none') stats.good += 1;
    else stats.bad += 1;
    stats.total += 1;
    updateStatsDisplay();
}

function updateStatsDisplay() {
    statGood.textContent = stats.good;
    statBad.textContent = stats.bad;
    statTotal.textContent = stats.total;
    const accuracy = stats.total > 0 ? Math.round((stats.good / stats.total) * 100) : 0;
    statAccuracy.textContent = `${accuracy}%`;
}

function resetStats() {
    stats = { good: 0, bad: 0, total: 0 };
    updateStatsDisplay();
}

function updateRepCount() {
    statReps.textContent = repCount;
    setOverlayRepCount(repCounterOverlay, repCount);
}

function resetRepCount() {
    repCount = 0;
    videoRepCount = 0;
    lastRepCount = 0;
    statReps.textContent = '0';
    setOverlayRepCount(repCounterOverlay, 0);
    setOverlayRepCount(videoRepCounterOverlay, 0);
}

function setOverlayRepCount(overlay, value) {
    if (!overlay) return;
    const repNumber = overlay.querySelector('.rep-number');
    if (repNumber) repNumber.textContent = value;
}

function flashRepCounter(overlay) {
    if (!overlay) return;
    const repNumber = overlay.querySelector('.rep-number');
    if (!repNumber) return;
    repNumber.style.transform = 'scale(1.25)';
    setTimeout(() => {
        repNumber.style.transform = 'scale(1)';
    }, 250);
}

function mostCommonIssue(results) {
    const counts = {};
    results.forEach((item) => {
        const key = prettifyLabel(item.prediction);
        counts[key] = (counts[key] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Good form';
}

async function saveSession(payload) {
    try {
        await fetch('/api/save_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        await loadHistory();
    } catch (error) {
        console.error('Could not save session', error);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/session_history?limit=8');
        const data = await response.json();
        if (!data.success || !data.sessions.length) {
            historyList.innerHTML = '<div class="empty-state">No saved sessions yet.</div>';
            return;
        }

        historyList.innerHTML = data.sessions.map((session) => {
            const date = new Date(session.created_at).toLocaleString([], {
                month: 'short',
                day: 'numeric',
                hour: 'numeric',
                minute: '2-digit'
            });
            return `
                <div class="history-item">
                    <div>
                        <strong>${prettifyLabel(session.exercise)} · ${prettifyLabel(session.mode)}</strong>
                        <span>${date} · ${safeText(session.main_issue, 'Session saved')}</span>
                    </div>
                    <div class="history-score">${session.score}/100</div>
                </div>
            `;
        }).join('');
    } catch (error) {
        historyList.innerHTML = '<div class="empty-state">History could not be loaded.</div>';
    }
}

refreshHistoryBtn.addEventListener('click', loadHistory);

demoAnalysisBtn.addEventListener('click', async () => {
    showLoading('Loading demo session...');
    try {
        const response = await fetch('/api/demo_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exercise_type: selectedExercise || 'squat' })
        });
        const data = await response.json();
        hideLoading();
        applyAnalysisResult(data);
        renderRepTimeline(data.rep_timeline || []);
        showNotification('Demo session loaded', 'success');
        await loadHistory();
        qs('#results-dashboard').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        hideLoading();
        showNotification(error.message, 'error');
    }
});

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.animation = 'slideIn 0.25s ease-out';
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.25s ease-out';
        setTimeout(() => notification.remove(), 250);
    }, 3200);
}

function showLoading(message) {
    loadingMessage.textContent = message;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

window.addEventListener('beforeunload', () => {
    if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
    }
    isVideoProcessing = false;
});

setNavActive();
loadHistory();
