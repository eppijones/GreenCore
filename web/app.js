/**
 * Pro Putt Sim - Minimalist Tech Refactor
 * 
 * Goals:
 * - Zero visual clutter (no terrain, trees, skybox)
 * - Flat physics for pure stroke analysis
 * - High-contrast "Dark Mode" aesthetic
 * - Dynamic aiming lines and data visualization
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    // Physics
    friction: 0.08,
    stopVelocity: 0.01,
    holeRadius: 0.054,      // 4.25 inch
    holeCaptureSpeed: 0.8,
    ballRadius: 0.02135,
    gravity: 9.81,
    
    // Dimensions
    greenRadius: 4.0,       // 4m radius green
    holeDistance: 3.0,      // 3m putt
    
    // Visuals
    themeColor: 0x4ade80,   // Neon Green
    gridColor: 0xffffff,
    gridOpacity: 0.1,
    bgColor: 0x050505,
    
    // WebSocket
    wsUrl: 'ws://localhost:8765',
};

// ============================================================================
// State
// ============================================================================

const state = {
    ball: {
        position: new THREE.Vector3(0, CONFIG.ballRadius, 0),
        velocity: new THREE.Vector2(0, 0),
        isRolling: false,
    },
    ws: null,
    connected: false,
    clock: new THREE.Clock(),
    showAimLines: true,
    
    // Test Lab Mode state
    testLabMode: false,
    calibration: {
        markersDetected: [],
        homographyValid: false,
        intrinsicsValid: false,
        undistortionEnabled: false,
        intrinsicsFrames: 0,
    },
};

// ============================================================================
// Three.js Globals
// ============================================================================

let scene, camera, renderer, controls;
let ballMesh, holeMesh, greenMesh;
let aimLineGroup;

// ============================================================================
// Initialization
// ============================================================================

function init() {
    initScene();
    createCourse();
    createAimingGuides();
    setupWebSocket();
    setupEvents();
    setupTestLabMode();
    
    // Hide loading screen
    setTimeout(() => {
        document.getElementById('loading').classList.add('hidden');
    }, 500);
    
    animate();
}

function initScene() {
    // 1. Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.bgColor);
    scene.fog = new THREE.FogExp2(CONFIG.bgColor, 0.08); // Fade distant grid

    // 2. Camera (Player perspective)
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    resetCamera();

    // 3. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // 4. Lighting (Dramatic studio setup)
    const ambient = new THREE.AmbientLight(0xffffff, 0.2);
    scene.add(ambient);

    const spotLight = new THREE.SpotLight(0xffffff, 2.0);
    spotLight.position.set(0, 8, -1.5); // Center over putt line
    spotLight.angle = Math.PI / 4;
    spotLight.penumbra = 0.5;
    spotLight.decay = 1;
    spotLight.distance = 20;
    spotLight.castShadow = true;
    spotLight.shadow.mapSize.width = 2048;
    spotLight.shadow.mapSize.height = 2048;
    scene.add(spotLight);

    // Rim light for ball definition
    const rimLight = new THREE.DirectionalLight(CONFIG.themeColor, 0.5);
    rimLight.position.set(-5, 2, -5);
    scene.add(rimLight);
    
    // 5. Minimal Orbit Controls (Restricted)
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableZoom = true;
    controls.minDistance = 1.0;
    controls.maxDistance = 10.0;
    controls.maxPolarAngle = Math.PI / 2 - 0.1; // Don't go below ground
    controls.target.set(0, 0, -CONFIG.holeDistance / 2); // Look at mid-putt
}

function resetCamera() {
    // Position behind ball, looking down the line
    camera.position.set(0, 1.2, 1.5);
    camera.lookAt(0, 0, -CONFIG.holeDistance);
}

// ============================================================================
// Object Creation
// ============================================================================

function createCourse() {
    // 1. The "Green" - Minimalist Grid Disc
    const geometry = new THREE.CircleGeometry(CONFIG.greenRadius, 64);
    
    // Custom grid shader material
    const material = new THREE.ShaderMaterial({
        uniforms: {
            color: { value: new THREE.Color(CONFIG.gridColor) },
            opacity: { value: CONFIG.gridOpacity },
            scale: { value: 20.0 }, // Grid density
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 color;
            uniform float opacity;
            uniform float scale;
            varying vec2 vUv;
            
            void main() {
                // Centered UVs
                vec2 uv = (vUv - 0.5) * scale;
                
                // Grid lines
                vec2 grid = abs(fract(uv - 0.5) - 0.5) / fwidth(uv);
                float line = min(grid.x, grid.y);
                float alpha = 1.0 - min(line, 1.0);
                
                // Radial fade
                float dist = length(vUv - 0.5) * 2.0;
                float fade = 1.0 - smoothstep(0.8, 1.0, dist);
                
                gl_FragColor = vec4(color, opacity * alpha * fade);
            }
        `,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false, // Don't block shadow receiver
    });

    greenMesh = new THREE.Mesh(geometry, material);
    greenMesh.rotation.x = -Math.PI / 2;
    greenMesh.position.y = 0.001; // Just above zero
    scene.add(greenMesh);
    
    // 2. Shadow Catcher (Invisible plane to receive shadows)
    const shadowGeo = new THREE.PlaneGeometry(20, 20);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.3 });
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.receiveShadow = true;
    scene.add(shadowPlane);

    // 3. The Hole
    const holeGeo = new THREE.RingGeometry(CONFIG.holeRadius, CONFIG.holeRadius + 0.005, 32);
    const holeMat = new THREE.MeshBasicMaterial({ color: CONFIG.themeColor });
    const holeRing = new THREE.Mesh(holeGeo, holeMat);
    holeRing.rotation.x = -Math.PI / 2;
    holeRing.position.set(0, 0.002, -CONFIG.holeDistance);
    scene.add(holeRing);
    
    // Hole depth (black cylinder)
    const cupGeo = new THREE.CylinderGeometry(CONFIG.holeRadius, CONFIG.holeRadius, 0.1, 32);
    const cupMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const cup = new THREE.Mesh(cupGeo, cupMat);
    cup.position.set(0, -0.05, -CONFIG.holeDistance);
    scene.add(cup);

    // 4. The Ball
    const ballGeo = new THREE.SphereGeometry(CONFIG.ballRadius, 32, 32);
    const ballMat = new THREE.MeshStandardMaterial({ 
        color: 0xffffff, 
        roughness: 0.2,
        metalness: 0.1 
    });
    ballMesh = new THREE.Mesh(ballGeo, ballMat);
    ballMesh.castShadow = true;
    resetBall();
    scene.add(ballMesh);
}

function createAimingGuides() {
    aimLineGroup = new THREE.Group();
    scene.add(aimLineGroup);
    
    // 1. Center Line (Ball to Hole)
    // Dashed line style
    const points = [];
    points.push(new THREE.Vector3(0, 0.005, 0));
    points.push(new THREE.Vector3(0, 0.005, -CONFIG.holeDistance));
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineDashedMaterial({
        color: 0xffffff,
        linewidth: 1,
        scale: 1,
        dashSize: 0.1,
        gapSize: 0.1,
        opacity: 0.3,
        transparent: true
    });
    
    const centerLine = new THREE.Line(geometry, material);
    centerLine.computeLineDistances();
    aimLineGroup.add(centerLine);
    
    // 2. Distance Markers (0.5m increments)
    for (let d = 0.5; d < CONFIG.holeDistance; d += 0.5) {
        const markerGeo = new THREE.PlaneGeometry(0.02, 0.1);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0xffffff, opacity: 0.2, transparent: true });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.rotation.x = -Math.PI / 2;
        marker.position.set(0, 0.004, -d);
        aimLineGroup.add(marker);
        
        // Text label could be added here if using a font loader
    }
}

// ============================================================================
// Physics (Simplified Flat Surface)
// ============================================================================

function startShot(speedMps, directionDeg) {
    if (state.ball.isRolling) return; // Prevent double hits
    
    // Convert degrees to radians (0 deg = straight to hole = -Z)
    // Standard Math: 0 is +X. We want 0 to be -Z.
    // So angle = directionDeg + 90 (to align with Z axis rotation logic)
    // Actually simpler: 
    // z component = -cos(angle)
    // x component = sin(angle)
    
    const angleRad = (directionDeg * Math.PI) / 180;
    
    state.ball.velocity.x = speedMps * Math.sin(angleRad);
    state.ball.velocity.y = -speedMps * Math.cos(angleRad); // Y is Z in 2D calc
    
    state.ball.isRolling = true;
    
    console.log(`Shot: ${speedMps.toFixed(2)}m/s @ ${directionDeg.toFixed(1)}°`);
    
    // Show aiming ray of actual shot
    showShotRay(angleRad, speedMps);
}

function updatePhysics(dt) {
    if (!state.ball.isRolling) return;
    
    // Time step
    const step = Math.min(dt, 0.05);
    
    // Simple deceleration (friction)
    // v = v0 - a*t
    const speed = state.ball.velocity.length();
    
    if (speed < CONFIG.stopVelocity) {
        stopBall();
        return;
    }
    
    // Friction force opposes velocity
    const frictionAccel = CONFIG.gravity * CONFIG.friction;
    const speedDrop = frictionAccel * step;
    
    // Update velocity magnitude
    const newSpeed = Math.max(0, speed - speedDrop);
    state.ball.velocity.multiplyScalar(newSpeed / speed);
    
    // Update position
    state.ball.position.x += state.ball.velocity.x * step;
    state.ball.position.z += state.ball.velocity.y * step;
    
    // Update mesh
    ballMesh.position.copy(state.ball.position);
    
    // Rotation (visual only)
    ballMesh.rotation.x -= state.ball.velocity.y * step / CONFIG.ballRadius;
    ballMesh.rotation.z += state.ball.velocity.x * step / CONFIG.ballRadius;
    
    checkHole();
}

function checkHole() {
    const holePos = new THREE.Vector2(0, -CONFIG.holeDistance);
    const ballPos = new THREE.Vector2(state.ball.position.x, state.ball.position.z);
    
    if (holePos.distanceTo(ballPos) < CONFIG.holeRadius) {
        // Simple capture logic
        if (state.ball.velocity.length() < CONFIG.holeCaptureSpeed) {
            ballInHole();
        }
    }
    
    // Reset if too far
    if (ballPos.length() > CONFIG.greenRadius + 1.0) {
        stopBall("OFF GREEN");
    }
}

function stopBall(msg) {
    state.ball.isRolling = false;
    
    // Calculate final distance to hole
    const dist = Math.sqrt(
        state.ball.position.x**2 + 
        (state.ball.position.z + CONFIG.holeDistance)**2
    );
    
    showResult(msg || (dist < 0.2 ? "GIMME!" : `${dist.toFixed(2)}m LEFT`));
}

function ballInHole() {
    state.ball.isRolling = false;
    ballMesh.position.y = -0.05; // Drop visual
    showResult("SUNK IT", "PERFECT SPEED");
}

function resetBall() {
    state.ball.position.set(0, CONFIG.ballRadius, 0);
    state.ball.velocity.set(0, 0);
    state.ball.isRolling = false;
    
    ballMesh.position.copy(state.ball.position);
    ballMesh.rotation.set(0, 0, 0);
    
    document.getElementById('result-overlay').classList.remove('visible');
    
    // Clear shot lines
    const existingShot = scene.getObjectByName('shotRay');
    if (existingShot) scene.remove(existingShot);
    
    // Reset camera
    resetCamera();
}

// ============================================================================
// Visual Helpers
// ============================================================================

function showShotRay(angleRad, speed) {
    // Remove old
    const old = scene.getObjectByName('shotRay');
    if (old) scene.remove(old);
    
    // Draw line showing shot direction
    // Length proportional to speed (approx roll dist)
    const distEstimate = (speed * speed) / (2 * CONFIG.gravity * CONFIG.friction);
    
    const endX = Math.sin(angleRad) * distEstimate;
    const endZ = -Math.cos(angleRad) * distEstimate;
    
    const points = [
        new THREE.Vector3(0, 0.01, 0),
        new THREE.Vector3(endX, 0.01, endZ)
    ];
    
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({ color: CONFIG.themeColor });
    const line = new THREE.Line(geo, mat);
    line.name = 'shotRay';
    scene.add(line);
}

function showResult(main, sub = "") {
    const el = document.getElementById('result-overlay');
    document.getElementById('result-main').textContent = main;
    document.getElementById('result-sub').textContent = sub;
    el.classList.add('visible');
}

// ============================================================================
// WebSocket & Events
// ============================================================================

function setupWebSocket() {
    // Mock connection for now or real if available
    state.ws = new WebSocket(CONFIG.wsUrl); // Try local
    
    state.ws.onopen = () => {
        state.connected = true;
        document.querySelector('#connection-status').classList.add('connected');
        console.log('[WS] Connected to server');
        
        // Request initial calibration status
        sendCalibrationMessage({ type: 'get_calibration_status' });
    };
    
    state.ws.onclose = () => {
        state.connected = false;
        document.querySelector('#connection-status').classList.remove('connected');
        console.log('[WS] Disconnected from server');
        
        // Attempt reconnection after 3 seconds
        setTimeout(() => {
            if (!state.connected) {
                console.log('[WS] Attempting reconnection...');
                setupWebSocket();
            }
        }, 3000);
    };
    
    state.ws.onerror = (err) => {
        console.log('[WS] Connection error');
    };
    
    state.ws.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            
            // Handle shot events
            if (data.speed_mps !== undefined) {
                updateHUD(data);
                startShot(data.speed_mps, data.direction_deg);
            }
            
            // Handle calibration status updates
            if (data.type === 'calibration_status') {
                handleCalibrationStatus(data);
            }
            
            // Handle calibration errors
            if (data.type === 'calibration_error') {
                console.error('[Calibration] Error:', data.error);
                showCalibrationMessage(data.error, 'error');
            }
            
            // Handle distance test results
            if (data.type === 'distance_test_results') {
                handleDistanceTestResults(data);
            }
            
            // Handle intrinsics calibration progress
            if (data.type === 'intrinsics_progress') {
                handleIntrinsicsProgress(data);
            }
            
        } catch(err) {
            console.log('[WS] Parse error:', err);
        }
    };
}

function sendCalibrationMessage(msg) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(msg));
    } else {
        console.log('[WS] Cannot send - not connected');
    }
}

function updateHUD(data) {
    document.getElementById('val-speed').textContent = data.speed_mps.toFixed(2);
    document.getElementById('val-angle').textContent = data.direction_deg.toFixed(1) + "°";
    
    // Est distance physics formula: d = v^2 / 2ug
    const dist = (data.speed_mps ** 2) / (2 * 9.81 * CONFIG.friction);
    document.getElementById('val-dist').textContent = dist.toFixed(1) + "m";
}

function setupEvents() {
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    
    window.addEventListener('keydown', (e) => {
        switch(e.key.toLowerCase()) {
            case 'r': resetBall(); break;
            case 't': 
                // Test shot
                const s = 1.5 + Math.random();
                const d = (Math.random() - 0.5) * 5;
                updateHUD({speed_mps: s, direction_deg: d});
                startShot(s, d);
                break;
            case 'l':
                state.showAimLines = !state.showAimLines;
                aimLineGroup.visible = state.showAimLines;
                break;
            case 'c':
                // Toggle Test Lab Mode
                toggleTestLabMode();
                break;
        }
    });
}

// ============================================================================
// Test Lab Mode
// ============================================================================

function setupTestLabMode() {
    // Toggle button
    const toggleBtn = document.getElementById('test-lab-toggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleTestLabMode);
    }
    
    // Close button
    const closeBtn = document.getElementById('lab-close-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            setTestLabMode(false);
        });
    }
    
    // Calibrate button
    const calibrateBtn = document.getElementById('btn-calibrate');
    if (calibrateBtn) {
        calibrateBtn.addEventListener('click', () => {
            const width = parseFloat(document.getElementById('mat-width').value) || 0.70;
            const height = parseFloat(document.getElementById('mat-height').value) || 1.00;
            
            sendCalibrationMessage({
                type: 'start_calibration',
                width_m: width,
                height_m: height
            });
            showCalibrationMessage('Calibrating homography...', 'info');
        });
    }
    
    // Save calibration button
    const saveBtn = document.getElementById('btn-save-calibration');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            sendCalibrationMessage({ type: 'save_calibration' });
            showCalibrationMessage('Saving calibration...', 'info');
        });
    }
    
    // Run distance tests button
    const testBtn = document.getElementById('btn-run-tests');
    if (testBtn) {
        testBtn.addEventListener('click', () => {
            sendCalibrationMessage({ type: 'run_distance_test' });
            showCalibrationMessage('Running distance tests...', 'info');
        });
    }
    
    // Mat dimension inputs
    const matWidthInput = document.getElementById('mat-width');
    const matHeightInput = document.getElementById('mat-height');
    
    if (matWidthInput) {
        matWidthInput.addEventListener('change', () => {
            sendCalibrationMessage({
                type: 'set_mat_dimensions',
                width_m: parseFloat(matWidthInput.value) || 0.70,
                height_m: parseFloat(matHeightInput.value) || 1.00
            });
        });
    }
    
    if (matHeightInput) {
        matHeightInput.addEventListener('change', () => {
            sendCalibrationMessage({
                type: 'set_mat_dimensions',
                width_m: parseFloat(matWidthInput.value) || 0.70,
                height_m: parseFloat(matHeightInput.value) || 1.00
            });
        });
    }
    
    // Undistortion toggle
    const undistortToggle = document.getElementById('undistort-toggle');
    if (undistortToggle) {
        undistortToggle.addEventListener('change', () => {
            sendCalibrationMessage({
                type: 'toggle_undistortion',
                enabled: undistortToggle.checked
            });
        });
    }
    
    // Intrinsics calibration buttons
    const startIntrinsicsBtn = document.getElementById('btn-start-intrinsics');
    if (startIntrinsicsBtn) {
        startIntrinsicsBtn.addEventListener('click', () => {
            sendCalibrationMessage({ type: 'start_intrinsics_calibration' });
            document.getElementById('intrinsics-progress').classList.remove('hidden');
            showCalibrationMessage('Place ChArUco board in view...', 'info');
        });
    }
    
    const captureFrameBtn = document.getElementById('btn-capture-frame');
    if (captureFrameBtn) {
        captureFrameBtn.addEventListener('click', () => {
            sendCalibrationMessage({ type: 'capture_intrinsics_frame' });
        });
    }
    
    const computeIntrinsicsBtn = document.getElementById('btn-compute-intrinsics');
    if (computeIntrinsicsBtn) {
        computeIntrinsicsBtn.addEventListener('click', () => {
            sendCalibrationMessage({ type: 'compute_intrinsics' });
            showCalibrationMessage('Computing intrinsics...', 'info');
        });
    }
    
    // Collapsible sections
    const collapsibleHeaders = document.querySelectorAll('.collapsible-header');
    collapsibleHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const section = header.closest('.collapsible');
            section.classList.toggle('collapsed');
        });
    });
}

function toggleTestLabMode() {
    setTestLabMode(!state.testLabMode);
}

function setTestLabMode(enabled) {
    state.testLabMode = enabled;
    
    const panel = document.getElementById('test-lab-panel');
    const toggleBtn = document.getElementById('test-lab-toggle');
    
    if (enabled) {
        panel.classList.remove('hidden');
        toggleBtn.classList.add('active');
        
        // Request current calibration status (also enables marker detection on backend)
        sendCalibrationMessage({ type: 'get_calibration_status' });
        
        // Start periodic status updates while panel is open
        startCalibrationPolling();
    } else {
        panel.classList.add('hidden');
        toggleBtn.classList.remove('active');
        
        // Stop polling
        stopCalibrationPolling();
        
        // Tell backend to exit Test Lab Mode
        sendCalibrationMessage({ type: 'stop_calibration' });
    }
}

let calibrationPollInterval = null;

function startCalibrationPolling() {
    if (calibrationPollInterval) return;
    
    calibrationPollInterval = setInterval(() => {
        if (state.testLabMode && state.connected) {
            sendCalibrationMessage({ type: 'get_calibration_status' });
        }
    }, 500); // Poll every 500ms
}

function stopCalibrationPolling() {
    if (calibrationPollInterval) {
        clearInterval(calibrationPollInterval);
        calibrationPollInterval = null;
    }
}

function handleCalibrationStatus(data) {
    // Update state
    state.calibration.markersDetected = data.marker_ids_detected || [];
    state.calibration.homographyValid = data.homography_valid || false;
    state.calibration.intrinsicsValid = data.intrinsics_valid || false;
    state.calibration.undistortionEnabled = data.undistortion_enabled || false;
    
    // Update marker detection UI
    updateMarkerDetectionUI(data);
    
    // Update homography status
    updateHomographyStatusUI(data);
    
    // Update intrinsics status
    updateIntrinsicsStatusUI(data);
    
    // Update mat dimensions if provided
    if (data.mat_dimensions) {
        document.getElementById('mat-width').value = data.mat_dimensions.width_m;
        document.getElementById('mat-height').value = data.mat_dimensions.height_m;
    }
    
    // Show message or error if provided
    if (data.message) {
        showCalibrationMessage(data.message, 'success');
    } else if (data.error) {
        showCalibrationMessage(data.error, 'error');
    }
}

function updateMarkerDetectionUI(data) {
    const cornerMarkers = [0, 1, 2, 3];
    const detectedIds = data.marker_ids_detected || [];
    
    cornerMarkers.forEach(id => {
        const element = document.getElementById(`marker-${id}`);
        if (element) {
            const stateSpan = element.querySelector('.marker-state');
            const isDetected = detectedIds.includes(id);
            
            if (isDetected) {
                element.classList.add('detected');
                stateSpan.textContent = '✓';
            } else {
                element.classList.remove('detected');
                stateSpan.textContent = '--';
            }
        }
    });
    
    // Update summary
    const detectedCount = cornerMarkers.filter(id => detectedIds.includes(id)).length;
    const summarySpan = document.getElementById('markers-detected');
    if (summarySpan) {
        summarySpan.textContent = detectedCount;
    }
}

function updateHomographyStatusUI(data) {
    const statusDiv = document.getElementById('homography-status');
    if (!statusDiv) return;
    
    const indicator = statusDiv.querySelector('.status-indicator');
    const text = statusDiv.querySelector('span:last-child');
    
    if (data.homography_valid) {
        indicator.classList.add('calibrated');
        indicator.classList.remove('not-calibrated');
        text.textContent = 'Calibrated';
    } else {
        indicator.classList.remove('calibrated');
        indicator.classList.add('not-calibrated');
        text.textContent = 'Not Calibrated';
    }
}

function updateIntrinsicsStatusUI(data) {
    const statusDiv = document.getElementById('intrinsics-status');
    const toggle = document.getElementById('undistort-toggle');
    
    if (statusDiv) {
        const indicator = statusDiv.querySelector('.status-indicator');
        const text = statusDiv.querySelector('span:last-child');
        
        if (data.intrinsics_valid) {
            indicator.classList.add('calibrated');
            indicator.classList.remove('not-calibrated');
            text.textContent = 'Calibrated';
            if (toggle) toggle.disabled = false;
        } else {
            indicator.classList.remove('calibrated');
            indicator.classList.add('not-calibrated');
            text.textContent = 'Not Calibrated';
            if (toggle) toggle.disabled = true;
        }
    }
    
    if (toggle) {
        toggle.checked = data.undistortion_enabled || false;
    }
}

function handleDistanceTestResults(data) {
    const results = data.results || {};
    
    // Update each test row - markers at 30cm, 60cm, 80cm
    const testMappings = {
        '0.30m': { measured: 'test-30cm', error: 'error-30cm' },
        '0.60m': { measured: 'test-60cm', error: 'error-60cm' },
        '0.80m': { measured: 'test-80cm', error: 'error-80cm' },
    };
    
    for (const [distance, ids] of Object.entries(testMappings)) {
        const measuredEl = document.getElementById(ids.measured);
        const errorEl = document.getElementById(ids.error);
        
        if (results[distance]) {
            const result = results[distance];
            
            if (measuredEl) {
                measuredEl.textContent = result.measured.toFixed(3) + 'm';
            }
            
            if (errorEl) {
                const errorMm = result.error_mm;
                errorEl.textContent = (errorMm >= 0 ? '+' : '') + errorMm.toFixed(1) + 'mm';
                
                // Color code based on error magnitude
                errorEl.classList.remove('good', 'warning', 'bad');
                if (Math.abs(errorMm) <= 5) {
                    errorEl.classList.add('good');
                } else if (Math.abs(errorMm) <= 15) {
                    errorEl.classList.add('warning');
                } else {
                    errorEl.classList.add('bad');
                }
            }
        } else {
            if (measuredEl) measuredEl.textContent = '--';
            if (errorEl) {
                errorEl.textContent = '--';
                errorEl.classList.remove('good', 'warning', 'bad');
            }
        }
    }
    
    showCalibrationMessage('Distance tests complete', 'success');
}

function handleIntrinsicsProgress(data) {
    const framesEl = document.getElementById('intrinsics-frames');
    const computeBtn = document.getElementById('btn-compute-intrinsics');
    
    if (framesEl) {
        framesEl.textContent = data.frames_collected || 0;
    }
    
    if (computeBtn) {
        computeBtn.disabled = (data.frames_collected || 0) < 5;
    }
    
    if (data.message) {
        showCalibrationMessage(data.message, data.success ? 'success' : 'info');
    }
}

function showCalibrationMessage(message, type = 'info') {
    // Log to console
    const prefix = type === 'error' ? '❌' : type === 'success' ? '✓' : 'ℹ️';
    console.log(`[Calibration] ${prefix} ${message}`);
    
    // Show toast notification in the UI
    const existingToast = document.getElementById('calibration-toast');
    if (existingToast) {
        existingToast.remove();
    }
    
    const toast = document.createElement('div');
    toast.id = 'calibration-toast';
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 80px;
        left: 50%;
        transform: translateX(-50%);
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        z-index: 10000;
        animation: toastSlideIn 0.3s ease;
        background: ${type === 'error' ? 'rgba(255, 80, 80, 0.95)' : type === 'success' ? 'rgba(74, 222, 128, 0.95)' : 'rgba(100, 100, 100, 0.95)'};
        color: ${type === 'error' || type === 'success' ? '#000' : '#fff'};
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;
    
    document.body.appendChild(toast);
    
    // Auto-remove after delay
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, type === 'error' ? 5000 : 3000);
}

function animate() {
    requestAnimationFrame(animate);
    
    const dt = state.clock.getDelta();
    updatePhysics(dt);
    
    controls.update();
    renderer.render(scene, camera);
}

// Start
init();
