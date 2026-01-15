/**
 * Putting Launch Monitor - 3D Simulator
 * 
 * Three.js based 3D putting visualization with WebSocket communication.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    // WebSocket
    wsUrl: 'ws://localhost:8765',
    wsReconnectInterval: 2000,
    
    // Physics
    friction: 0.12,              // Rolling friction coefficient
    stopVelocity: 0.01,          // Velocity threshold to stop (m/s)
    holeRadius: 0.054,           // Regulation hole radius (m)
    holeCaptureSpeed: 0.8,       // Max speed to fall in hole (m/s)
    ballRadius: 0.02135,         // Golf ball radius (m)
    
    // Simulation
    timeScale: 1.0,              // Time multiplier
    maxSimTime: 30,              // Max simulation time (seconds)
    
    // Scene
    greenWidth: 4,               // Green width (m)
    greenLength: 8,              // Green length (m)
    holeDistance: 3,             // Distance from start to hole (m)
};

// ============================================================================
// Application State
// ============================================================================

const state = {
    // Ball physics
    ball: {
        position: new THREE.Vector3(0, CONFIG.ballRadius, 0),
        velocity: new THREE.Vector2(0, 0),
        isRolling: false,
        startPosition: new THREE.Vector3(0, CONFIG.ballRadius, 0),
    },
    
    // Shot data
    lastShot: null,
    totalDistance: 0,
    
    // Connection
    ws: null,
    connected: false,
    
    // Animation
    clock: new THREE.Clock(),
};

// ============================================================================
// Three.js Setup
// ============================================================================

let scene, camera, renderer, controls;
let ballMesh, holeMesh, greenMesh, flagMesh;

function initScene() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a1a0f);
    scene.fog = new THREE.Fog(0x0a1a0f, 10, 30);
    
    // Camera
    camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        100
    );
    camera.position.set(0, 5, 6);
    camera.lookAt(0, 0, 0);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    
    document.getElementById('canvas-container').appendChild(renderer.domElement);
    
    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.1;
    controls.minDistance = 2;
    controls.maxDistance = 15;
    controls.target.set(0, 0, -1);
    
    // Lights
    setupLights();
    
    // Objects
    createGreen();
    createHole();
    createBall();
    createFlag();
    createEnvironment();
    
    // Events
    window.addEventListener('resize', onWindowResize);
    window.addEventListener('keydown', onKeyDown);
}

function setupLights() {
    // Ambient light
    const ambient = new THREE.AmbientLight(0x4a7c59, 0.5);
    scene.add(ambient);
    
    // Main directional light (sun)
    const sun = new THREE.DirectionalLight(0xfff5e6, 1.2);
    sun.position.set(5, 10, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 50;
    sun.shadow.camera.left = -10;
    sun.shadow.camera.right = 10;
    sun.shadow.camera.top = 10;
    sun.shadow.camera.bottom = -10;
    sun.shadow.bias = -0.0001;
    scene.add(sun);
    
    // Fill light
    const fill = new THREE.DirectionalLight(0x6fa8dc, 0.3);
    fill.position.set(-5, 5, -5);
    scene.add(fill);
    
    // Rim light
    const rim = new THREE.DirectionalLight(0xffd700, 0.2);
    rim.position.set(0, 2, -10);
    scene.add(rim);
}

function createGreen() {
    // Main green surface
    const greenGeometry = new THREE.PlaneGeometry(
        CONFIG.greenWidth,
        CONFIG.greenLength,
        32,
        64
    );
    
    // Add subtle undulation
    const positions = greenGeometry.attributes.position.array;
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const z = positions[i + 1];
        positions[i + 2] = Math.sin(x * 2) * Math.cos(z * 1.5) * 0.01;
    }
    greenGeometry.computeVertexNormals();
    
    // Green material with grass-like appearance
    const greenMaterial = new THREE.MeshStandardMaterial({
        color: 0x2d5a27,
        roughness: 0.9,
        metalness: 0.0,
    });
    
    greenMesh = new THREE.Mesh(greenGeometry, greenMaterial);
    greenMesh.rotation.x = -Math.PI / 2;
    greenMesh.position.y = 0;
    greenMesh.receiveShadow = true;
    scene.add(greenMesh);
    
    // Fringe/rough around green
    const fringeGeometry = new THREE.PlaneGeometry(
        CONFIG.greenWidth + 2,
        CONFIG.greenLength + 2
    );
    const fringeMaterial = new THREE.MeshStandardMaterial({
        color: 0x3d6f2f,
        roughness: 1.0,
        metalness: 0.0,
    });
    const fringe = new THREE.Mesh(fringeGeometry, fringeMaterial);
    fringe.rotation.x = -Math.PI / 2;
    fringe.position.y = -0.002;
    fringe.receiveShadow = true;
    scene.add(fringe);
}

function createHole() {
    // Hole (dark cylinder sunk into ground)
    const holeGeometry = new THREE.CylinderGeometry(
        CONFIG.holeRadius,
        CONFIG.holeRadius,
        0.1,
        32
    );
    const holeMaterial = new THREE.MeshStandardMaterial({
        color: 0x0a0a0a,
        roughness: 1.0,
        metalness: 0.0,
    });
    
    holeMesh = new THREE.Mesh(holeGeometry, holeMaterial);
    holeMesh.position.set(0, -0.05, -CONFIG.holeDistance);
    scene.add(holeMesh);
    
    // Hole rim (white edge)
    const rimGeometry = new THREE.RingGeometry(
        CONFIG.holeRadius,
        CONFIG.holeRadius + 0.005,
        32
    );
    const rimMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.5,
        metalness: 0.0,
        side: THREE.DoubleSide,
    });
    const rim = new THREE.Mesh(rimGeometry, rimMaterial);
    rim.rotation.x = -Math.PI / 2;
    rim.position.set(0, 0.001, -CONFIG.holeDistance);
    scene.add(rim);
}

function createBall() {
    // Golf ball
    const ballGeometry = new THREE.SphereGeometry(CONFIG.ballRadius, 32, 32);
    const ballMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.2,
        metalness: 0.0,
    });
    
    ballMesh = new THREE.Mesh(ballGeometry, ballMaterial);
    ballMesh.position.copy(state.ball.position);
    ballMesh.castShadow = true;
    scene.add(ballMesh);
}

function createFlag() {
    // Flag pole
    const poleGeometry = new THREE.CylinderGeometry(0.005, 0.005, 1.5, 8);
    const poleMaterial = new THREE.MeshStandardMaterial({
        color: 0xdddddd,
        roughness: 0.3,
        metalness: 0.8,
    });
    const pole = new THREE.Mesh(poleGeometry, poleMaterial);
    pole.position.set(0, 0.75, -CONFIG.holeDistance);
    pole.castShadow = true;
    scene.add(pole);
    
    // Flag
    const flagGeometry = new THREE.PlaneGeometry(0.25, 0.15);
    const flagMaterial = new THREE.MeshStandardMaterial({
        color: 0xef4444,
        roughness: 0.5,
        metalness: 0.0,
        side: THREE.DoubleSide,
    });
    flagMesh = new THREE.Mesh(flagGeometry, flagMaterial);
    flagMesh.position.set(0.125, 1.42, -CONFIG.holeDistance);
    flagMesh.castShadow = true;
    scene.add(flagMesh);
}

function createEnvironment() {
    // Ground plane extending beyond green
    const groundGeometry = new THREE.PlaneGeometry(50, 50);
    const groundMaterial = new THREE.MeshStandardMaterial({
        color: 0x2a4a1f,
        roughness: 1.0,
        metalness: 0.0,
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);
    
    // Sky dome gradient (simple)
    const skyGeometry = new THREE.SphereGeometry(40, 32, 32);
    const skyMaterial = new THREE.ShaderMaterial({
        uniforms: {
            topColor: { value: new THREE.Color(0x1a3a20) },
            bottomColor: { value: new THREE.Color(0x0a1a0f) },
        },
        vertexShader: `
            varying vec3 vWorldPosition;
            void main() {
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 topColor;
            uniform vec3 bottomColor;
            varying vec3 vWorldPosition;
            void main() {
                float h = normalize(vWorldPosition).y;
                gl_FragColor = vec4(mix(bottomColor, topColor, max(h, 0.0)), 1.0);
            }
        `,
        side: THREE.BackSide,
    });
    const sky = new THREE.Mesh(skyGeometry, skyMaterial);
    scene.add(sky);
}

// ============================================================================
// Physics & Animation
// ============================================================================

function startShot(speedMps, directionDeg) {
    // Reset ball position
    resetBall();
    
    // Convert direction from degrees to radians
    // Direction is relative to target line (forward = -Z in Three.js)
    // Positive = right of target
    const angleRad = (directionDeg * Math.PI) / 180;
    
    // Calculate velocity components
    // Forward direction is -Z in Three.js
    state.ball.velocity.x = speedMps * Math.sin(angleRad);
    state.ball.velocity.y = -speedMps * Math.cos(angleRad);  // -Z is forward
    
    state.ball.isRolling = true;
    state.totalDistance = 0;
    
    console.log(`Shot: ${speedMps.toFixed(2)} m/s, ${directionDeg.toFixed(1)}°`);
}

function updateBallPhysics(deltaTime) {
    if (!state.ball.isRolling) return;
    
    const dt = deltaTime * CONFIG.timeScale;
    
    // Get current speed
    const speed = state.ball.velocity.length();
    
    if (speed < CONFIG.stopVelocity) {
        stopBall();
        return;
    }
    
    // Apply friction (exponential decay)
    const frictionFactor = Math.exp(-CONFIG.friction * dt * 10);
    state.ball.velocity.multiplyScalar(frictionFactor);
    
    // Update position
    const dx = state.ball.velocity.x * dt;
    const dz = state.ball.velocity.y * dt;
    
    state.ball.position.x += dx;
    state.ball.position.z += dz;
    
    // Track distance
    state.totalDistance += Math.sqrt(dx * dx + dz * dz);
    
    // Update ball mesh
    ballMesh.position.copy(state.ball.position);
    
    // Rotate ball based on movement
    const rotationSpeed = speed / CONFIG.ballRadius;
    ballMesh.rotation.x -= dz / CONFIG.ballRadius;
    ballMesh.rotation.z += dx / CONFIG.ballRadius;
    
    // Check for hole
    checkHole();
    
    // Check boundaries
    checkBoundaries();
}

function checkHole() {
    const holePos = new THREE.Vector2(0, -CONFIG.holeDistance);
    const ballPos = new THREE.Vector2(state.ball.position.x, state.ball.position.z);
    
    const distance = holePos.distanceTo(ballPos);
    const speed = state.ball.velocity.length();
    
    // Ball is in if:
    // 1. Center is within hole radius
    // 2. Speed is below capture threshold
    if (distance < CONFIG.holeRadius && speed < CONFIG.holeCaptureSpeed) {
        // Ball falls in!
        ballInHole();
    } else if (distance < CONFIG.holeRadius && speed >= CONFIG.holeCaptureSpeed) {
        // Ball lips out - deflect based on entry point
        console.log("Lip out! Too fast.");
    }
}

function checkBoundaries() {
    const halfWidth = CONFIG.greenWidth / 2 + 0.5;
    const maxZ = CONFIG.greenLength / 2 + 0.5;
    const minZ = -CONFIG.greenLength / 2 - 0.5;
    
    // Stop if off green
    if (Math.abs(state.ball.position.x) > halfWidth ||
        state.ball.position.z > maxZ ||
        state.ball.position.z < minZ) {
        stopBall();
        showResult("Off Green", false);
    }
}

function ballInHole() {
    state.ball.isRolling = false;
    
    // Animate ball dropping
    const dropAnimation = () => {
        if (state.ball.position.y > -0.02) {
            state.ball.position.y -= 0.005;
            ballMesh.position.copy(state.ball.position);
            requestAnimationFrame(dropAnimation);
        } else {
            showResult("In The Hole!", true);
        }
    };
    dropAnimation();
}

function stopBall() {
    state.ball.isRolling = false;
    state.ball.velocity.set(0, 0);
    
    // Update distance display
    updateDistanceDisplay(state.totalDistance);
    
    // Check if close to hole
    const holePos = new THREE.Vector2(0, -CONFIG.holeDistance);
    const ballPos = new THREE.Vector2(state.ball.position.x, state.ball.position.z);
    const toHole = holePos.distanceTo(ballPos);
    
    if (toHole < 0.3) {
        showResult(`${(toHole * 100).toFixed(0)}cm short!`, false);
    }
}

function resetBall() {
    state.ball.position.set(0, CONFIG.ballRadius, 0);
    state.ball.velocity.set(0, 0);
    state.ball.isRolling = false;
    state.totalDistance = 0;
    
    ballMesh.position.copy(state.ball.position);
    ballMesh.rotation.set(0, 0, 0);
    
    hideResult();
}

// ============================================================================
// WebSocket Communication
// ============================================================================

function connectWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        return;
    }
    
    console.log('Connecting to WebSocket...');
    
    try {
        state.ws = new WebSocket(CONFIG.wsUrl);
        
        state.ws.onopen = () => {
            console.log('WebSocket connected');
            state.connected = true;
            updateConnectionStatus(true);
        };
        
        state.ws.onclose = () => {
            console.log('WebSocket disconnected');
            state.connected = false;
            updateConnectionStatus(false);
            
            // Reconnect after delay
            setTimeout(connectWebSocket, CONFIG.wsReconnectInterval);
        };
        
        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        state.ws.onmessage = (event) => {
            handleWebSocketMessage(event.data);
        };
        
    } catch (error) {
        console.error('WebSocket connection failed:', error);
        setTimeout(connectWebSocket, CONFIG.wsReconnectInterval);
    }
}

function handleWebSocketMessage(data) {
    try {
        const message = JSON.parse(data);
        
        if (message.type === 'connected') {
            console.log('Server:', message.message);
            return;
        }
        
        // Shot event
        if (message.speed_mps !== undefined) {
            console.log('Received shot:', message);
            state.lastShot = message;
            
            // Update stats display
            updateStatsDisplay(message);
            
            // Start the shot animation
            startShot(message.speed_mps, message.direction_deg);
        }
        
    } catch (error) {
        console.error('Failed to parse message:', error);
    }
}

// ============================================================================
// UI Updates
// ============================================================================

function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    const textEl = statusEl.querySelector('.text');
    
    if (connected) {
        statusEl.classList.remove('disconnected');
        statusEl.classList.add('connected');
        textEl.textContent = 'Connected';
    } else {
        statusEl.classList.remove('connected');
        statusEl.classList.add('disconnected');
        textEl.textContent = 'Disconnected';
    }
}

function updateStatsDisplay(shot) {
    document.getElementById('stat-speed').textContent = shot.speed_mps.toFixed(2);
    document.getElementById('stat-direction').textContent = 
        (shot.direction_deg >= 0 ? '+' : '') + shot.direction_deg.toFixed(1);
    document.getElementById('stat-confidence').textContent = 
        Math.round(shot.confidence * 100);
}

function updateDistanceDisplay(distance) {
    document.getElementById('stat-distance').textContent = distance.toFixed(2);
}

function showResult(text, isSuccess) {
    const resultEl = document.getElementById('result-message');
    const textEl = resultEl.querySelector('.text');
    
    resultEl.classList.remove('hidden', 'success', 'miss');
    resultEl.classList.add(isSuccess ? 'success' : 'miss');
    textEl.textContent = text;
    
    // Auto-hide after delay
    setTimeout(() => {
        resultEl.classList.add('hidden');
    }, 3000);
}

function hideResult() {
    document.getElementById('result-message').classList.add('hidden');
}

// ============================================================================
// Event Handlers
// ============================================================================

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function onKeyDown(event) {
    switch (event.key.toLowerCase()) {
        case 'r':
            resetBall();
            break;
        case 't':
            // Test shot
            const testSpeed = 1.5 + Math.random() * 1.0;
            const testDirection = (Math.random() - 0.5) * 10;
            
            const testShot = {
                speed_mps: testSpeed,
                direction_deg: testDirection,
                confidence: 0.9,
            };
            updateStatsDisplay(testShot);
            startShot(testSpeed, testDirection);
            break;
    }
}

// ============================================================================
// Animation Loop
// ============================================================================

function animate() {
    requestAnimationFrame(animate);
    
    const deltaTime = state.clock.getDelta();
    
    // Update physics
    updateBallPhysics(deltaTime);
    
    // Animate flag waving
    if (flagMesh) {
        flagMesh.rotation.y = Math.sin(Date.now() * 0.003) * 0.1;
    }
    
    // Update controls
    controls.update();
    
    // Render
    renderer.render(scene, camera);
}

// ============================================================================
// Initialization
// ============================================================================

function init() {
    console.log('Putting Launch Monitor - 3D Simulator');
    console.log('Initializing...');
    
    initScene();
    connectWebSocket();
    animate();
    
    console.log('Ready!');
    console.log('Press T for test shot, R to reset ball');
}

// Start application
init();
