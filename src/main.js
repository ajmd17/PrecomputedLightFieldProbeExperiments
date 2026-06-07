import * as shaders from './shaders.js';

const RES = 256;
const OCT_RES = 256;
const NUM_PROBES = 8;

const info = document.getElementById('info');

// ─── WebGL 2.0 Context ───────────────────────────────────────────────
const canvas = document.getElementById('glcanvas');
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;

const gl = canvas.getContext('webgl2', { alpha: false });
if (!gl) {
  info.textContent = 'ERROR: WebGL 2.0 not supported';
  throw new Error('WebGL 2.0 required');
}
info.textContent = 'WebGL 2.0 OK';

// ─── Extension Checks ────────────────────────────────────────────────
const extFloat = gl.getExtension('EXT_color_buffer_float');
if (!extFloat) {
  info.textContent = 'WARNING: EXT_color_buffer_float missing';
}

console.log(`MAX_DRAW_BUFFERS: ${gl.getParameter(gl.MAX_DRAW_BUFFERS)}`);
console.log(`MAX_COLOR_ATTACHMENTS: ${gl.getParameter(gl.MAX_COLOR_ATTACHMENTS)}`);

// ─── Shader Compilation ──────────────────────────────────────────────
function compileShader(src, type) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}

function createProgram(vsSrc, fsSrc) {
  const vs = compileShader(vsSrc, gl.VERTEX_SHADER);
  const fs = compileShader(fsSrc, gl.FRAGMENT_SHADER);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(prog));
    return null;
  }
  return prog;
}

function uniformLoc(prog, name) {
  const loc = gl.getUniformLocation(prog, name);
  if (loc === null) console.warn(`Uniform ${name} not found in program`);
  return loc;
}

// ─── Texture Creation ────────────────────────────────────────────────
function createCubemapF16(w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);
  for (let i = 0; i < 6; i++) {
    gl.texImage2D(
      gl.TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
      gl.RGBA16F, w, h, 0,
      gl.RGBA, gl.HALF_FLOAT, null
    );
  }
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return tex;
}

function createCubemapR16F(w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);
  for (let i = 0; i < 6; i++) {
    gl.texImage2D(
      gl.TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
      gl.R16F, w, h, 0,
      gl.RED, gl.HALF_FLOAT, null
    );
  }
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return tex;
}

function createTexture2DF16(w, h, channels) {
  const ch = channels || 4;
  const ifmt = ch === 1 ? gl.R16F : gl.RGBA16F;
  const fmt  = ch === 1 ? gl.RED  : gl.RGBA;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

// ─── MRT G-Buffer Framebuffer ────────────────────────────────────────
function createGBufferFBO(w, h) {
  const radianceCM = createCubemapF16(w, h);
  const normalCM   = createCubemapF16(w, h);
  const distanceCM = createCubemapR16F(w, h);
  const fbo = gl.createFramebuffer();
  return { fbo, radianceCM, normalCM, distanceCM };
}

const drawBuf3 = [
  gl.COLOR_ATTACHMENT0,
  gl.COLOR_ATTACHMENT1,
  gl.COLOR_ATTACHMENT2,
];

function bindGBufferFace(fbo, face, radianceCM, normalCM, distanceCM) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const target = gl.TEXTURE_CUBE_MAP_POSITIVE_X + face;
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, target, radianceCM, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, target, normalCM,   0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, target, distanceCM, 0);
  gl.drawBuffers(drawBuf3);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.error(`FBO incomplete for face ${face}: ${status}`);
  }
}

// ─── Octahedral Map Framebuffer (3 MRT targets) ──────────────────────
function createOctahedralFBO(w, h) {
  const radiance2D = createTexture2DF16(w, h);
  const normal2D   = createTexture2DF16(w, h);
  const distance2D = createTexture2DF16(w, h, 1);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, radiance2D, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normal2D,   0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, distance2D, 0);
  gl.drawBuffers(drawBuf3);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) console.error('Oct FBO incomplete');
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, radiance2D, normal2D, distance2D };
}

// ─── Single-target Filter Framebuffer ────────────────────────────────
function createFilterFBO(w, h, channels) {
  const tex = createTexture2DF16(w, h, channels);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) console.error('Filter FBO incomplete');
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, tex };
}

// ─── Compile Shaders ─────────────────────────────────────────────────
const progGBuffer    = createProgram(shaders.gbufferVert, shaders.gbufferFrag);
const progOctahedral = createProgram(shaders.octahedralVert, shaders.octahedralFrag);
const progIrradiance = createProgram(shaders.octahedralVert, shaders.irradianceFilterFrag);
const progVSM        = createProgram(shaders.octahedralVert, shaders.varianceShadowFrag);

if (!progGBuffer || !progOctahedral || !progIrradiance || !progVSM) {
  info.textContent = 'ERROR: shader compilation failed';
  throw new Error('Shader compilation failed');
}
info.textContent = 'Shaders compiled OK';

// ─── Full-Screen Quad (for octahedral / filter passes) ──────────────
const quadVerts = new Float32Array([
  -1, -1,   1, -1,   1,  1,
  -1, -1,   1,  1,  -1,  1,
]);
const quadVAO = gl.createVertexArray();
gl.bindVertexArray(quadVAO);
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// ─── Probe Data ──────────────────────────────────────────────────────
const probeData = [];
for (let i = 0; i < NUM_PROBES; i++) {
  probeData.push({
    gbuf: createGBufferFBO(RES, RES),
    oct:  createOctahedralFBO(OCT_RES * 2, OCT_RES),
    irr:  createFilterFBO(OCT_RES * 2, OCT_RES),
    vsm:  createFilterFBO(OCT_RES * 2, OCT_RES, 2), // RG = depth, variance
  });
}

// ─── Scene Geometry (Cornell Box) ────────────────────────────────────
function makeBox(cx, cy, cz, sx, sy, sz, r, g, b) {
  const hx = sx / 2, hy = sy / 2, hz = sz / 2;
  const p = [
    // Order: front, back, left, right, top, bottom
    // Each face: 4 verts (CCW when viewed from outside)
    cx-hx, cy-hy, cz+hz,  cx+hx, cy-hy, cz+hz,  cx+hx, cy+hy, cz+hz,  cx-hx, cy+hy, cz+hz,
    cx+hx, cy-hy, cz-hz,  cx-hx, cy-hy, cz-hz,  cx-hx, cy+hy, cz-hz,  cx+hx, cy+hy, cz-hz,
    cx-hx, cy-hy, cz-hz,  cx-hx, cy-hy, cz+hz,  cx-hx, cy+hy, cz+hz,  cx-hx, cy+hy, cz-hz,
    cx+hx, cy-hy, cz+hz,  cx+hx, cy-hy, cz-hz,  cx+hx, cy+hy, cz-hz,  cx+hx, cy+hy, cz+hz,
    cx-hx, cy+hy, cz-hz,  cx+hx, cy+hy, cz-hz,  cx+hx, cy+hy, cz+hz,  cx-hx, cy+hy, cz+hz,
    cx-hx, cy-hy, cz+hz,  cx+hx, cy-hy, cz+hz,  cx+hx, cy-hy, cz-hz,  cx-hx, cy-hy, cz-hz,
  ];
  const n = [];
  const faceNormals = [
    [0,0,1], [0,0,-1], [-1,0,0], [1,0,0], [0,1,0], [0,-1,0]
  ];
  for (const fn of faceNormals)
    for (let j = 0; j < 4; j++)
      n.push(fn[0], fn[1], fn[2]);

  // Indices: 2 triangles per face (CCW winding)
  const idx = [];
  for (let f = 0; f < 6; f++) {
    const base = f * 4;
    idx.push(base, base+1, base+2, base, base+2, base+3);
  }
  return { pos: p, nrm: n, color: [r, g, b], idx };
}

const meshes = [
  makeBox(0, -4, 0, 10, 0.5, 10, 0.9, 0.9, 0.9),     // floor
  makeBox(-5, 0, 0, 0.5, 8, 10, 1.0, 0.2, 0.2),      // left wall (red)
  makeBox(5, 0, 0, 0.5, 8, 10, 0.2, 0.2, 1.0),       // right wall (blue)
  makeBox(0, 0, -5, 10, 8, 0.5, 0.8, 0.8, 0.8),      // back wall
  makeBox(0, 4, 0, 10, 0.5, 10, 0.6, 0.6, 0.6),      // ceiling
  makeBox(-2, -3, 0, 2, 2, 2, 0.1, 0.9, 0.1),        // green cube
  makeBox(2.5, -2.5, -2, 1.5, 3, 1.5, 1.0, 0.6, 0.1),// orange cube
  makeBox(0, 3.5, 0, 4, 0.2, 4, 5.0, 5.0, 5.0),     // light (emissive)
];

// Upload each mesh as its own VAO (self-contained, indices start at 0)
function uploadScene() {
  const list = [];
  for (const m of meshes) {
    const vertCount = m.pos.length / 3;
    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(m.pos), gl.STATIC_DRAW);

    const nrmBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(m.nrm), gl.STATIC_DRAW);

    const colData = new Float32Array(vertCount * 3);
    for (let i = 0; i < vertCount; i++) {
      colData[i*3] = m.color[0]; colData[i*3+1] = m.color[1]; colData[i*3+2] = m.color[2];
    }
    const colBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, colData, gl.STATIC_DRAW);

    const idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(m.idx), gl.STATIC_DRAW);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bindVertexArray(null);

    list.push({ vao, count: m.idx.length });
  }
  return list;
}
const sceneVAOs = uploadScene();

// ─── Math Helpers ────────────────────────────────────────────────────
function perspective(fovY, aspect, near, far) {
  const f = 1.0 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function normalize(v) { const l = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return [v[0]/l, v[1]/l, v[2]/l]; }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

function mul(a, b) {
  const r = new Float32Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      r[j*4+i] = a[i]*b[j*4] + a[4+i]*b[j*4+1] + a[8+i]*b[j*4+2] + a[12+i]*b[j*4+3];
  return r;
}

const CUBE_TARGETS = [
  [ 1, 0, 0], [-1, 0, 0],
  [ 0, 1, 0], [ 0,-1, 0],
  [ 0, 0, 1], [ 0, 0,-1],
];
const CUBE_UPS = [
  [0,-1,0], [0,-1,0],
  [0, 0,1], [0, 0,-1],
  [0,-1,0], [0,-1,0],
];

const IDENTITY = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

// ─── Render G-Buffer (cubemap with MRT) ──────────────────────────────
function renderGBuffer(probeIdx) {
  const probePos = [(probeIdx % 4) * 2.5 - 3.75, 0, Math.floor(probeIdx / 4) * 2.5 - 1.25];
  const gbuf = probeData[probeIdx].gbuf;
  const proj = perspective(Math.PI / 2, 1, 0.1, 20);

  gl.useProgram(progGBuffer);
  const uProbePos = uniformLoc(progGBuffer, 'uProbePos');
  const uViewProj = uniformLoc(progGBuffer, 'uViewProj');
  const uModel = uniformLoc(progGBuffer, 'uModel');
  gl.uniform3fv(uProbePos, probePos);

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  // No face culling — we want to capture back faces of room interior
  gl.disable(gl.CULL_FACE);

  for (let face = 0; face < 6; face++) {
    bindGBufferFace(gbuf.fbo, face, gbuf.radianceCM, gbuf.normalCM, gbuf.distanceCM);

    const view = lookAt(probePos, [
      probePos[0] + CUBE_TARGETS[face][0],
      probePos[1] + CUBE_TARGETS[face][1],
      probePos[2] + CUBE_TARGETS[face][2],
    ], CUBE_UPS[face]);

    const vp = mul(proj, view);
    gl.uniformMatrix4fv(uViewProj, false, vp);

    gl.viewport(0, 0, RES, RES);
    gl.clearColor(0, 0, 0, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    for (const mesh of sceneVAOs) {
      gl.uniformMatrix4fv(uModel, false, IDENTITY);
      gl.bindVertexArray(mesh.vao);
      gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
    }
    gl.bindVertexArray(null);
  }
}

// ─── Octahedral Mapping (cubemap → 2D) ──────────────────────────────
function convertToOctahedral(probeIdx) {
  const pd = probeData[probeIdx];
  const { radianceCM, normalCM, distanceCM } = pd.gbuf;
  const { fbo } = pd.oct;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.useProgram(progOctahedral);
  gl.viewport(0, 0, OCT_RES * 2, OCT_RES);
  gl.disable(gl.DEPTH_TEST);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, radianceCM);
  gl.uniform1i(uniformLoc(progOctahedral, 'uRadianceCubemap'), 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, normalCM);
  gl.uniform1i(uniformLoc(progOctahedral, 'uNormalCubemap'), 1);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, distanceCM);
  gl.uniform1i(uniformLoc(progOctahedral, 'uDistanceCubemap'), 2);

  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

// ─── Irradiance Filter ───────────────────────────────────────────────
function filterIrradiance(probeIdx) {
  const pd = probeData[probeIdx];
  const { radianceCM } = pd.gbuf;
  const { fbo, tex } = pd.irr;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.useProgram(progIrradiance);
  gl.viewport(0, 0, OCT_RES * 2, OCT_RES);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, radianceCM);
  gl.uniform1i(uniformLoc(progIrradiance, 'uRadianceCubemap'), 0);
  gl.uniform1i(uniformLoc(progIrradiance, 'uSampleCount'), 256);

  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

// ─── Variance Shadow Map Filter ──────────────────────────────────────
function filterVSM(probeIdx) {
  const pd = probeData[probeIdx];
  const { distanceCM } = pd.gbuf;
  const { fbo, tex } = pd.vsm;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.useProgram(progVSM);
  gl.viewport(0, 0, OCT_RES * 2, OCT_RES);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, distanceCM);
  gl.uniform1i(uniformLoc(progVSM, 'uDistanceCubemap'), 0);

  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

// ─── Precompute All Probes ───────────────────────────────────────────
function precomputeAllProbes() {
  for (let i = 0; i < probeData.length; i++) {
    renderGBuffer(i);
    convertToOctahedral(i);
    filterIrradiance(i);
    filterVSM(i);
  }
  info.textContent = `Precomputed ${probeData.length} probes. Ready.`;
}

// ─── Camera State for Debug View ────────────────────────────────────
let camAngle = 0;

// ─── Render Loop ─────────────────────────────────────────────────────
function render() {
  const w = canvas.clientWidth * devicePixelRatio;
  const h = canvas.clientHeight * devicePixelRatio;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.08, 0.08, 0.08, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Debug: draw first probe's octahedral radiance map to screen
  const pd = probeData[0];
  if (pd && pd.oct.radiance2D) {
    drawDebugTexture(pd.oct.radiance2D);
  }

  requestAnimationFrame(render);
}

// Minimal copy program for debug display
const debugVS = `#version 300 es
layout(location=0) in vec2 aPosition;
out vec2 vUV;
void main() {
  vUV = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;
const debugFS = `#version 300 es
precision highp float;
uniform sampler2D uTex;
in vec2 vUV;
out vec4 fragColor;
void main() {
  fragColor = vec4(texture(uTex, vUV).rgb, 1.0);
}`;
const progDebug = createProgram(debugVS, debugFS);

function drawDebugTexture(tex) {
  if (!progDebug) return;
  gl.useProgram(progDebug);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(uniformLoc(progDebug, 'uTex'), 0);
  gl.disable(gl.DEPTH_TEST);
  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

// ─── Boot ────────────────────────────────────────────────────────────
precomputeAllProbes();
render();
