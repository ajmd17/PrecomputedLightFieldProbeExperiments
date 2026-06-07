import { IndirectLightingSystem } from './interface.js';

/**
 * TetrahedralProbeSystem — CPU tetrahedral search + SH interpolation
 * based on Robert Cupisz's "Light Probe Interpolation Using Tetrahedral
 * Tessellations" (GDC 2012).
 *
 * Each frame, for each dynamic object, finds the enclosing tetrahedron
 * via a neighbour-walk search, computes barycentric coordinates from a
 * precomputed 3x3 matrix, and interpolates the 4 corner probes' SH.
 *
 * Usage:
 *   const gi = new TetrahedralProbeSystem();
 *   gi.init(gl, sceneData);
 *   // ... each frame for each object ...
 *   gi.interpolateAt(objectPos);   // CPU — returns { sh, tetIndex, bary }
 *   gi.bindInterpolated(gl, program, 'uSH');  // upload as 9×vec3 uniform
 */

// ─── SH helpers ─────────────────────────────────────────────────────────
// 3-band RGB SH = 27 floats
const SH_FLOATS = 27;

function zeroSH() {
  return new Float32Array(SH_FLOATS);
}

/** Copy RGB SH from an array (ordered R0..R8, G0..G8, B0..B8) */
function copySH(src) {
  const dst = new Float32Array(SH_FLOATS);
  dst.set(src);
  return dst;
}

/** Add src multiplied by weight into dst (dst += src * w) */
function accumulateSH(dst, src, w) {
  for (let k = 0; k < SH_FLOATS; k++) dst[k] += src[k] * w;
}

/** 3×3 matrix inverse (for precomputed tet matrices) */
function inv3x3(m) {
  // m = [e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z]
  const [a,b,c, d,e,f, g,h,i] = m;
  const det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
  if (Math.abs(det) < 1e-12) return null;
  const inv = 1/det;
  return [
    (e*i-f*h)*inv, -(b*i-c*h)*inv,  (b*f-c*e)*inv,
    -(d*i-f*g)*inv,  (a*i-c*g)*inv, -(a*f-c*d)*inv,
    (d*h-e*g)*inv, -(a*h-b*g)*inv,  (a*e-b*d)*inv,
  ];
}

/**
 * Precompute the tet matrix M such that:
 *   [λ₁, λ₂, λ₃] = M · (P - v0)
 *   λ₀ = 1 - λ₁ - λ₂ - λ₃
 *
 * M is the inverse of the 3×3 matrix whose columns are (v1-v0, v2-v0, v3-v0).
 */
function computeTetMatrix(v0, v1, v2, v3) {
  const e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
  const e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
  const e3 = [v3[0]-v0[0], v3[1]-v0[1], v3[2]-v0[2]];
  // Matrix is column-major: cols are e1, e2, e3
  const mat = [e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2]];
  return inv3x3(mat);
}

/**
 * Compute barycentric coordinates using precomputed tet matrix.
 * Returns [λ₀, λ₁, λ₂, λ₃] or null if degenerate.
 */
function baryFromMatrix(P, v0, M) {
  const dx = P[0] - v0[0], dy = P[1] - v0[1], dz = P[2] - v0[2];
  const l1 = M[0]*dx + M[3]*dy + M[6]*dz;   // row 0 · (P-v0)
  const l2 = M[1]*dx + M[4]*dy + M[7]*dz;   // row 1 · (P-v0)
  const l3 = M[2]*dx + M[5]*dy + M[8]*dz;   // row 2 · (P-v0)
  const l0 = 1 - l1 - l2 - l3;
  return [l0, l1, l2, l3];
}

// ═══════════════════════════════════════════════════════════════════════════
//  TETRAHEDRAL PROBE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

export class TetrahedralProbeSystem extends IndirectLightingSystem {
  constructor() {
    super();

    /** @type {{pos:number[], sh:Float32Array}[]} */
    this.probes = [];

    /**
     * @type {{
     *   v:number[],       // 4 vertex indices
     *   n:number[],       // 4 neighbour indices (same pos as opposite vertex)
     *   M:number[]|null,  // precomputed 3×3 matrix (9 floats)
     * }[]}
     */
    this.tets = [];

    // Cache for temporal coherence
    this._tetIndex   = 0;
    this._tetDirty   = true;  // reset to 0 on first query

    // For showing which tet we're in
    this.lastBary   = null;
    this.lastTetIdx = -1;
  }

  // ─── Initialisation ────────────────────────────────────────────────────

  init(gl, sceneData) {
    this._buildTestData();
    console.log(`[TET] ${this.probes.length} probes, ${this.tets.length} tetrahedra`);
  }

  /**
   * Build 8 probes at box corners + 6 tetrahedra filling the box.
   *
   * The box spans the central scene volume:
   *   x: [-3.5, 3.5]   y: [-4, 3]   z: [-2.5, 2.5]
   *
   * Probes:
   *   0: (-3.5, -4, -2.5)  1: ( 3.5, -4, -2.5)
   *   2: (-3.5,  3, -2.5)  3: ( 3.5,  3, -2.5)
   *   4: (-3.5, -4,  2.5)  5: ( 3.5, -4,  2.5)
   *   6: (-3.5,  3,  2.5)  7: ( 3.5,  3,  2.5)
   *
   * 6 tetrahedra (standard diagonal decomposition):
   *   Each tet connects a triangular face of the box to the opposite
   *   corner along the diagonal (0→7).
   */
  _buildTestData() {
    const P = [
      [-3.5, -4, -2.5],
      [ 3.5, -4, -2.5],
      [-3.5,  3, -2.5],
      [ 3.5,  3, -2.5],
      [-3.5, -4,  2.5],
      [ 3.5, -4,  2.5],
      [-3.5,  3,  2.5],
      [ 3.5,  3,  2.5],
    ];

    // SH colours: natural indirect gradient — warmer near ceiling
    // (light bounce), cooler at back, neutral at center.
    // Each probe gets a distinct colour so interpolation is visible
    // across the volume.
    const cols = [
      [0.20, 0.25, 0.35],  // 0: cool dark (bottom-left-front)
      [0.20, 0.25, 0.35],  // 1: cool dark (bottom-right-front)
      [0.45, 0.40, 0.30],  // 2: warm (top-left-front)
      [0.45, 0.40, 0.30],  // 3: warm (top-right-front)
      [0.15, 0.20, 0.40],  // 4: cool blue (bottom-left-back)
      [0.15, 0.20, 0.40],  // 5: cool blue (bottom-right-back)
      [0.35, 0.35, 0.45],  // 6: neutral-warm (top-left-back)
      [0.35, 0.35, 0.45],  // 7: neutral-warm (top-right-back)
    ];

    this.probes = P.map((pos, i) => {
      const [r, g, b] = cols[i];
      const sh = zeroSH();
      const dc = 1.0 / 0.282095;
      sh[0] = r * dc;   // band 0 R
      sh[1] = g * dc;   // band 0 G
      sh[2] = b * dc;   // band 0 B
      return { pos, sh };
    });

    // 6 tetrahedra — standard diagonal decomposition (diagonal 0→7)
    const tetDefs = [
      { v: [0, 1, 3, 7] },
      { v: [0, 1, 5, 7] },
      { v: [0, 2, 3, 7] },
      { v: [0, 2, 6, 7] },
      { v: [0, 4, 5, 7] },
      { v: [0, 4, 6, 7] },
    ];

    // Build neighbour adjacency:
    //   Tet 0 (v:0,1,2,4) shares face (0,1,2) with no-one; shares (0,1,4) with tet1;
    //   vertex 2 is not in tet1 → neighbour[2] = 1
    //   vertex 3 is not in tet0 → neighbour of tet1 at position of v=3 = 0
    // We need to figure out which vertex is opposite each face.

    // Helper: for each tet, find which neighbour shares the face opposite vertex k
    const findOpposite = (tets, ti, k) => {
      const face = tets[ti].v.filter((_, i) => i !== k);  // 3 vertices of the face
      for (let ni = 0; ni < tets.length; ni++) {
        if (ni === ti) continue;
        const ov = tets[ni].v;
        // Check if all 3 face vertices are in this neighbour
        if (face.every(v => ov.includes(v))) return ni;
      }
      return -1;
    };

    this.tets = tetDefs.map((d, i) => {
      const n = d.v.map((_, k) => findOpposite(tetDefs, i, k));
      const v0 = P[d.v[0]], v1 = P[d.v[1]], v2 = P[d.v[2]], v3 = P[d.v[3]];
      const M = computeTetMatrix(v0, v1, v2, v3);
      return { v: d.v, n, M };
    });
  }

  // ─── Tetrahedral search (CPU, every frame) ────────────────────────────

  /**
   * Find the tetrahedron containing world-space position P.
   *
   * Uses a neighbour-walk starting from the cached tetIndex for temporal
   * coherence.  Returns the tet index, or -1 if outside the entire mesh.
   *
   * Algorithm:
   *   1. Compute barycentric coords via precomputed matrix
   *   2. If all λ ≥ 0 → inside; return
   *   3. Otherwise, step through the neighbour of the most-negative λ
   *   4. Repeat (bounded by max iteration = num tets)
   */
  findTetrahedron(P) {
    const { tets } = this;
    if (tets.length === 0) return -1;

    // Start from cached index (or 0 on first call)
    let ti = this._tetDirty ? 0 : this._tetIndex;
    this._tetDirty = false;

    const maxIter = tets.length + 1;
    for (let iter = 0; iter < maxIter; iter++) {
      const tet = tets[ti];
      if (!tet.M) { ti = (ti + 1) % tets.length; continue; }

      const v0 = this.probes[tet.v[0]].pos;
      const bary = baryFromMatrix(P, v0, tet.M);

      let inside = true;
      let worst = 0;       // most-negative value
      let worstK = -1;     // which bary coord

      for (let k = 0; k < 4; k++) {
        if (bary[k] < -1e-7) {  // small epsilon to avoid oscillation
          inside = false;
          if (bary[k] < worst) { worst = bary[k]; worstK = k; }
        }
      }

      if (inside) {
        // Cache and return
        this._tetIndex = ti;
        this.lastTetIdx = ti;
        this.lastBary = bary;
        return ti;
      }

      // Walk to neighbour across the face opposite worstK
      const next = tet.n[worstK];
      if (next < 0 || next >= tets.length) {
        // Boundary hit — clamp to nearest face
        // Return -1 for now; caller decides fallback
        this.lastTetIdx = -1;
        this.lastBary = null;
        return -1;
      }
      ti = next;
    }

    // Should not reach here (mesh is disconnected or malformed)
    console.warn('[TET] Search exceeded max iterations');
    this.lastTetIdx = -1;
    this.lastBary = null;
    return -1;
  }

  // ─── SH interpolation ──────────────────────────────────────────────────

  /**
   * Interpolate SH at world-space position P.
   *
   * Finds the enclosing tetrahedron, computes barycentric weights,
   * and accumulates the 4 corner probes' SH.
   *
   * Returns { sh: Float32Array(27), tetIndex: number, bary: number[]|null }
   * On failure, sh is zeros and tetIndex is -1.
   */
  interpolateAt(P) {
    const tetIdx = this.findTetrahedron(P);

    if (tetIdx < 0) {
      // Outside mesh — fallback: nearest-neighbour or zero
      return { sh: zeroSH(), tetIndex: -1, bary: null };
    }

    const tet = this.tets[tetIdx];
    const v0 = this.probes[tet.v[0]].pos;
    const bary = baryFromMatrix(P, v0, tet.M);

    // Accumulate SH from 4 corner probes
    const sh = zeroSH();
    for (let k = 0; k < 4; k++) {
      accumulateSH(sh, this.probes[tet.v[k]].sh, bary[k]);
    }

    return { sh, tetIndex: tetIdx, bary };
  }

  // ─── GL binding ────────────────────────────────────────────────────────

  /**
   * Upload a single interpolated SH array as two uniforms:
   *   uSH — vec3[9] (9 RGB coefficients)
   *
   * The companion shader can evaluate:
   *   vec3 evalSH(vec3 dir) { ... }
   */
  bindInterpolated(gl, program, shArray) {
    // Upload 9×vec3 — GLSL expects:
    //   uniform vec3 uSH[9];
    // Data layout: R0,G0,B0,  R1,G1,B1,  ..., R8,G8,B8
    gl.uniform3fv(gl.getUniformLocation(program, 'uSH'), shArray);
  }

  // ─── Precomputation (render cubemaps + project to SH) ────────────────

  /**
   * Render cubemaps from each probe position and project radiance into
   * SH coefficients using the GPU.
   *
   * ctx: { gl, pProbe, pProject, sceneVAOs, qVAO, lightPos, lightCol, res }
   */
  precompute(gl, ctx) {
    const { pProbe, pProject, sceneVAOs, qVAO, res } = ctx;
    const lPos = ctx.lightPos || [0, 3.5, 0];
    const lCol = ctx.lightCol || [10, 10, 10];
    const sampleCount = ctx.sampleCount || 1024;

    const persp = _persp(Math.PI/2, 1, 0.1, 20);
    const blk = new Float32Array([0,0,0,1]);
    const farD = new Float32Array([20,0,0,0]);
    const cT = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
    const cU = [[0,-1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,-1,0],[0,-1,0]];
    const ID = new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);

    // Output texture: 9×1 RGBA32F (one pixel per SH band)
    const outTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, outTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 9, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const outFBO = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

    const readBuf = new Float32Array(36);

    for (let pi = 0; pi < this.probes.length; pi++) {
      const pp = this.probes[pi].pos;

      if (ctx.info) ctx.info.textContent = `Tet probe ${pi+1}/${this.probes.length}...`;

      // Render cubemap from this probe position
      const gb = _allocCM(gl, res);
      gl.useProgram(pProbe);
      gl.uniform3fv(gl.getUniformLocation(pProbe, 'uQ'), pp);
      gl.uniform3fv(gl.getUniformLocation(pProbe, 'uLightPos'), lPos);
      gl.uniform3fv(gl.getUniformLocation(pProbe, 'uLightCol'), lCol);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);

      for (let f = 0; f < 6; f++) {
        _bindCMFace(gl, gb, f);
        const vp = _mul4(persp, _lookAt(pp, [pp[0]+cT[f][0],pp[1]+cT[f][1],pp[2]+cT[f][2]], cU[f]));
        gl.uniformMatrix4fv(gl.getUniformLocation(pProbe, 'uVP'), false, vp);
        gl.viewport(0, 0, res, res);
        gl.clearBufferfv(gl.COLOR, 0, blk);
        gl.clearBufferfv(gl.COLOR, 1, blk);
        gl.clearBufferfv(gl.COLOR, 2, farD);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        for (const m of sceneVAOs) {
          gl.uniformMatrix4fv(gl.getUniformLocation(pProbe,'uM'), false, ID);
          gl.bindVertexArray(m.vao);
          gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
          gl.bindVertexArray(null);
        }
      }

      // Project cubemap to SH coefficients
      gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
      gl.viewport(0, 0, 9, 1);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(pProject);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.rad);
      gl.uniform1i(gl.getUniformLocation(pProject, 'uRad'), 0);
      gl.uniform1i(gl.getUniformLocation(pProject, 'uSampleCount'), sampleCount);
      gl.bindVertexArray(qVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);

      // Read back SH coefficients
      gl.readPixels(0, 0, 9, 1, gl.RGBA, gl.FLOAT, readBuf);

      // Convert readback to interleaved SH layout
      const sh = zeroSH();
      for (let b = 0; b < 9; b++) {
        sh[b*3]   = readBuf[b*4];     // R
        sh[b*3+1] = readBuf[b*4+1];   // G
        sh[b*3+2] = readBuf[b*4+2];   // B
      }
      this.probes[pi].sh = sh;

      _freeCM(gl, gb);
    }

    // Cleanup
    gl.deleteFramebuffer(outFBO);
    gl.deleteTexture(outTex);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    if (ctx.info) ctx.info.textContent = `Tet SH done.`;
    console.log(`[TET] Precomputed SH for ${this.probes.length} probes`);
  }

  bindUniforms(gl, program, camera) {
    // Tetrahedral system doesn't bind global uniforms for raymarching;
    // per-object SH is bound via bindInterpolated() instead.
  }

  destroy(gl) {
    // No GPU resources
  }

  // ─── Debug ─────────────────────────────────────────────────────────────

  /**
   * Return the cached tet index for the last interpolateAt() call.
   * Useful for overlay / debug rendering.
   */
  getDebugInfo() {
    return {
      tetIndex: this.lastTetIdx,
      bary: this.lastBary,
      numTets: this.tets.length,
      numProbes: this.probes.length,
    };
  }
}

// ─── Cubemap rendering helpers ──────────────────────────────────────────

function _allocCM(gl, res) {
  const depth = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, depth);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.DEPTH_COMPONENT24, res, res, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  const rad = _cmF16(gl, res, res);
  const nrm = _cmF16(gl, res, res);
  const dist = _cmR16F(gl, res, res);
  return { fbo, rad, nrm, dist, depth };
}
function _cmF16(gl, w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  for (const [n,v] of [[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  return t;
}
function _cmR16F(gl, w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, gl.R16F, w, h, 0, gl.RED, gl.HALF_FLOAT, null);
  for (const [n,v] of [[gl.TEXTURE_MIN_FILTER,gl.NEAREST],[gl.TEXTURE_MAG_FILTER,gl.NEAREST],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  return t;
}
function _bindCMFace(gl, gb, face) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, gb.fbo);
  const t = gl.TEXTURE_CUBE_MAP_POSITIVE_X+face;
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, gb.rad, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, t, gb.nrm, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, t, gb.dist, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, t, gb.depth, 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
}
function _freeCM(gl, gb) {
  gl.deleteFramebuffer(gb.fbo);
  gl.deleteTexture(gb.rad); gl.deleteTexture(gb.nrm); gl.deleteTexture(gb.dist); gl.deleteTexture(gb.depth);
}
function _persp(fov,a,near,far) {
  const f=1/Math.tan(fov/2), nf=1/(near-far);
  return new Float32Array([f/a,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function _lookAt(e,t,u) {
  const z=_norm(_sub(e,t)), x=_norm(_cross(u,z)), y=_cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -_dot(x,e),-_dot(y,e),-_dot(z,e),1]);
}
function _norm(v) { const l=Math.hypot(v[0],v[1],v[2]); return [v[0]/l,v[1]/l,v[2]/l]; }
function _sub(a,b) { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function _cross(a,b) { return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]; }
function _dot(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function _mul4(a,b) {
  const r=new Float32Array(16);
  for(let i=0;i<4;i++) for(let j=0;j<4;j++) r[j*4+i]=a[i]*b[j*4]+a[4+i]*b[j*4+1]+a[8+i]*b[j*4+2]+a[12+i]*b[j*4+3];
  return r;
}
