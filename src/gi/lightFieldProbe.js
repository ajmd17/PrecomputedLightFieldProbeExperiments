import { IndirectLightingSystem } from './interface.js';

/**
 * LightFieldProbeSystem — wraps the GPU-driven light field probe pipeline.
 * Manages cubemap G-buffers, octahedral encoding, VSM filtering, and ray-march uniforms.
 *
 * Requires pre-compiled programs passed via init(gl, sceneData, programs):
 *   { pProbe, pOct, pIrr, pVSM }
 */
export class LightFieldProbeSystem extends IndirectLightingSystem {
  constructor(cfg) {
    super();
    this.RES        = cfg.res      || 128;
    this.OCT_RES    = cfg.octRes   || 256;
    this.LOW_RES    = cfg.lowRes   || 64;
    this.NX         = cfg.nx       || 4;
    this.NY         = cfg.ny       || 4;
    this.NZ         = cfg.nz       || 4;
    this.GMIN       = cfg.gMin     || [-4, -5, -3.5];
    this.GMAX       = cfg.gMax     || [4, 3, 2];
    this.MAX_PROBES = cfg.maxProbes || 256;
    this.NUM_PROBES = this.NX * this.NY * this.NZ;

    this.probePos     = [];
    this.probeOffsets = [];
    this.adjProbePos  = [];
    this.probeActive  = null;
    this.probePosFlat = null;

    this.irrArr   = null;
    this.vsmArr   = null;
    this.distHArr = null;
    this.distLArr = null;
    this.layerFBO = null;

    this.cT = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
    this.cU = [[0,-1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,-1,0],[0,-1,0]];

    this.pProbe = null;
    this.pOct   = null;
    this.pIrr   = null;
    this.pVSM   = null;
    this.singleProbe = -1;
  }

  init(gl, sceneData, programs) {
    this.pProbe = programs.pProbe;
    this.pOct   = programs.pOct;
    this.pIrr   = programs.pIrr;
    this.pVSM   = programs.pVSM;

    const { M } = sceneData;
    const { NX, NY, NZ, GMIN, GMAX, NUM_PROBES, OCT_RES, LOW_RES } = this;

    // Probe grid positions
    this.probePos = Array.from({length: NUM_PROBES}, (_, i) => {
      const ix = i % NX, iy = Math.floor(i / NX) % NY, iz = Math.floor(i / (NX * NY));
      const tx = ix / (NX - 1), ty = iy / (NY - 1), tz = iz / (NZ - 1);
      return [ GMIN[0] + tx * (GMAX[0] - GMIN[0]), GMIN[1] + ty * (GMAX[1] - GMIN[1]), GMIN[2] + tz * (GMAX[2] - GMIN[2]) ];
    });

    // Geometry push-offsets
    this._computeOffsets(gl, M);

    // Texture arrays
    const OW = OCT_RES * 2, OH = OCT_RES;
    const LW = LOW_RES * 2, LH = LOW_RES;
    this.irrArr   = _t2DA(gl, OW, OH, NUM_PROBES, 4, gl.NEAREST);
    this.vsmArr   = _t2DA(gl, OW, OH, NUM_PROBES, 2, gl.LINEAR);
    this.distHArr = _t2DA(gl, OW, OH, NUM_PROBES, 1, gl.LINEAR);
    this.distLArr = _t2DA(gl, LW, LH, NUM_PROBES, 1, gl.LINEAR);
    this.layerFBO = gl.createFramebuffer();
  }

  _computeOffsets(gl, M) {
    const { NX, NY, NZ, GMIN, GMAX, MAX_PROBES, NUM_PROBES } = this;
    const probeOffsets = [];
    const probeActive = new Float32Array(MAX_PROBES);
    let pushed = 0, inactive = 0;

    for (let i = 0; i < NUM_PROBES; i++) {
      const ix = i % NX, iy = Math.floor(i / NX) % NY, iz = Math.floor(i / (NX * NY));
      const wx = GMIN[0] + (ix / (NX - 1)) * (GMAX[0] - GMIN[0]);
      const wy = GMIN[1] + (iy / (NY - 1)) * (GMAX[1] - GMIN[1]);
      const wz = GMIN[2] + (iz / (NZ - 1)) * (GMAX[2] - GMIN[2]);

      if (_insideMesh(wx, wy, wz, M)) {
        const off = _findNearestExit(wx, wy, wz, M);
        if (off) {
          probeOffsets.push(off);
          probeActive[i] = 1.0; pushed++;
        } else {
          probeOffsets.push([0, 0, 0]);
          probeActive[i] = 0.0; inactive++;
        }
      } else {
        probeOffsets.push([0, 0, 0]);
        probeActive[i] = 1.0;
      }
    }

    this.probeOffsets = probeOffsets;
    this.probeActive = probeActive;
    this.adjProbePos = this.probePos.map((p, i) => [
      p[0] + probeOffsets[i][0], p[1] + probeOffsets[i][1], p[2] + probeOffsets[i][2],
    ]);

    const flat = new Float32Array(MAX_PROBES * 3);
    for (let i = 0; i < NUM_PROBES; i++) {
      flat[i * 3] = this.adjProbePos[i][0];
      flat[i * 3 + 1] = this.adjProbePos[i][1];
      flat[i * 3 + 2] = this.adjProbePos[i][2];
    }
    this.probePosFlat = flat;
    console.log(`[LFP] Offsets: ${pushed} pushed, ${inactive} inactive, ${NUM_PROBES - pushed - inactive} untouched`);
  }

  // ─── Precomputation ───────────────────────────────────────────────

  precompute(gl) {
    const { NUM_PROBES, RES, OCT_RES, LOW_RES, pProbe, pOct, pIrr, pVSM } = this;
    if (!pProbe) return;

    const persp  = _persp(Math.PI / 2, 1, 0.1, 100);
    const blk    = new Float32Array([0, 0, 0, 1]);
    const farD   = new Float32Array([20, 0, 0, 0]);
    const db3    = [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2];

    for (let pi = 0; pi < NUM_PROBES; pi++) {
      const gb = _allocProbeGB(gl, RES);
      const pp = this.adjProbePos[pi];

      // Cubemap G-buffer
      gl.useProgram(pProbe);
      gl.uniform3fv(_ul(gl, pProbe, 'uQ'), pp);
      gl.uniform3fv(_ul(gl, pProbe, 'uLightPos'), [0, 3.5, 0]);
      gl.uniform3fv(_ul(gl, pProbe, 'uLightCol'), [10, 10, 10]);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
      for (let f = 0; f < 6; f++) {
        _bindGBFace(gl, gb, f, db3, RES);
        const look = _lookAt(pp, [pp[0]+this.cT[f][0], pp[1]+this.cT[f][1], pp[2]+this.cT[f][2]], this.cU[f]);
        gl.uniformMatrix4fv(_ul(gl, pProbe, 'uVP'), false, _mul4(persp, look));
        gl.viewport(0, 0, RES, RES);
        gl.clearBufferfv(gl.COLOR, 0, blk);
        gl.clearBufferfv(gl.COLOR, 1, blk);
        gl.clearBufferfv(gl.COLOR, 2, farD);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        for (const m of this._sceneVAOs) {
          gl.uniformMatrix4fv(_ul(gl, pProbe, 'uM'), false, _ID);
          gl.bindVertexArray(m.vao);
          gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
          gl.bindVertexArray(null);
        }
      }

      // Octahedral high-res distance
      this._octPass(gl, pi, gb.dist, this.distHArr, pOct, gb.nrm, gb.dist);
      // Octahedral low-res distance
      {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.layerFBO);
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.distLArr, 0, pi);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.viewport(0, 0, LOW_RES*2, LOW_RES);
        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(pOct);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.dist);
        gl.uniform1i(_ul(gl, pOct, 'uRad'), 0);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.nrm);
        gl.uniform1i(_ul(gl, pOct, 'uNrm'), 1);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.dist);
        gl.uniform1i(_ul(gl, pOct, 'uDist'), 2);
        gl.bindVertexArray(this._qVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.bindVertexArray(null);
      }
      // Irradiance filter
      this._octPass(gl, pi, gb.rad, this.irrArr, pIrr);
      // VSM filter
      {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.layerFBO);
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.vsmArr, 0, pi);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.viewport(0, 0, OCT_RES*2, OCT_RES);
        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(pVSM);
        gl.uniform1f(_ul(gl, pVSM, 'uCosinePower'), 50.0);
        gl.uniform1i(_ul(gl, pVSM, 'uSampleCount'), 64);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, gb.dist);
        gl.uniform1i(_ul(gl, pVSM, 'uDist'), 0);
        gl.bindVertexArray(this._qVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.bindVertexArray(null);
      }

      _freeProbeGB(gl, gb);
    }
  }

  // Set VAOs and quad VAO from main.js during init
  setRenderResources(qVAO, sceneVAOs) {
    this._qVAO = qVAO;
    this._sceneVAOs = sceneVAOs;
  }

  _octPass(gl, layer, cm, arr, prog, nrmCM, distCM) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.layerFBO);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, arr, 0, layer);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.OCT_RES*2, this.OCT_RES);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_CUBE_MAP, cm);
    gl.uniform1i(_ul(gl, prog, 'uRad'), 0);
    if (prog === this.pOct) {
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_CUBE_MAP, nrmCM);
      gl.uniform1i(_ul(gl, prog, 'uNrm'), 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_CUBE_MAP, distCM);
      gl.uniform1i(_ul(gl, prog, 'uDist'), 2);
    }
    if (prog === this.pIrr) {
      gl.uniform1i(_ul(gl, prog, 'uN'), 1024);
      gl.uniform1f(_ul(gl, prog, 'uWrap'), 0.3);
    }
    gl.bindVertexArray(this._qVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  bindUniforms(gl, program) {
    gl.uniform3fv(_ul(gl, program, 'uProbePos'), this.probePosFlat);
    gl.uniform1fv(_ul(gl, program, 'uProbeActive'), this.probeActive);
    gl.uniform3fv(_ul(gl, program, 'uGMin'), this.GMIN);
    gl.uniform3fv(_ul(gl, program, 'uGMax'), this.GMAX);
    gl.uniform3i(_ul(gl, program, 'uGridDim'), this.NX, this.NY, this.NZ);
    gl.uniform1i(_ul(gl, program, 'uSingleProbe'), this.singleProbe);

    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.irrArr);
    gl.uniform1i(_ul(gl, program, 'uIrr'), 3);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.vsmArr);
    gl.uniform1i(_ul(gl, program, 'uVSM'), 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.distHArr);
    gl.uniform1i(_ul(gl, program, 'uDistH'), 5);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.distLArr);
    gl.uniform1i(_ul(gl, program, 'uDistL'), 6);
  }

  destroy(gl) {
    [this.irrArr, this.vsmArr, this.distHArr, this.distLArr].forEach(t => { if (t) gl.deleteTexture(t); });
    if (this.layerFBO) gl.deleteFramebuffer(this.layerFBO);
  }
}

// ─── Module-level helpers ────────────────────────────────────────────────

const _ID = new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);

function _ul(gl, p, n) { return gl.getUniformLocation(p, n); }

function _t2DA(gl, w, h, layers, ch, filter) {
  const fmt = ch === 1 ? gl.R16F : ch === 2 ? gl.RG32F : gl.RGBA16F;
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, fmt, w, h, layers);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function _allocProbeGB(gl, res) {
  const depthCM = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, depthCM);
  for (let j = 0; j < 6; j++)
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + j, 0, gl.DEPTH_COMPONENT24, res, res, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return { fbo: gl.createFramebuffer(), rad: _cmF16(gl, res, res), nrm: _cmF16(gl, res, res), dist: _cmR16F(gl, res, res), depthCM };
}

function _cmF16(gl, w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  const p = (n, v) => gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  p(gl.TEXTURE_MIN_FILTER, gl.LINEAR); p(gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  p(gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return t;
}

function _cmR16F(gl, w, h) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_CUBE_MAP, t);
  for (let i = 0; i < 6; i++) gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.R16F, w, h, 0, gl.RED, gl.HALF_FLOAT, null);
  const p = (n, v) => gl.texParameteri(gl.TEXTURE_CUBE_MAP, n, v);
  p(gl.TEXTURE_MIN_FILTER, gl.NEAREST); p(gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  p(gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); p(gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return t;
}

function _bindGBFace(gl, gb, face, db3, res) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, gb.fbo);
  const t = gl.TEXTURE_CUBE_MAP_POSITIVE_X + face;
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, gb.rad, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, t, gb.nrm, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, t, gb.dist, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, t, gb.depthCM, 0);
  gl.drawBuffers(db3);
}

function _freeProbeGB(gl, gb) {
  gl.deleteFramebuffer(gb.fbo);
  gl.deleteTexture(gb.rad); gl.deleteTexture(gb.nrm); gl.deleteTexture(gb.dist); gl.deleteTexture(gb.depthCM);
}

function _persp(fov, a, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return new Float32Array([f/a,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}

function _lookAt(e, t, u) {
  const z = _norm(_sub(e, t)), x = _norm(_cross(u, z)), y = _cross(z, x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -_dot(x,e),-_dot(y,e),-_dot(z,e),1]);
}

function _norm(v) { const l = Math.hypot(v[0], v[1], v[2]); return [v[0]/l, v[1]/l, v[2]/l]; }
function _sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function _cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function _dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function _mul4(a, b) {
  const r = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    r[j*4+i] = a[i]*b[j*4] + a[4+i]*b[j*4+1] + a[8+i]*b[j*4+2] + a[12+i]*b[j*4+3];
  return r;
}

// ── Ray-casting helpers (for geometry-push) ──────────────────────────────

function _rayDistToTri(ro, rd, v0, v1, v2) {
  const e10=v1[0]-v0[0], e11=v1[1]-v0[1], e12=v1[2]-v0[2];
  const e20=v2[0]-v0[0], e21=v2[1]-v0[1], e22=v2[2]-v0[2];
  const pv0=rd[1]*e22-rd[2]*e21, pv1=rd[2]*e20-rd[0]*e22, pv2=rd[0]*e21-rd[1]*e20;
  const det=e10*pv0+e11*pv1+e12*pv2;
  if (Math.abs(det)<1e-12) return null;
  const inv=1/det;
  const tv0=ro[0]-v0[0], tv1=ro[1]-v0[1], tv2=ro[2]-v0[2];
  const u=(tv0*pv0+tv1*pv1+tv2*pv2)*inv;
  if (u<0||u>1) return null;
  const qv0=tv1*e12-tv2*e11, qv1=tv2*e10-tv0*e12, qv2=tv0*e11-tv1*e10;
  const v=(rd[0]*qv0+rd[1]*qv1+rd[2]*qv2)*inv;
  if (v<0||u+v>1) return null;
  const t=(e20*qv0+e21*qv1+e22*qv2)*inv;
  return t>1e-8?t:null;
}

function _insideMesh(px, py, pz, meshes) {
  const ro=[px,py,pz];
  const dirs=[[1,0,0],[0,1,0],[0,0,1]];
  for (const rd of dirs) {
    let hits=0;
    for (const mesh of meshes) {
      const pos=mesh.pos, idx=mesh.idx;
      for (let ti=0; ti<idx.length; ti+=3) {
        const v0=[pos[idx[ti]*3],pos[idx[ti]*3+1],pos[idx[ti]*3+2]];
        const v1=[pos[idx[ti+1]*3],pos[idx[ti+1]*3+1],pos[idx[ti+1]*3+2]];
        const v2=[pos[idx[ti+2]*3],pos[idx[ti+2]*3+1],pos[idx[ti+2]*3+2]];
        if (_rayDistToTri(ro,rd,v0,v1,v2)!==null) hits++;
      }
    }
    if (hits%2===1) return true;
  }
  return false;
}

function _findNearestExit(px, py, pz, meshes) {
  const ro=[px,py,pz];
  const dirs=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
  let bestT=Infinity, bestD=null;
  for (const rd of dirs) {
    for (const mesh of meshes) {
      const pos=mesh.pos, idx=mesh.idx;
      for (let ti=0; ti<idx.length; ti+=3) {
        const v0=[pos[idx[ti]*3],pos[idx[ti]*3+1],pos[idx[ti]*3+2]];
        const v1=[pos[idx[ti+1]*3],pos[idx[ti+1]*3+1],pos[idx[ti+1]*3+2]];
        const v2=[pos[idx[ti+2]*3],pos[idx[ti+2]*3+1],pos[idx[ti+2]*3+2]];
        const t=_rayDistToTri(ro,rd,v0,v1,v2);
        if (t!==null&&t<bestT){bestT=t;bestD=rd;}
      }
    }
  }
  return bestD===null?null:[bestD[0]*(bestT+0.05),bestD[1]*(bestT+0.05),bestD[2]*(bestT+0.05)];
}
